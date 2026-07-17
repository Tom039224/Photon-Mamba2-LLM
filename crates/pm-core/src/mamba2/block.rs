//! Mamba2 block — `Module<O>` impl.
//!
//! Faithful re-implementation of the official `Mamba2` block (Dao & Gu 2024,
//! §6) with PHOTON's Tenstorrent-friendly constraints baked in:
//! - A is scalar-per-head (§6.2): one log-parameter per head.
//! - B, C share parameters across heads when `n_groups < n_heads`
//!   (Multi-Value Attention / MIS pattern, §6.3).
//!
//! Forward (B = batch, T = sequence, D = `d_model`, H = `n_heads`,
//! P = `d_head`, N = `d_state`, G = `n_groups`):
//!
//! ```text
//!   xzd                = x @ W_in
//!   z, xBC, dt_raw     = split(xzd, [d_inner, xBC, H], dim=-1)
//!   xBC                = silu(depthwise_conv1d_causal(xBC))
//!   x_ssm, B, C        = split(xBC, [d_inner, G*N, G*N], dim=-1)
//!   dt                 = softplus(dt_raw + dt_bias)
//!   A                  = -exp(a_log)
//!   y                  = ssd_scan(x_ssm * dt, A * dt, B, C)
//!   y                 += D * x_ssm                                # skip
//!   out                = rmsnorm(y * silu(z)) @ W_out
//! ```
//!
//! ### Mixed-precision ("fp32 islands", memory-efficiency plan Phase A2)
//!
//! `forward` is dtype-polymorphic: it runs in whatever dtype `x` (the
//! ambient "compute dtype", `cdt`) arrives in — `F32` by default,
//! optionally `BF16` (`PhotonMamba::compute_dtype`, `docs/perf-log.md`
//! 2026-07-03). The big `(B,T,·)` activations and the `in_proj`/
//! `out_proj` matmuls run in `cdt` (that is where the activation-tape
//! memory saving comes from — fp32-stored `Param`s are cast to `cdt`
//! inline at each matmul). A handful of numerically-sensitive
//! sub-computations are always run in `F32` regardless of `cdt`
//! ("islands"): `softplus(dt)`, `exp(a_log)`, the depthwise `conv1d`,
//! `rmsnorm`, and the entire `ssd_scan` call (its internal cumsum +
//! exp of `A`-cumulative differences under/overflows in bf16 — see
//! `mamba2::ssd_ops` module docs). Every island upcasts its inputs via
//! `Ops::to_dtype` and downcasts the result back to `cdt` before
//! rejoining the ambient flow. `to_dtype` is a no-op when the source is
//! already the target dtype, so with `cdt = F32` (the default) every
//! cast in this function is a genuine no-op and the forward is
//! bit-identical to the pre-Phase-A2 implementation.

use crate::mamba2::state::Mamba2State;
use crate::{Dtype, Module, Ops, Param, Parameterized, Tensor};

#[derive(Debug, Clone)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub d_state: usize, // N
    pub d_head: usize,  // P
    pub n_heads: usize,
    pub n_groups: usize,  // must be 1 or n_heads (Phase 1 simplification)
    pub d_conv: usize,    // typically 4
    pub block_len: usize, // SSD chunk size Q (paper default 64)
    pub rmsnorm_eps: f32,
}

impl Mamba2Config {
    #[must_use]
    pub const fn d_inner(&self) -> usize {
        self.n_heads * self.d_head
    }
    #[must_use]
    pub const fn xbc_dim(&self) -> usize {
        self.d_inner() + 2 * self.n_groups * self.d_state
    }
    #[must_use]
    pub const fn in_proj_dim(&self) -> usize {
        self.d_inner() + self.xbc_dim() + self.n_heads
    }
}

/// Mamba2 block, parameterised over the backend.
///
/// All trainable weights are stored as `O::Param`s so the optimiser
/// reaches them via `Parameterized::append_params`.
pub struct Mamba2Block<O: Ops> {
    pub config: Mamba2Config,
    pub in_proj_weight: O::Param,
    pub conv1d_weight: O::Param,
    pub conv1d_bias: O::Param,
    pub a_log: O::Param,
    pub d_skip: O::Param,
    pub dt_bias: O::Param,
    pub norm_weight: O::Param,
    pub out_proj_weight: O::Param,
}

impl<O: Ops> Mamba2Block<O> {
    /// Construct a block where every weight tensor is filled with the
    /// same scalar, except `conv1d_bias`, `dt_bias`, `a_log`, `d_skip`
    /// (zero) and `norm_weight` (one). For smoke tests and grad checks.
    pub fn from_constants(
        ops: &O,
        config: Mamba2Config,
        weight_scale: f32,
    ) -> Result<Self, O::Error> {
        let mk = |shape: &[usize], val: f32| -> Result<O::Param, O::Error> {
            let n: usize = shape.iter().product();
            ops.param_from_slice_f32(&vec![val; n], shape)
        };
        Ok(Self {
            in_proj_weight: mk(&[config.d_model, config.in_proj_dim()], weight_scale)?,
            conv1d_weight: mk(&[config.xbc_dim(), 1, config.d_conv], weight_scale)?,
            conv1d_bias: mk(&[config.xbc_dim()], 0.0)?,
            a_log: mk(&[config.n_heads], 0.0)?,
            d_skip: mk(&[config.n_heads], 0.0)?,
            dt_bias: mk(&[config.n_heads], 0.0)?,
            norm_weight: mk(&[config.d_inner()], 1.0)?,
            out_proj_weight: mk(&[config.d_inner(), config.d_model], weight_scale)?,
            config,
        })
    }
}

impl<O: Ops> Module<O> for Mamba2Block<O> {
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 3, "Mamba2Block::forward expects (B,T,D)");
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let cfg = &self.config;
        assert_eq!(
            d, cfg.d_model,
            "d_model mismatch: got {d}, expected {}",
            cfg.d_model
        );
        assert!(
            cfg.n_groups == 1 || cfg.n_groups == cfg.n_heads,
            "Mamba2Block (Phase 1): n_groups must be 1 or n_heads"
        );

        // Ambient compute dtype ("cdt", see module docs). `F32` by
        // default; every `to_dtype` call below is then a documented
        // no-op and this function is bit-identical to the pre-Phase-A2
        // implementation.
        let cdt = x.dtype();

        let d_inner = cfg.d_inner();
        let xbc_dim = cfg.xbc_dim();
        let gn = cfg.n_groups * cfg.d_state;

        let in_proj_w = ops.to_dtype(self.in_proj_weight.as_tensor(), cdt)?;
        let xzd = ops.matmul(x, &in_proj_w)?;
        let z = ops.narrow(&xzd, 2, 0, d_inner)?;
        let xbc = ops.narrow(&xzd, 2, d_inner, xbc_dim)?;
        let dt_raw = ops.narrow(&xzd, 2, d_inner + xbc_dim, cfg.n_heads)?;

        // conv1d island: kept fp32 for v1 (small tensor; see module docs).
        let xbc_chw = ops.transpose(&xbc, 1, 2)?;
        let xbc_chw_f32 = ops.to_dtype(&xbc_chw, Dtype::F32)?;
        let xbc_conv_f32 = ops.conv1d(
            &xbc_chw_f32,
            self.conv1d_weight.as_tensor(),
            Some(self.conv1d_bias.as_tensor()),
            1,
            cfg.d_conv - 1,
            xbc_dim,
        )?;
        let xbc_conv = ops.to_dtype(&xbc_conv_f32, cdt)?;
        let xbc_conv = ops.narrow(&xbc_conv, 2, 0, t)?;
        let xbc = ops.transpose(&xbc_conv, 1, 2)?;
        let xbc = ops.silu(&xbc)?;

        let x_ssm = ops.narrow(&xbc, 2, 0, d_inner)?;
        let b_ssm = ops.narrow(&xbc, 2, d_inner, gn)?;
        let c_ssm = ops.narrow(&xbc, 2, d_inner + gn, gn)?;
        let x_ssm_4d = ops.reshape(&x_ssm, &[b, t, cfg.n_heads, cfg.d_head])?;
        let b_4d = ops.reshape(&b_ssm, &[b, t, cfg.n_groups, cfg.d_state])?;
        let c_4d = ops.reshape(&c_ssm, &[b, t, cfg.n_groups, cfg.d_state])?;

        let (b_4d, c_4d) = if cfg.n_groups == cfg.n_heads {
            (b_4d, c_4d)
        } else {
            let target = [b, t, cfg.n_heads, cfg.d_state];
            (
                ops.broadcast_as(&b_4d, &target)?,
                ops.broadcast_as(&c_4d, &target)?,
            )
        };

        // dt island: softplus(dt_raw + dt_bias). `dt_bias` is a fp32-
        // native Param; `dt_raw` is upcast to match it.
        let dt_raw_f32 = ops.to_dtype(&dt_raw, Dtype::F32)?;
        let dt_bias_r = ops.reshape(self.dt_bias.as_tensor(), &[1, 1, cfg.n_heads])?;
        let dt_pre_f32 = ops.add(&dt_raw_f32, &dt_bias_r)?;
        let dt_f32 = ops.softplus(&dt_pre_f32)?;
        let dt = ops.to_dtype(&dt_f32, cdt)?;

        // A = -exp(a_log) island. `a_log` is already fp32-native; the
        // *output* is downcast to `cdt` so it can multiply `dt` below.
        let a_pos = ops.exp(self.a_log.as_tensor())?;
        let a_neg = ops.neg(&a_pos)?;
        let a_neg_cdt = ops.to_dtype(&a_neg, cdt)?;
        let a_r = ops.reshape(&a_neg_cdt, &[1, 1, cfg.n_heads])?;
        let a_bth = ops.mul(&a_r, &dt)?;

        let dt_4d = ops.reshape(&dt, &[b, t, cfg.n_heads, 1])?;
        let x_dt = ops.mul(&x_ssm_4d, &dt_4d)?;

        // ssd_scan island: the internal cumsum + exp of `A`-cumulative
        // differences under/overflows in bf16 (mamba2::ssd_ops docs).
        let x_dt_f32 = ops.to_dtype(&x_dt, Dtype::F32)?;
        let a_bth_f32 = ops.to_dtype(&a_bth, Dtype::F32)?;
        let b_4d_f32 = ops.to_dtype(&b_4d, Dtype::F32)?;
        let c_4d_f32 = ops.to_dtype(&c_4d, Dtype::F32)?;
        let y_f32 = ops.ssd_scan(&x_dt_f32, &a_bth_f32, &b_4d_f32, &c_4d_f32, cfg.block_len)?;
        let y = ops.to_dtype(&y_f32, cdt)?;

        let d_skip_cdt = ops.to_dtype(self.d_skip.as_tensor(), cdt)?;
        let d_r = ops.reshape(&d_skip_cdt, &[1, 1, cfg.n_heads, 1])?;
        let d_term = ops.mul(&d_r, &x_ssm_4d)?;
        let y = ops.add(&y, &d_term)?;
        let y = ops.reshape(&y, &[b, t, d_inner])?;

        let z_silu = ops.silu(&z)?;
        let gated = ops.mul(&y, &z_silu)?;

        // rmsnorm island.
        let gated_f32 = ops.to_dtype(&gated, Dtype::F32)?;
        let normed_f32 = ops.rmsnorm(&gated_f32, self.norm_weight.as_tensor(), cfg.rmsnorm_eps)?;
        let normed = ops.to_dtype(&normed_f32, cdt)?;

        let out_proj_w = ops.to_dtype(self.out_proj_weight.as_tensor(), cdt)?;
        ops.matmul(&normed, &out_proj_w)
    }
}

// -------- Phase C: O(1)-memory recurrent decode --------

impl<O: Ops> Mamba2Block<O>
where
    O::Tensor: Clone,
{
    /// One-token step through this block's SSD recurrence.
    ///
    /// Specialises [`Module::forward`]'s formula to `T = 1`, replacing
    /// the chunked SSD scan with one exact update of a *carried*
    /// recurrent state: the causal `conv1d` becomes a rolling
    /// `d_conv`-wide window ([`Mamba2State::conv_window`]), and the SSM
    /// scan collapses to Mamba2's fundamental recurrence (§2.1 Eq.
    /// `h_t = A h_{t-1} + B x_t`, `y_t = Cᵀh_t`) with Mamba2's
    /// discretisation `Ā = exp(A·dt)`, `B̄ = dt·B` — the same
    /// `x_dt = x_ssm · dt` pre-scaling `forward` feeds into
    /// `Ops::ssd_scan` above (this file, `x_dt` a few lines up), rather
    /// than scaling `B` directly (equivalent: the outer product
    /// commutes with the scalar `dt`).
    ///
    /// **This is exact, not an approximation.** Instantiate
    /// `ssd_scan_chunked`'s per-chunk recurrence (`mamba2::ssd_ops`
    /// module docs, `ssd_ops.rs:150-166`) at chunk length `Q = 1`:
    /// with a single intra-chunk position `q = 0`, `A_cum[c,0] =
    /// a[c,0]` (cumsum of one element), so `decay_to_end =
    /// exp(A_cum_end − A_cum[c,0]) = exp(0) = 1`, giving `h_end[c] =
    /// decay·h_end[c−1] + B[c,0] ⊗ x_dt[c,0]` — exactly step 6's
    /// `h_new` below (`x_dt = dt · x_ssm`, so the outer product `B ⊗
    /// x_dt` is `dt · (B ⊗ x_ssm)`, i.e. `dt · bx`). And `y[c,0] =
    /// y_intra + y_inter = (C·B)·x_dt[c,0] + decay·(C·h_end[c−1])`,
    /// which equals `C · h_new` directly — no separate same-timestep
    /// term is needed, because contracting `C` into the outer product
    /// distributes: `C · (dt · (B ⊗ x_ssm)) = dt · (C·B) · x_ssm =
    /// (C·B) · x_dt = y_intra`, and `C · (decay · h_old) = decay ·
    /// (C·h_old) = y_inter`. So `y = matmul(C, h_new)` alone reproduces
    /// the chunked scan bit-for-bit at `Q = 1`; `pm-candle/tests/
    /// decode_state.rs::step_parity_matches_forward` checks this
    /// end-to-end against `forward` itself.
    ///
    /// Functional / non-mutating: `state` is borrowed, a fresh
    /// [`Mamba2State`] is returned. This lets callers try extra
    /// "provisional" steps (`PhotonMamba::predict_logits`'s
    /// pad-completion) without touching the persisted state.
    ///
    /// The returned state is detached from the autograd graph (see
    /// [`detach`], below) — `y_t` (this call's *output*, consumed only
    /// within the current token's forward) is left attached; its own
    /// graph depth is bounded by the layer count, not the generation
    /// length.
    pub fn step(
        &self,
        ops: &O,
        x_t: &O::Tensor,
        state: &Mamba2State<O>,
    ) -> Result<(O::Tensor, Mamba2State<O>), O::Error> {
        let shape = x_t.shape();
        assert_eq!(shape.len(), 3, "Mamba2Block::step expects (B,1,D)");
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        assert_eq!(t, 1, "Mamba2Block::step: x_t must have T=1 (got T={t})");
        let cfg = &self.config;
        assert_eq!(
            d, cfg.d_model,
            "d_model mismatch: got {d}, expected {}",
            cfg.d_model
        );
        assert!(
            cfg.n_groups == 1 || cfg.n_groups == cfg.n_heads,
            "Mamba2Block (Phase 1): n_groups must be 1 or n_heads"
        );

        // Ambient compute dtype ("cdt"), same convention as `forward`.
        let cdt = x_t.dtype();

        let d_inner = cfg.d_inner();
        let xbc_dim = cfg.xbc_dim();
        let gn = cfg.n_groups * cfg.d_state;
        let (h, p, n) = (cfg.n_heads, cfg.d_head, cfg.d_state);

        // 1. in_proj.
        let in_proj_w = ops.to_dtype(self.in_proj_weight.as_tensor(), cdt)?;
        let xzd = ops.matmul(x_t, &in_proj_w)?; // (B,1,in_proj_dim)
        let z = ops.narrow(&xzd, 2, 0, d_inner)?; // (B,1,d_inner)
        let xbc = ops.narrow(&xzd, 2, d_inner, xbc_dim)?; // (B,1,xbc_dim)
        let dt_raw = ops.narrow(&xzd, 2, d_inner + xbc_dim, h)?; // (B,1,H)

        // 2. Rolling causal conv1d (fp32 island). `forward` pads the
        // whole sequence with `d_conv - 1` zeros then crops; here we
        // instead concat this token's pre-conv column onto the last
        // `d_conv - 1` columns and run one `padding=0` conv, producing
        // exactly the same causal window one token at a time.
        let xbc_col = ops.to_dtype(&ops.transpose(&xbc, 1, 2)?, Dtype::F32)?; // (B,xbc_dim,1)
        let win = ops.concat(&[&state.conv_window, &xbc_col], 2)?; // (B,xbc_dim,d_conv)
        let conv = ops.conv1d(
            &win,
            self.conv1d_weight.as_tensor(),
            Some(self.conv1d_bias.as_tensor()),
            1,
            0,
            xbc_dim,
        )?; // (B,xbc_dim,1)
        let xbc_act = ops.silu(&ops.to_dtype(&ops.transpose(&conv, 1, 2)?, cdt)?)?; // (B,1,xbc_dim)
                                                                                    // Drop the oldest column, keep the newest `d_conv - 1` (FIFO).
        let conv_window_next = ops.narrow(&win, 2, 1, cfg.d_conv - 1)?; // (B,xbc_dim,d_conv-1) F32

        // 3. Split x_ssm / B / C; broadcast B, C across heads when
        // sharing a single group (MIS pattern, deviations M.1).
        let x_ssm = ops.narrow(&xbc_act, 2, 0, d_inner)?; // (B,1,d_inner)
        let b_ssm = ops.narrow(&xbc_act, 2, d_inner, gn)?; // (B,1,gn)
        let c_ssm = ops.narrow(&xbc_act, 2, d_inner + gn, gn)?; // (B,1,gn)
        let x_ssm_4d = ops.reshape(&x_ssm, &[b, 1, h, p])?; // (B,1,H,P)
        let b_4d = ops.reshape(&b_ssm, &[b, 1, cfg.n_groups, n])?; // (B,1,G,N)
        let c_4d = ops.reshape(&c_ssm, &[b, 1, cfg.n_groups, n])?; // (B,1,G,N)
        let (b_4d, c_4d) = if cfg.n_groups == h {
            (b_4d, c_4d)
        } else {
            let target = [b, 1, h, n];
            (
                ops.broadcast_as(&b_4d, &target)?,
                ops.broadcast_as(&c_4d, &target)?,
            )
        };

        // 4. dt island: softplus(dt_raw + dt_bias).
        let dt_raw_f32 = ops.to_dtype(&dt_raw, Dtype::F32)?; // (B,1,H)
        let dt_bias_r = ops.reshape(self.dt_bias.as_tensor(), &[1, 1, h])?;
        let dt_pre_f32 = ops.add(&dt_raw_f32, &dt_bias_r)?;
        let dt_f32 = ops.softplus(&dt_pre_f32)?; // (B,1,H) F32

        // 5. A = -exp(a_log); decay = exp(A·dt) (discretised Ā).
        let a_neg_f32 = ops.neg(&ops.exp(self.a_log.as_tensor())?)?; // (H,) F32
        let a_r_f32 = ops.reshape(&a_neg_f32, &[1, 1, h])?;
        let a_dt_f32 = ops.mul(&a_r_f32, &dt_f32)?; // (B,1,H) F32 = A·dt
        let decay_f32 = ops.exp(&a_dt_f32)?; // (B,1,H) F32

        // 6. SSM one-step (fp32 island) — LOAD-BEARING, see the
        // derivation in this method's docstring.
        let x_ssm_f32 = ops.to_dtype(&x_ssm_4d, Dtype::F32)?; // (B,1,H,P)
        let b_f32 = ops.to_dtype(&b_4d, Dtype::F32)?; // (B,1,H,N)
        let c_f32 = ops.to_dtype(&c_4d, Dtype::F32)?; // (B,1,H,N)
        let x_ssm_3d = ops.reshape(&x_ssm_f32, &[b, h, p])?; // (B,H,P)
        let b_3d = ops.reshape(&b_f32, &[b, h, n])?; // (B,H,N)
        let c_3d = ops.reshape(&c_f32, &[b, h, n])?; // (B,H,N)
        let dt_3d = ops.reshape(&dt_f32, &[b, h])?; // (B,H)
        let decay_3d = ops.reshape(&decay_f32, &[b, h])?; // (B,H)

        // bx = B ⊗ x_ssm, (B,H,N,P) outer product.
        let b_col = ops.broadcast_as(&ops.reshape(&b_3d, &[b, h, n, 1])?, &[b, h, n, p])?;
        let x_row = ops.broadcast_as(&ops.reshape(&x_ssm_3d, &[b, h, 1, p])?, &[b, h, n, p])?;
        let bx = ops.mul(&b_col, &x_row)?; // (B,H,N,P)

        // h_new = decay ⊙ h_old + dt ⊙ bx.
        let decay_bc = ops.broadcast_as(&ops.reshape(&decay_3d, &[b, h, 1, 1])?, &[b, h, n, p])?;
        let dt_bc = ops.broadcast_as(&ops.reshape(&dt_3d, &[b, h, 1, 1])?, &[b, h, n, p])?;
        let h_decayed = ops.mul(&decay_bc, &state.ssm_state)?;
        let h_input = ops.mul(&dt_bc, &bx)?;
        let h_new = ops.add(&h_decayed, &h_input)?; // (B,H,N,P) F32 — h_t

        // y = C · h_new, contracting the state dim N.
        let c_row = ops.reshape(&c_3d, &[b, h, 1, n])?; // (B,H,1,N)
        let y_bh1p = ops.matmul(&c_row, &h_new)?; // (B,H,1,P)
        let y_3d_f32 = ops.reshape(&y_bh1p, &[b, h, p])?; // (B,H,P)
        let y_4d = ops.reshape(&ops.to_dtype(&y_3d_f32, cdt)?, &[b, 1, h, p])?; // (B,1,H,P) cdt

        // 7. D-skip (raw x_ssm, not dt-scaled — matches `forward`).
        let d_skip_cdt = ops.to_dtype(self.d_skip.as_tensor(), cdt)?;
        let d_r = ops.reshape(&d_skip_cdt, &[1, 1, h, 1])?;
        let d_term = ops.mul(&d_r, &x_ssm_4d)?; // (B,1,H,P)
        let y = ops.add(&y_4d, &d_term)?; // (B,1,H,P)
        let y = ops.reshape(&y, &[b, 1, d_inner])?; // (B,1,d_inner)

        // 8. gated rmsnorm island.
        let z_silu = ops.silu(&z)?;
        let gated = ops.mul(&y, &z_silu)?;
        let gated_f32 = ops.to_dtype(&gated, Dtype::F32)?;
        let normed_f32 = ops.rmsnorm(&gated_f32, self.norm_weight.as_tensor(), cfg.rmsnorm_eps)?;
        let normed = ops.to_dtype(&normed_f32, cdt)?;

        // 9. out_proj.
        let out_proj_w = ops.to_dtype(self.out_proj_weight.as_tensor(), cdt)?;
        let y_t = ops.matmul(&normed, &out_proj_w)?; // (B,1,D)

        Ok((
            y_t,
            Mamba2State {
                ssm_state: detach(ops, &h_new)?,
                conv_window: detach(ops, &conv_window_next)?,
            },
        ))
    }
}

/// Cut `t`'s autograd lineage via a fresh [`Ops::param_from_tensor`]
/// boundary — the same mechanism `checkpoint.rs::forward_checkpointed`
/// uses to stop a segment's intermediates from staying reachable
/// (`param_from_tensor`'s own docs: "introduce a stable id at segment
/// boundaries"). Needed here because [`Mamba2State`] tensors are
/// carried across *every* `step()` call in a generation: without
/// cutting the graph, each step would extend the previous one's
/// backward graph by one more link, so a 4096-token decode would keep
/// ~4096 steps' worth of activations reachable — precisely the `O(T)`
/// memory Phase C exists to avoid. `to_dtype` is *not* a substitute:
/// its contract requires it be a genuine no-op (same tensor, same
/// graph) whenever the dtype already matches, which is always true
/// here (`Mamba2State` is always `F32`).
fn detach<O: Ops>(ops: &O, t: &O::Tensor) -> Result<O::Tensor, O::Error>
where
    O::Tensor: Clone,
{
    Ok(ops.param_from_tensor(t)?.as_tensor().clone())
}

impl<O: Ops> Parameterized<O> for Mamba2Block<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        out.push(&self.in_proj_weight);
        out.push(&self.conv1d_weight);
        out.push(&self.conv1d_bias);
        out.push(&self.a_log);
        out.push(&self.d_skip);
        out.push(&self.dt_bias);
        out.push(&self.norm_weight);
        out.push(&self.out_proj_weight);
    }
}
