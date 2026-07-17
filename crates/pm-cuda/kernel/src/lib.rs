//! pm-cuda-kernel — PTX kernels for the pm-cuda host runtime.
//!
//! Compiled with `--target nvptx64-nvidia-cuda` via pm-cuda/build.rs.
//! Kernels in this crate are launched from `crates/pm-cuda/src/`.

#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::arch::{asm, nvptx};

/// `expf(x)` implemented via PTX hardware `ex2.approx.f32`.
///
/// Background: the NVPTX LLVM backend has no libcall for
/// `llvm.exp.f32`, and `__nv_expf` from libdevice is left as an
/// extern reference that the CUDA driver will not auto-resolve when
/// loading the bare PTX. The hardware `ex2.approx.f32` is ≤ 2 ULP
/// accurate on Blackwell — far inside the 1e-4 parity budget the
/// J.2 reference test uses.
#[inline(always)]
fn expf(x: f32) -> f32 {
    const LOG2_E: f32 = 1.442_695_040_888_963_4_f32;
    let y = x * LOG2_E;
    let result: f32;
    unsafe {
        asm!(
            "ex2.approx.f32 {r}, {y};",
            r = out(reg32) result,
            y = in(reg32) y,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Upper bound on `n_dim` (= d_state). 128 covers the 100M target
/// config (d_state = 128) and below.
const N_MAX: usize = 128;

/// Upper bound on `block_len` (= Q, intra-chunk size). 128 covers
/// every Mamba2 Q we plan to use; production runs use Q = 64.
const Q_MAX: usize = 128;

// ---- B4.1 elementwise kernels -----------------------------------------
//
// All take `(a, b, c, n)` (binary) or `(a, c, n)` (unary), with
// contiguous row-major inputs of length `n`. Broadcasting is the
// host's job; these are the lowest level. Grid is 1-D.

#[inline(always)]
fn elem_idx() -> u32 {
    (unsafe { nvptx::_block_idx_x() * nvptx::_block_dim_x() + nvptx::_thread_idx_x() }) as u32
}

/// `c[i] = a[i] + b[i]`. B4.1.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn add_f32(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) + *b.add(i) };
    }
}

/// `c[i] = a[i] - b[i]`. B4.1.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sub_f32(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) - *b.add(i) };
    }
}

/// `c[i] = -a[i]`. B4.1.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn neg_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = -*a.add(i) };
    }
}

// ---- B4.2b elementwise kernels ----------------------------------------

/// `sqrt` via PTX `sqrt.approx.f32`. B4.2b.
#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    let result: f32;
    unsafe {
        asm!(
            "sqrt.approx.f32 {r}, {x};",
            r = out(reg32) result,
            x = in(reg32) x,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Natural log via PTX `lg2.approx.f32` × ln(2). B4.2b.
#[inline(always)]
fn lnf(x: f32) -> f32 {
    const LN2: f32 = 0.693_147_180_559_945_f32;
    let lg2: f32;
    unsafe {
        asm!(
            "lg2.approx.f32 {r}, {x};",
            r = out(reg32) lg2,
            x = in(reg32) x,
            options(pure, nomem, nostack),
        );
    }
    lg2 * LN2
}

/// `c[i] = exp(a[i])`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exp_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = expf(*a.add(i)) };
    }
}

/// `c[i] = sqrt(a[i])`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sqrt_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = sqrtf(*a.add(i)) };
    }
}

/// `c[i] = a[i] / b[i]`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn div_f32(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) / *b.add(i) };
    }
}

/// `c[i] = a[i] * scalar`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn mul_scalar_f32(a: *const f32, scalar: f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) * scalar };
    }
}

/// `c[i] = a[i] + scalar`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn add_scalar_f32(a: *const f32, scalar: f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) + scalar };
    }
}

/// `c[i] = sigmoid(a[i]) = 1 / (1 + exp(-a[i]))`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sigmoid_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        let x = unsafe { *a.add(i) };
        // 1 / (1 + exp(-x)); rcp.approx is safe here as denominator ≥ 1.
        let denom = 1.0_f32 + expf(-x);
        let result: f32;
        unsafe {
            asm!(
                "rcp.approx.f32 {r}, {d};",
                r = out(reg32) result,
                d = in(reg32) denom,
                options(pure, nomem, nostack),
            );
        }
        unsafe { *c.add(i) = result };
    }
}

/// `c[i] = silu(a[i]) = a[i] * sigmoid(a[i])`. B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn silu_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        let x = unsafe { *a.add(i) };
        let denom = 1.0_f32 + expf(-x);
        let sig: f32;
        unsafe {
            asm!(
                "rcp.approx.f32 {r}, {d};",
                r = out(reg32) sig,
                d = in(reg32) denom,
                options(pure, nomem, nostack),
            );
        }
        unsafe { *c.add(i) = x * sig };
    }
}

/// `c[i] = softplus(a[i]) = log(1 + exp(a[i]))`.
///
/// Numerically stable form: `max(x, 0) + log(1 + exp(-|x|))`.
/// B4.2b.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn softplus_f32(a: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        let x = unsafe { *a.add(i) };
        // relu(x) = max(x, 0)
        let relu_x = if x > 0.0_f32 { x } else { 0.0_f32 };
        // log1p(exp(-|x|)) = ln(1 + exp(-|x|))
        let neg_abs = if x < 0.0_f32 { x } else { -x };
        let log1p = lnf(1.0_f32 + expf(neg_abs));
        unsafe { *c.add(i) = relu_x + log1p };
    }
}

/// Elementwise f32 multiply — `c[i] = a[i] * b[i]`. I.3 smoke kernel,
/// also the B4.1 mul. (Kept under the original `mul` symbol so the
/// existing smoke example keeps working.)
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn mul(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
    let idx = elem_idx();
    if idx < n {
        let i = idx as usize;
        unsafe { *c.add(i) = *a.add(i) * *b.add(i) };
    }
}

/// Mamba2 chunked SSD scan — fused forward kernel (PLAN J.1).
///
/// Computes the recurrence
/// ```text
///   y[b,t,h,p] = Σ_{s≤t} exp(A_cum[b,t,h] − A_cum[b,s,h])
///                       · (C[b,t,h,:] · B[b,s,h,:])
///                       · x[b,s,h,p]
/// ```
/// using the chunked decomposition from Mamba2 Listing 1 (Dao & Gu 2024):
///
///   y[c,q] = y_intra[c,q] + y_inter[c,q]
///   y_intra[c,q] = Σ_{q'≤q} exp(A_cum[c,q]−A_cum[c,q']) · (C·B) · x
///   y_inter[c,q] = exp(A_cum[c,q]) · C[c,q,:] · h[c−1]
///   h[c]         = exp(A_cum_end[c]) · h[c−1]
///                + Σ_{q'} exp(A_cum_end[c]−A_cum[c,q']) · B[c,q'] · x[c,q']ᵀ
///
/// Parallelism: one thread per `(b, h, p)` tuple. Each thread carries
/// its own column of the SSM state `h[h, :, p]` in registers
/// (`[f32; N_MAX]`) and walks chunks sequentially. This makes the
/// inter-chunk dependency a register-only update — no shared-memory
/// barriers, no global atomics, no autograd tape blowup. Performance
/// optimisations (block-cooperative bc table, tensor-core matmuls,
/// register tiling) belong to J.3 and later.
///
/// # Layout (row-major, last axis fastest)
/// - `x_ptr`:  `(B, T, H, P)`
/// - `a_ptr`:  `(B, T, H)`             (scalar-per-head SSM, typically ≤ 0)
/// - `b_ptr`:  `(B, T, H, N)`
/// - `c_ptr`:  `(B, T, H, N)`
/// - `y_ptr`:  `(B, T, H, P)` — written, must be pre-allocated
///
/// # Constraints
/// - `n_dim ≤ N_MAX` (128) and `block_len ≤ Q_MAX` (128). The host
///   must check these before launching.
/// - `t_len % block_len == 0`.
/// - Total threads launched ≥ `B * H * P`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ssd_scan_chunked(
    x_ptr: *const f32,
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *const f32,
    y_ptr: *mut f32,
    batch: u32,
    t_len: u32,
    n_heads: u32,
    p_dim: u32,
    n_dim: u32,
    block_len: u32,
) {
    let tid =
        unsafe { nvptx::_block_idx_x() * nvptx::_block_dim_x() + nvptx::_thread_idx_x() } as u32;

    let p = tid % p_dim;
    let h = (tid / p_dim) % n_heads;
    let b = tid / (p_dim * n_heads);
    if b >= batch {
        return;
    }

    let b_us = b as usize;
    let h_us = h as usize;
    let p_us = p as usize;
    let t_len_us = t_len as usize;
    let n_heads_us = n_heads as usize;
    let p_dim_us = p_dim as usize;
    let n_dim_us = n_dim as usize;
    let q = block_len as usize;
    let n_chunks = t_len_us / q;

    let mut hstate = [0f32; N_MAX];
    let mut a_cum = [0f32; Q_MAX];

    for c_idx in 0..n_chunks {
        // ---- 1. inclusive prefix sum of A over the chunk ----
        let mut acc = 0f32;
        for q_idx in 0..q {
            let global_t = c_idx * q + q_idx;
            let a_val = unsafe { *a_ptr.add((b_us * t_len_us + global_t) * n_heads_us + h_us) };
            acc += a_val;
            a_cum[q_idx] = acc;
        }
        let a_cum_end = acc;

        // ---- 2. per-position output: y_intra + y_inter ----
        for q_idx in 0..q {
            let global_t = c_idx * q + q_idx;
            let a_cum_t = a_cum[q_idx];

            let mut y_acc = 0f32;

            // intra-chunk: y_intra[q] = Σ_{q'≤q} exp(A_t−A_s) (C·B) x_s
            let c_row_base = (((b_us * t_len_us + global_t) * n_heads_us) + h_us) * n_dim_us;
            for qprime in 0..=q_idx {
                let s_t = c_idx * q + qprime;
                let decay = expf(a_cum_t - a_cum[qprime]);
                let b_row_base = (((b_us * t_len_us + s_t) * n_heads_us) + h_us) * n_dim_us;
                let mut bc = 0f32;
                for n_idx in 0..n_dim_us {
                    let c_val = unsafe { *c_ptr.add(c_row_base + n_idx) };
                    let b_val = unsafe { *b_ptr.add(b_row_base + n_idx) };
                    bc += c_val * b_val;
                }
                let x_val = unsafe {
                    *x_ptr.add((((b_us * t_len_us + s_t) * n_heads_us) + h_us) * p_dim_us + p_us)
                };
                y_acc += decay * bc * x_val;
            }

            // inter-chunk: y_inter[q] = exp(A_cum[q]) C[q,:] · h
            let decay_from_start = expf(a_cum_t);
            for n_idx in 0..n_dim_us {
                let c_val = unsafe { *c_ptr.add(c_row_base + n_idx) };
                y_acc += decay_from_start * c_val * hstate[n_idx];
            }

            unsafe {
                *y_ptr
                    .add((((b_us * t_len_us + global_t) * n_heads_us) + h_us) * p_dim_us + p_us) =
                    y_acc;
            }
        }

        // ---- 3. carry h to next chunk ----
        // h ← exp(A_cum_end) · h + Σ_{q'} exp(A_cum_end − A_cum[q']) B[q',n] x[q',p]
        let decay_full = expf(a_cum_end);
        for n_idx in 0..n_dim_us {
            let mut new_h = hstate[n_idx] * decay_full;
            for qprime in 0..q {
                let s_t = c_idx * q + qprime;
                let decay_to_end = expf(a_cum_end - a_cum[qprime]);
                let b_val = unsafe {
                    *b_ptr.add((((b_us * t_len_us + s_t) * n_heads_us) + h_us) * n_dim_us + n_idx)
                };
                let x_val = unsafe {
                    *x_ptr.add((((b_us * t_len_us + s_t) * n_heads_us) + h_us) * p_dim_us + p_us)
                };
                new_h += decay_to_end * b_val * x_val;
            }
            hstate[n_idx] = new_h;
        }
    }
}

// ---- J.3.P2  SSD scan — coop B/C load + shared hstate + full decay caches ---
//
// Optimizations vs the naive J.1 kernel (1 thread per (b,h,p)):
//
// 1. **Grid redesign**: one block per (b, h), P=64 threads per block (one per p).
//    Eliminates the 1040-byte/thread local-memory spill from the J.1 kernel.
//
// 2. **Cooperative B/C load** (per chunk): all 64 threads load Q×N B and C
//    matrices into shared memory, eliminating P=64× redundant DRAM reads.
//
// 3. **bc row cache + decay cache** (per q_idx): each thread computes ONE
//    bc dot product `C[q_idx,:]·B[tid,:]` and ONE expf, eliminating P=64×
//    redundant computation.  Double-buffered across adjacent q_idx pairs.
//
// 4. **Register x cache** (step 3): pre-loads Q=64 x values into registers,
//    eliminating 2080 strided global reads per chunk per thread.
//
// 5. **Hstate decay cache**: once per chunk, thread `tid` computes ONE
//    `expf(a_cum_end - a_cum[tid])`, replacing Q×N=8192 expf calls per thread.
//
// 6. **Inter-chunk decay cache** (NEW): s_decay_start[Q] = expf(a_cum[q_idx])
//    precomputed at step 2b.  Step 3 reads from shared instead of calling expf
//    64 times per thread per chunk.
//
// 7. **Bank-conflict padding**: s_hstate uses stride N_PAD = N+1 = 129 to
//    eliminate 64-way bank conflicts (N=128 maps all threads to the same bank).
//
// 8. **Tiled hstate update** (TILE=4): step 4's qprime loop is outer so all
//    64 threads' x reads are fully coalesced.
//
// Compile-time constants (production shape only):
//   N  = 128  (d_state)
//   Q  = 64   (chunk length)
//   P  = 64   (p_dim, one thread per p)
//   N_PAD = 129 (bank-conflict padding)
//
// Shared memory layout (100 352 bytes ≈ 98 KB):
//   offset          0..Q         : s_a_cum[Q]          (64 floats =     256 B)
//   offset          Q..Q+Q*N     : s_b[Q*N]            (8192 floats = 32768 B)
//   offset    Q+Q*N..Q+2*Q*N    : s_c[Q*N]            (8192 floats = 32768 B)
//   offset  Q+2*Q*N..+P*N_PAD   : s_hstate[P*N_PAD]   (8256 floats = 33024 B)
//   HSTATE_END = Q+2*Q*N+P*N_PAD = 24704
//   offset HSTATE_END..+Q        : s_decay_end[Q]      (64 floats =     256 B)
//   offset HSTATE_END+Q..+2Q     : s_bc_row_A[Q]       (64 floats =     256 B)
//   offset HSTATE_END+2Q..+3Q    : s_bc_row_B[Q]       (64 floats =     256 B)
//   offset HSTATE_END+3Q..+4Q    : s_decay_row_A[Q]    (64 floats =     256 B)
//   offset HSTATE_END+4Q..+5Q    : s_decay_row_B[Q]    (64 floats =     256 B)
//   offset HSTATE_END+5Q..+6Q    : s_decay_start[Q]    (64 floats =     256 B)
//
// Total = (HSTATE_END + 6*Q) * 4 = 25088 * 4 = 100 352 bytes
// RTX 5070 sharedMemPerBlockOptin = 101 376 B > 100 352 B.  Fits.
// Requires cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 100352).
//
// N=128 is a compile-time const so LLVM can unroll all inner loops.

// Module-level PTX: declare the dynamic shared-memory region at module scope.
//
// `global_asm!` emits raw PTX text at module level (outside any function),
// which is the only valid position for `.extern .shared` in PTX ISA.
// Inline `asm!` inside a function body would place the declaration inside
// the function, which PTX assembler rejects with a syntax error.
//
// SAFETY: This is a PTX-level declaration with no side effects.  All
// accesses happen through `shared_mem_base()` which reads this symbol.
core::arch::global_asm!(".extern .shared .b8 SHARED_BUF[];");

/// Returns the base address of the kernel's dynamic shared memory region
/// as a raw generic-address `*mut f32` pointer.
///
/// `SHARED_BUF` is declared at module level as `.extern .shared` (via
/// `global_asm!` above).  The PTX `cvta.shared.u64` instruction converts
/// the shared-space address to a generic address that Rust's pointer
/// arithmetic can use with plain loads and stores (the PTX runtime routes
/// them back to shared memory).
///
/// The host must request the required bytes in `LaunchConfig::shared_mem_bytes`
/// before launching the kernel.
///
/// SAFETY:
/// - Must only be called from inside a `ptx-kernel` function on a device.
/// - The returned pointer is valid only for the duration of the kernel launch.
/// - All accesses must stay within `[0, shared_mem_bytes)`.
#[inline(always)]
unsafe fn shared_mem_base() -> *mut f32 {
    let ptr: u64;
    // SAFETY: `SHARED_BUF` was declared as `.extern .shared .b8` at module
    // level (via `global_asm!`).  `cvta.shared.u64` converts the raw
    // shared-space address to a generic address.  `nomem` is not set so the
    // compiler cannot hoist this before the syncthreads that precede it.
    unsafe {
        asm!(
            "cvta.shared.u64 {ptr}, SHARED_BUF;",
            ptr = out(reg64) ptr,
            options(nostack),
        );
    }
    ptr as *mut f32
}

/// Inline barrier wrapping PTX `bar.sync 0;`.
///
/// SAFETY: must be called by **all** threads in the block
/// simultaneously.
#[inline(always)]
unsafe fn syncthreads() {
    // SAFETY: `bar.sync 0` is the block-level barrier.  All threads in
    // the block must reach this call — guaranteed by the surrounding
    // code structure (no early-exit inside chunk loops).
    unsafe {
        asm!("bar.sync 0;", options(nostack));
    }
}

/// Mamba2 chunked SSD scan — P2 coop B/C load + shared hstate + bc/decay caches (J.3.P2).
///
/// Supersedes the P1 kernel for the production shape
/// (B=4, T=2048, H=12, P=64, N=128, Q=64).
///
/// **Grid**: `(batch * n_heads, 1, 1)` — one block per (b, h).
/// **Block**: `(P_PER_BLOCK, 1, 1)` — one thread per p-index.
///
/// ## Key improvements over P1 (naive 1-thread-per-(b,h,p))
///
/// 1. **Cooperative B/C load**: 64 threads load Q×N B and C from global to
///    shared, reducing DRAM traffic by P=64×.
/// 2. **bc row cache** (double-buffered): per q_idx, thread `tid` computes ONE
///    bc value (`C[q_idx,:]·B[tid,:]`), eliminating P=64× redundant dot products.
/// 3. **Intra-chunk decay cache** (double-buffered): per q_idx, thread `tid`
///    computes ONE `expf(a_cum_t - a_cum[tid])`, eliminating P=64× expf calls.
/// 4. **Register x cache**: thread `tid` pre-loads Q=64 x values into registers
///    before the q_idx loop, eliminating 2080 strided global reads per chunk.
/// 5. **Hstate decay cache**: once per chunk, thread `tid` computes ONE
///    `expf(a_cum_end - a_cum[tid])`, reducing step 4 expf cost from Q×N to 1.
/// 6. **Inter-chunk decay cache**: `s_decay_start[Q]` holds `expf(a_cum[q_idx])`
///    precomputed at step 2b, replacing Q expf calls per thread with 1 expf.
/// 7. **Bank-conflict padding for hstate**: `N_PAD = N+1 = 129` eliminates
///    64-way bank conflicts (unpadded N=128 maps every p-thread to the same bank).
/// 8. **Tiled hstate update** (TILE=4): step 4 loops over N in 4-element tiles
///    with qprime as the inner loop, making x-reads fully coalesced.
/// 9. **N=128 compile-time const**: LLVM can fully unroll all inner loops.
///
/// ## Shared memory layout (100 352 bytes ≈ 98 KB)
/// ```text
///   [0..Q)                 s_a_cum[Q]          — thread 0 writes, all read
///   [Q..Q+Q*N)             s_b[Q*N]            — cooperative load
///   [Q+Q*N..Q+2*Q*N)       s_c[Q*N]            — cooperative load
///   [Q+2*Q*N..+P*N_PAD)    s_hstate[P*N_PAD]   — each thread owns [tid*N_PAD..)
///   [HSTATE_END..+Q)       s_decay_end[Q]      — hstate decay cache (step 4)
///   [HSTATE_END+Q..+2Q)    s_bc_row_A[Q]       — double-buffered bc row
///   [HSTATE_END+2Q..+3Q)   s_bc_row_B[Q]       — double-buffered bc row
///   [HSTATE_END+3Q..+4Q)   s_decay_row_A[Q]    — double-buffered intra decay
///   [HSTATE_END+4Q..+5Q)   s_decay_row_B[Q]    — double-buffered intra decay
///   [HSTATE_END+5Q..+6Q)   s_decay_start[Q]    — inter-chunk decay cache
/// ```
/// HSTATE_END = Q + 2*Q*N + P*N_PAD = 64 + 16384 + 8256 = 24704 floats
/// Total = (HSTATE_END + 6*Q) × 4 = (24704 + 384) × 4 = 100 352 bytes.
/// RTX 5070 sharedMemPerBlockOptin = 101 376 B > 100 352 B. Fits.
///
/// ## Layout (row-major, last axis fastest)
/// Same as `ssd_scan_chunked`:
/// - `x_ptr`:  `(B, T, H, P)`
/// - `a_ptr`:  `(B, T, H)`
/// - `b_ptr`:  `(B, T, H, N)`
/// - `c_ptr`:  `(B, T, H, N)`
/// - `y_ptr`:  `(B, T, H, P)` — written, pre-allocated
///
/// ## Constraints (asserted by host before launch)
/// `n_dim == 128`, `block_len == 64`, `p_dim == 64`,
/// `t_len % 64 == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ssd_scan_chunked_p1(
    x_ptr: *const f32,
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *const f32,
    y_ptr: *mut f32,
    _batch: u32,
    t_len: u32,
    n_heads: u32,
    _p_dim: u32,
    _n_dim: u32,
    _block_len: u32,
) {
    // P2 compile-time constants (production shape).
    const N: usize = 128; // d_state — compile-time const for LLVM to unroll
    const Q: usize = 64; // chunk length
    const P: usize = 64; // threads per block = p_dim

    // Bank-conflict padding: N + 1 = 129.
    // With P=64 threads and 32 shared-memory banks, stride N=128 (divisible
    // by 32) maps each thread to the SAME bank for the same n_idx → 64-way
    // bank conflict.  Padding by 1 gives stride 129 (= 4×32 + 1), so thread k
    // at n_idx=0 lands in bank (k*129)%32 = k%32 — all different for k<32.
    // Result: 2-way max conflict (64 threads / 32 banks) instead of 64-way.
    const N_PAD: usize = N + 1; // = 129

    // Each thread loads (Q*N) / P = 8192 / 64 = 128 floats of s_b / s_c.
    const LOAD_PER_THREAD: usize = (Q * N) / P; // = 128

    // Shared memory boundary after s_hstate.
    // s_hstate starts at Q + 2*Q*N and has P*N_PAD floats.
    const HSTATE_END: usize = Q + 2 * Q * N + P * N_PAD; // 64+16384+8256 = 24704

    // Thread / block identifiers.
    // SAFETY: nvptx intrinsics are always valid inside a ptx-kernel.
    let tid = unsafe { nvptx::_thread_idx_x() } as usize; // 0..P
    let bid = unsafe { nvptx::_block_idx_x() } as usize; // 0..B*H

    let n_heads_us = n_heads as usize;
    let t_len_us = t_len as usize;
    let h = bid % n_heads_us;
    let b = bid / n_heads_us;

    // ---- Shared memory layout ----
    //
    // SAFETY: `shared_mem_base()` returns a generic-address pointer to the
    // dynamic shared-memory region (allocated by the host via LaunchConfig
    // and cuFuncSetAttribute, size P1_SMEM_BYTES).  All named sub-slices are
    // non-overlapping and stay within the total allocation.
    //
    // Float counts:
    //   s_a_cum     : Q         = 64
    //   s_b         : Q*N       = 8192
    //   s_c         : Q*N       = 8192
    //   s_hstate    : P*N_PAD   = 64*129 = 8256
    //   s_decay_end : Q         = 64
    //   s_bc_a/b    : 2*Q       = 128
    //   s_decay_a/b : 2*Q       = 128
    //   s_decay_start: Q        = 64
    //   Total       : HSTATE_END + 6*Q = 24704 + 384 = 25088 floats = 100 352 B
    let sbase: *mut f32 = unsafe { shared_mem_base() };

    // s_a_cum[Q] at offset 0: thread 0 writes, all read after syncthreads.
    let s_a_cum: *mut f32 = sbase;

    // s_b[Q*N] at offset Q: cooperatively loaded; read-only during steps 3/4.
    // SAFETY: non-overlapping: Q < Q+Q*N < Q+2*Q*N < HSTATE_END.
    let s_b: *mut f32 = unsafe { sbase.add(Q) };

    // s_c[Q*N] at offset Q+Q*N.
    let s_c: *mut f32 = unsafe { sbase.add(Q + Q * N) };

    // s_hstate[P*N_PAD] at offset Q+2*Q*N.
    // Thread `tid` owns slice [tid*N_PAD .. tid*N_PAD + N].
    // SAFETY: tid < P, N_PAD = N+1 = 129 → tid*N_PAD+N < P*N_PAD; within SHARED_BUF.
    let s_hstate: *mut f32 = unsafe { sbase.add(Q + 2 * Q * N) };
    let my_hstate: *mut f32 = unsafe { s_hstate.add(tid * N_PAD) };

    // Scratch arrays starting at HSTATE_END (offset 24704 floats = 98 816 B).
    // s_decay_end[Q]: expf(a_cum_end - a_cum[qprime]) per qprime (step 4).
    // SAFETY: HSTATE_END < HSTATE_END+Q ≤ 25088 within total 25088.
    let s_decay_end: *mut f32 = unsafe { sbase.add(HSTATE_END) };

    // Double-buffered bc and intra-chunk decay rows (4 arrays of Q each).
    let s_bc_a: *mut f32 = unsafe { sbase.add(HSTATE_END + Q) };
    let s_bc_b: *mut f32 = unsafe { sbase.add(HSTATE_END + 2 * Q) };
    let s_decay_a: *mut f32 = unsafe { sbase.add(HSTATE_END + 3 * Q) };
    let s_decay_b: *mut f32 = unsafe { sbase.add(HSTATE_END + 4 * Q) };

    // Inter-chunk decay cache: expf(a_cum[q_idx]) precomputed at step 2b.
    // Thread `tid` writes slot `tid`; all threads read in step 3.
    // SAFETY: HSTATE_END+5*Q = 24704+320 = 25024 < 25088.
    let s_decay_start: *mut f32 = unsafe { sbase.add(HSTATE_END + 5 * Q) };

    // Zero-initialise this thread's hstate slice.
    // SAFETY: tid*N_PAD+n_idx < P*N_PAD (within s_hstate).
    for n_idx in 0..N {
        unsafe { *my_hstate.add(n_idx) = 0.0_f32 };
    }
    // SAFETY: syncthreads — all threads initialise hstate before the chunk loop.
    unsafe { syncthreads() };

    let n_chunks = t_len_us / Q;

    for c_idx in 0..n_chunks {
        // ---- Step 1: thread 0 builds a_cum for this chunk ----
        if tid == 0 {
            let mut acc = 0_f32;
            for q_idx in 0..Q {
                let global_t = c_idx * Q + q_idx;
                // SAFETY: layout (B, T, H); all indices within declared shape.
                let a_val = unsafe { *a_ptr.add((b * t_len_us + global_t) * n_heads_us + h) };
                acc += a_val;
                // SAFETY: q_idx < Q, within s_a_cum.
                unsafe { *s_a_cum.add(q_idx) = acc };
            }
        }
        // SAFETY: syncthreads — thread 0 has finished writing s_a_cum.
        unsafe { syncthreads() };

        // ---- Step 2: cooperatively load s_b and s_c (round-robin coalesced) ----
        //
        // M-1 review fix (commit 4842bca review): the previous linear partition
        //   i = tid * LOAD_PER_THREAD + k
        // mapped consecutive threads to indices 128 apart → 1536-element global
        // stride between adjacent threads (= n_heads * N) → 32 separate L2
        // transactions per warp load.
        //
        // Round-robin: thread `tid` loads indices
        //   i_k = k * P + tid     for k in 0..LOAD_PER_THREAD
        // → within a warp, consecutive threads access consecutive n_col values in
        // the same q_row (since stride P=64 stays within a single (q_row, n) row
        // until k crosses a row boundary). 128-byte coalesced loads per warp.
        for k in 0..LOAD_PER_THREAD {
            let i = k * P + tid;
            let q_row = i / N;
            let n_col = i % N;
            let global_t = c_idx * Q + q_row;
            let g_off = ((b * t_len_us + global_t) * n_heads_us + h) * N + n_col;
            // SAFETY: g_off within (B,T,H,N) bounds; i < Q*N within s_b/s_c.
            unsafe {
                *s_b.add(i) = *b_ptr.add(g_off);
                *s_c.add(i) = *c_ptr.add(g_off);
            }
        }
        // SAFETY: syncthreads — all threads must finish s_b / s_c before step 2b.
        unsafe { syncthreads() };

        // SAFETY: Q-1 < Q, s_a_cum is filled.
        let a_cum_end = unsafe { *s_a_cum.add(Q - 1) };

        // ---- Step 2b: decay caches + seed q_idx=0 bc/decay ----
        //
        // Thread `tid` computes independently (no cross-thread reads):
        //   s_decay_end[tid]   = expf(a_cum_end − a_cum[tid])  → step 4
        //   s_decay_start[tid] = expf(a_cum[tid])              → step 3 inter
        //   s_bc_a[tid]        = bc[0, tid]                    → step 3 seed
        //   s_decay_a[tid]     = expf(a_cum[0] − a_cum[tid])   → step 3 seed
        //
        // SAFETY: tid < P = Q; all target arrays are size Q.
        let a_tid = unsafe { *s_a_cum.add(tid) };
        unsafe {
            *s_decay_end.add(tid) = expf(a_cum_end - a_tid);
            *s_decay_start.add(tid) = expf(a_tid);
        }

        // Seed bc/decay for q_idx=0 into buffer A.
        // bc[0, tid] = C[0,:]·B[tid,:] is generically non-zero for any tid;
        // we skip slots [tid > 0] because the consuming intra-chunk loop
        // `for qprime in 0..=q_idx` never reads them at q_idx=0.  This is
        // a dead-write elision, not a math property of the bc product.
        // SAFETY: s_c[0..N] and s_b[0..N] already loaded.
        let a_cum_t0 = unsafe { *s_a_cum.add(0) };
        let mut bc0 = 0_f32;
        if tid == 0 {
            for n_idx in 0..N {
                let c_val = unsafe { *s_c.add(n_idx) };
                let b_val = unsafe { *s_b.add(n_idx) };
                bc0 += c_val * b_val;
            }
        }
        // SAFETY: tid < Q; s_bc_a, s_decay_a within SHARED_BUF.
        unsafe {
            *s_bc_a.add(tid) = bc0;
            *s_decay_a.add(tid) = expf(a_cum_t0 - a_tid);
        }
        // SAFETY: syncthreads — all threads must see all 5 scratch arrays.
        unsafe { syncthreads() };

        // ---- Step 3: double-buffered bc/decay rows, direct x reads ----
        //
        // x values are read directly from global memory via coalesced accesses:
        // all 64 threads read consecutive tid values for the same (b,t,h) at
        // each qprime step → 256-byte coalesced transaction → L1/L2 resident.
        // The Q=64 unique timesteps per chunk fit in 16 KB of L1 cache, so
        // repeated reads of the same qprime across q_idx iterations are L1 hits.
        //
        // Precompute the base offset for this chunk's x block to reduce
        // per-iteration address arithmetic.
        let x_chunk_base = ((b * t_len_us + c_idx * Q) * n_heads_us + h) * P;

        for q_idx in 0..Q {
            let global_t = c_idx * Q + q_idx;
            let c_base = q_idx * N;

            // Double-buffer select: even q_idx reads A, writes B; odd reads B, writes A.
            let (s_bc_rd, s_decay_rd, s_bc_wr, s_decay_wr) = if q_idx % 2 == 0 {
                (s_bc_a, s_decay_a, s_bc_b, s_decay_b)
            } else {
                (s_bc_b, s_decay_b, s_bc_a, s_decay_a)
            };

            // Intra-chunk: Σ_{q'≤q_idx} decay_rd[q'] · bc_rd[q'] · x[q']
            // x read: all 64 threads access consecutive tid values (fully coalesced).
            // SAFETY: qprime ≤ q_idx < Q; within s_decay_rd, s_bc_rd;
            //         x_chunk_base + qprime*P*H + tid within x_ptr bounds.
            let mut y_acc = 0_f32;
            for qprime in 0..=q_idx {
                let decay = unsafe { *s_decay_rd.add(qprime) };
                let bc = unsafe { *s_bc_rd.add(qprime) };
                // SAFETY: layout (B, T, H, P); qprime < Q; tid < P.
                let x_val = unsafe { *x_ptr.add(x_chunk_base + qprime * n_heads_us * P + tid) };
                y_acc += decay * bc * x_val;
            }

            // Inter-chunk: s_decay_start[q_idx] · Σ_n C[q_idx,n] · hstate[n]
            // s_decay_start avoids a per-q_idx expf call (precomputed at step 2b).
            // SAFETY: q_idx < Q → within s_decay_start; c_base+N-1 < Q*N in s_c;
            //         n_idx < N → within my_hstate (owned by this thread).
            let decay_from_start = unsafe { *s_decay_start.add(q_idx) };
            for n_idx in 0..N {
                let c_val = unsafe { *s_c.add(c_base + n_idx) };
                let h_val = unsafe { *my_hstate.add(n_idx) };
                y_acc += decay_from_start * c_val * h_val;
            }

            // Write y[b, c*Q+q_idx, h, tid].
            // SAFETY: global_t < t_len; tid < P; within y_ptr shape.
            unsafe {
                *y_ptr.add((((b * t_len_us + global_t) * n_heads_us) + h) * P + tid) = y_acc;
            }

            // Prefetch bc/decay for q_idx+1 into the write buffer.
            if q_idx + 1 < Q {
                let next_q = q_idx + 1;
                let a_cum_next = unsafe { *s_a_cum.add(next_q) };
                let b_base = tid * N;
                let c_next_base = next_q * N;
                let mut bc_next = 0_f32;
                // Slots [tid > next_q] are never read by the q_idx+1 iteration's
                // `for qprime in 0..=q_idx` loop, so we elide the wasted dot
                // product.  This is dead-write elision, NOT a lower-triangle
                // property of the C·Bᵀ product itself.
                // SAFETY: c_next_base+N-1 < Q*N; b_base+N-1 < Q*N.
                if tid <= next_q {
                    for n_idx in 0..N {
                        let c_val = unsafe { *s_c.add(c_next_base + n_idx) };
                        let b_val = unsafe { *s_b.add(b_base + n_idx) };
                        bc_next += c_val * b_val;
                    }
                }
                // SAFETY: tid < Q; s_bc_wr, s_decay_wr within SHARED_BUF.
                unsafe {
                    *s_bc_wr.add(tid) = bc_next;
                    *s_decay_wr.add(tid) = expf(a_cum_next - a_tid);
                }
            }

            // SAFETY: syncthreads — all threads finish reading current buffer
            // AND writing next buffer before the next iteration advances.
            unsafe { syncthreads() };
        }

        // ---- Step 4: update hstate (tiled, coalesced x, register tiles) ----
        //
        // hstate[n] ← exp(A_end) · hstate[n]
        //           + Σ_{q'=0..Q} s_decay_end[q'] · s_b[q', n] · x[q', tid]
        //
        // Tiling (TILE=4): for each 4-float tile, the qprime loop is OUTER
        // so x_ptr reads are fully coalesced across all 64 threads.
        // s_decay_end eliminates Q*N=8192 expf calls per thread per chunk.
        const TILE: usize = 4;
        const N_TILES: usize = N / TILE; // = 32

        let decay_full = expf(a_cum_end);

        for tile in 0..N_TILES {
            let n0 = tile * TILE;
            // Load 4-element hstate tile from shared into registers and apply decay.
            // SAFETY: n0+3 < N; my_hstate[n0..n0+4] within P*N_PAD allocation.
            let mut h0 = unsafe { *my_hstate.add(n0) } * decay_full;
            let mut h1 = unsafe { *my_hstate.add(n0 + 1) } * decay_full;
            let mut h2 = unsafe { *my_hstate.add(n0 + 2) } * decay_full;
            let mut h3 = unsafe { *my_hstate.add(n0 + 3) } * decay_full;

            for qprime in 0..Q {
                // SAFETY: qprime < Q, within s_decay_end.
                let decay_to_end = unsafe { *s_decay_end.add(qprime) };
                // Coalesced: all 64 threads read consecutive p-values for same timestep.
                // SAFETY: layout (B, T, H, P); tid < P; c_idx*Q+qprime < t_len.
                let x_val = unsafe {
                    *x_ptr.add(
                        (((b * t_len_us + c_idx * Q + qprime) * n_heads_us) + h) * P + tid,
                    )
                };
                let scale = decay_to_end * x_val;
                // SAFETY: qprime*N+n0+3 < Q*N, within s_b.
                h0 += scale * unsafe { *s_b.add(qprime * N + n0) };
                h1 += scale * unsafe { *s_b.add(qprime * N + n0 + 1) };
                h2 += scale * unsafe { *s_b.add(qprime * N + n0 + 2) };
                h3 += scale * unsafe { *s_b.add(qprime * N + n0 + 3) };
            }

            // Write updated tile back to shared hstate.
            // SAFETY: n0+3 < N; my_hstate within s_hstate.
            unsafe {
                *my_hstate.add(n0) = h0;
                *my_hstate.add(n0 + 1) = h1;
                *my_hstate.add(n0 + 2) = h2;
                *my_hstate.add(n0 + 3) = h3;
            }
        }
        // SAFETY: syncthreads — all threads must finish reading s_b / s_decay_end
        // before the next chunk iteration overwrites s_a_cum, s_b, s_c, and scratch.
        unsafe { syncthreads() };
    }
}

// ---- B4.2c shape copy kernels ------------------------------------------

/// 2D transpose: `out[col * rows + row] = in[row * cols + col]`.
///
/// Kernel is launched with 1 thread per output element.
/// `n` = total element count = rows * cols.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn transpose_2d_f32(
    src: *const f32,
    dst: *mut f32,
    rows: u32,
    cols: u32,
) {
    let idx = elem_idx();
    let n = rows * cols;
    if idx < n {
        // dst[idx]: idx = out_col * rows + out_row
        let out_row = idx % rows;
        let out_col = idx / rows;
        // src[out_col * cols + out_row]  — wait, dst is transposed layout.
        // dst shape is (cols, rows); out_row indexes 0..rows, out_col 0..cols.
        // src is row-major (rows, cols): src[r * cols + c]
        // dst is row-major (cols, rows): dst[c * rows + r]
        // so dst[idx] corresponds to (out_col=c, out_row=r):
        //   src index = out_row * cols + out_col
        let src_idx = out_row * cols + out_col;
        unsafe { *dst.add(idx as usize) = *src.add(src_idx as usize) };
    }
}

/// ND transpose: permute axes.
///
/// `rank` ≤ 8. `in_strides` / `out_strides` are computed on the host
/// and passed in as flat arrays (length = rank). The output index `idx`
/// is mapped to a multi-index, then gathered using the **inverse**
/// permutation from `in_strides`.
///
/// Specifically: given out-element linear index `idx`, we decode the
/// multi-index using `out_strides`, then read from `src` using
/// `in_strides` with the permuted coordinate order.
///
/// Host pre-computes (for a permutation P):
///   out_strides  = strides of output tensor (transposed shape)
///   in_strides   = strides of input tensor in **output coordinate order**
///                  i.e. in_strides[i] = input stride along the axis that
///                  maps to output axis i.
///
/// This kernel does NOT use shared memory — it's a pure gather.
/// Launched with 1 thread per output element.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn transpose_nd_f32(
    src: *const f32,
    dst: *mut f32,
    in_strides_ptr: *const u32,  // length `rank`
    out_strides_ptr: *const u32, // length `rank`
    rank: u32,
    n: u32,
) {
    let idx = elem_idx();
    if idx >= n {
        return;
    }
    // Decode multi-index from output-flat index using out_strides.
    // Then compute source index using in_strides.
    let rank_us = rank as usize;
    let mut remaining = idx;
    let mut src_idx: u32 = 0;
    for d in 0..rank_us {
        let os = unsafe { *out_strides_ptr.add(d) };
        let coord = remaining / os;
        remaining %= os;
        let is_ = unsafe { *in_strides_ptr.add(d) };
        src_idx += coord * is_;
    }
    unsafe { *dst.add(idx as usize) = *src.add(src_idx as usize) };
}

/// Narrow copy: copy a contiguous slice along `axis`.
///
/// All dimensions are flattened into:
///   outer_count × axis_len_dst × inner_len
///
/// Each thread handles one output element.
/// `outer_count` = product of dims before axis
/// `axis_len_dst` = `len` (slice length)
/// `inner_len` = product of dims after axis
/// `axis_offset` = `start` in the source
/// `axis_len_src` = full size of axis in source
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn narrow_copy_f32(
    src: *const f32,
    dst: *mut f32,
    _outer_count: u32,
    axis_len_src: u32,
    axis_len_dst: u32,
    axis_offset: u32,
    inner_len: u32,
    n: u32,
) {
    let idx = elem_idx();
    if idx >= n {
        return;
    }
    // Decode flat output index.
    let inner = inner_len;
    let axis_dst = axis_len_dst;
    let i_inner = idx % inner;
    let tmp = idx / inner;
    let i_axis_dst = tmp % axis_dst;
    let i_outer = tmp / axis_dst;

    let i_axis_src = i_axis_dst + axis_offset;
    let src_idx = i_outer * (axis_len_src * inner) + i_axis_src * inner + i_inner;
    unsafe { *dst.add(idx as usize) = *src.add(src_idx as usize) };
}

/// `Ops::narrow`'s backward (VJP): scatter `grad_out` (shape = `orig_shape`
/// with axis `dim` replaced by `axis_len_grad`) into a fresh `orig_shape`
/// tensor, zero outside the `[axis_start, axis_start+axis_len_grad)` window
/// along `dim` — the exact inverse mapping of `narrow_copy_f32` above, but
/// launched one thread per **output** (`orig_shape`-sized) element so the
/// zero-fill and the gather happen in the same pass (no separate memset,
/// no atomics: every output element is written exactly once).
///
/// Replaces `CudaBackend::scatter_to_narrow`'s host round trip
/// (`docs/perf-log.md` B'.3): that helper's `dim == rank - 1` case — the
/// *only* case `Mamba2Block::forward` ever hits, since every `narrow` there
/// slices the trailing feature dim — degenerated to `outer * axis_len_grad`
/// individual 4-byte `copy_from_slice` calls because `inner_len == 1` in
/// that geometry (measured ~1.3 ms/call, dominated by that host loop).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn narrow_backward_f32(
    grad_out: *const f32,
    dst: *mut f32,
    axis_len_orig: u32,
    axis_len_grad: u32,
    axis_start: u32,
    inner_len: u32,
    n: u32,
) {
    let idx = elem_idx();
    if idx >= n {
        return;
    }
    let inner = inner_len;
    let axis_o = axis_len_orig;
    let i_inner = idx % inner;
    let tmp = idx / inner;
    let i_axis_o = tmp % axis_o;
    let i_outer = tmp / axis_o;

    if i_axis_o >= axis_start && i_axis_o < axis_start + axis_len_grad {
        let i_axis_g = i_axis_o - axis_start;
        let src_idx = i_outer * (axis_len_grad * inner) + i_axis_g * inner + i_inner;
        unsafe { *dst.add(idx as usize) = *grad_out.add(src_idx as usize) };
    } else {
        unsafe { *dst.add(idx as usize) = 0.0_f32 };
    }
}

/// Broadcast copy: expand size-1 dimensions.
///
/// `src_strides` encodes per-dimension strides of the source (using 0
/// for broadcast dims where source size == 1). Length = `rank`.
/// `out_strides` encodes strides for the output layout.
/// Launched with 1 thread per output element.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn broadcast_copy_f32(
    src: *const f32,
    dst: *mut f32,
    src_strides_ptr: *const u32, // length rank; 0 for broadcast dims
    out_strides_ptr: *const u32, // length rank
    rank: u32,
    n: u32,
) {
    let idx = elem_idx();
    if idx >= n {
        return;
    }
    let rank_us = rank as usize;
    let mut remaining = idx;
    let mut src_idx: u32 = 0;
    for d in 0..rank_us {
        let os = unsafe { *out_strides_ptr.add(d) };
        let coord = remaining / os;
        remaining %= os;
        let ss = unsafe { *src_strides_ptr.add(d) };
        src_idx += coord * ss;
    }
    unsafe { *dst.add(idx as usize) = *src.add(src_idx as usize) };
}

/// Broadcast-aware elementwise multiply: `dst[idx] = a[a_idx] * b[b_idx]`,
/// where `a_idx`/`b_idx` are each computed from `idx`'s multi-index via
/// their own per-operand strides (0 along that operand's broadcast dims) —
/// same convention as `broadcast_copy_f32`, generalised to two operands.
///
/// Replaces the host round-trip `CudaBackend::broadcast_binary_op` fallback
/// on `mul`'s broadcast path — both the forward `Ops::mul` broadcast branch
/// and the two broadcast arms of `vjp:Mul` (`docs/perf-log.md` B'.3):
/// that helper's `clone_dtoh` → host divmod loop → `clone_htod` measured
/// ~1.3 ms/call, with the host-side scalar divmod loop (not the PCIe
/// copies) as the dominant cost (measured ~350 ms compute vs. ~24 ms dtoh
/// / ~13 ms htod over one training step's worth of calls).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn broadcast_mul_f32(
    a: *const f32,
    b: *const f32,
    dst: *mut f32,
    a_strides_ptr: *const u32,   // length rank; 0 for a's broadcast dims
    b_strides_ptr: *const u32,   // length rank; 0 for b's broadcast dims
    out_strides_ptr: *const u32, // length rank
    rank: u32,
    n: u32,
) {
    let idx = elem_idx();
    if idx >= n {
        return;
    }
    let rank_us = rank as usize;
    let mut remaining = idx;
    let mut a_idx: u32 = 0;
    let mut b_idx: u32 = 0;
    for d in 0..rank_us {
        let os = unsafe { *out_strides_ptr.add(d) };
        let coord = remaining / os;
        remaining %= os;
        let a_s = unsafe { *a_strides_ptr.add(d) };
        let b_s = unsafe { *b_strides_ptr.add(d) };
        a_idx += coord * a_s;
        b_idx += coord * b_s;
    }
    let av = unsafe { *a.add(a_idx as usize) };
    let bv = unsafe { *b.add(b_idx as usize) };
    unsafe { *dst.add(idx as usize) = av * bv };
}

// ---- B4.2c reduction kernels -------------------------------------------

/// Inclusive cumulative sum along the **last** dimension.
///
/// One thread per "row" (all dimensions except the last).
/// `last_dim` = length of the last dimension.
/// `n_rows` = total elements / last_dim.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn cumsum_lastdim_f32(
    src: *const f32,
    dst: *mut f32,
    n_rows: u32,
    last_dim: u32,
) {
    let row = elem_idx();
    if row >= n_rows {
        return;
    }
    let row_us = row as usize;
    let ld = last_dim as usize;
    let base = row_us * ld;
    let mut acc = 0f32;
    for i in 0..ld {
        acc += unsafe { *src.add(base + i) };
        unsafe { *dst.add(base + i) = acc };
    }
}

/// RMSNorm: `out[row, d] = x[row, d] * weight[d] * rsqrt(mean(x[row]^2) + eps)`.
///
/// One thread per row. `d_model` = last dimension.
/// `weight` is shape `(d_model,)`.
/// `n_rows` = numel / d_model.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn rmsnorm_f32(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n_rows: u32,
    d_model: u32,
    eps: f32,
) {
    let row = elem_idx();
    if row >= n_rows {
        return;
    }
    let row_us = row as usize;
    let d = d_model as usize;
    let base = row_us * d;

    // Compute mean(x^2) for this row.
    let mut sum_sq = 0f32;
    for i in 0..d {
        let xi = unsafe { *x.add(base + i) };
        sum_sq += xi * xi;
    }
    let mean_sq = sum_sq / (d as f32);

    // rsqrt(mean_sq + eps) via PTX intrinsic.
    let scale: f32;
    unsafe {
        asm!(
            "rsqrt.approx.f32 {r}, {v};",
            r = out(reg32) scale,
            v = in(reg32) mean_sq + eps,
            options(pure, nomem, nostack),
        );
    }

    for i in 0..d {
        let xi = unsafe { *x.add(base + i) };
        let wi = unsafe { *weight.add(i) };
        unsafe { *out.add(base + i) = xi * wi * scale };
    }
}

/// log_softmax along the **last** dimension.
///
/// One thread per row. Uses `ex2.approx` and `lg2.approx` for fast
/// exp/log, combined with a numerically stable max-subtraction.
/// `n_rows` = numel / last_dim.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn log_softmax_lastdim_f32(
    src: *const f32,
    dst: *mut f32,
    n_rows: u32,
    last_dim: u32,
) {
    let row = elem_idx();
    if row >= n_rows {
        return;
    }
    let row_us = row as usize;
    let ld = last_dim as usize;
    let base = row_us * ld;

    // Step 1: max for numerical stability.
    let mut max_val = unsafe { *src.add(base) };
    for i in 1..ld {
        let v = unsafe { *src.add(base + i) };
        if v > max_val {
            max_val = v;
        }
    }

    // Step 2: sum of exp(x - max) using ex2.approx.
    const LOG2_E: f32 = 1.442_695_040_888_963_4_f32;
    let mut sum_exp = 0f32;
    for i in 0..ld {
        let v = unsafe { *src.add(base + i) };
        let y = (v - max_val) * LOG2_E;
        let e: f32;
        unsafe {
            asm!(
                "ex2.approx.f32 {r}, {y};",
                r = out(reg32) e,
                y = in(reg32) y,
                options(pure, nomem, nostack),
            );
        }
        sum_exp += e;
    }

    // Step 3: log(sum_exp) via lg2.approx * ln(2).
    const LN2: f32 = 0.693_147_180_559_945_f32;
    let log_sum_exp: f32;
    unsafe {
        asm!(
            "lg2.approx.f32 {r}, {v};",
            r = out(reg32) log_sum_exp,
            v = in(reg32) sum_exp,
            options(pure, nomem, nostack),
        );
    }
    let log_sum_exp = log_sum_exp * LN2;

    // Step 4: write log_softmax = (x - max) - log_sum_exp.
    for i in 0..ld {
        let v = unsafe { *src.add(base + i) };
        unsafe { *dst.add(base + i) = (v - max_val) - log_sum_exp };
    }
}

// ---- B4.2d indexing kernels -------------------------------------------

/// Embedding lookup: `out[i * D + d] = table[indices[i] * D + d]`.
///
/// `indices_len` = total number of indices (flat).
/// `embed_dim`   = D (embedding dimension, last dim of table).
/// 1 thread per (flat_index, dim) pair — grid is 1-D over `indices_len * embed_dim`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn embedding_f32(
    table: *const f32,   // (V, D)
    indices: *const i64, // flat indices, length = indices_len
    out: *mut f32,       // (indices_len, D)
    indices_len: u32,
    embed_dim: u32,
    vocab_size: u32,
) {
    let tid = elem_idx();
    let n = indices_len * embed_dim;
    if tid >= n {
        return;
    }
    let d = tid % embed_dim;
    let i = tid / embed_dim;
    let raw = unsafe { *indices.add(i as usize) };
    // Bounds check guards against (a) negative i64 (raw < 0) and (b)
    // out-of-vocab row indices. Without this, the cast `raw as usize`
    // for a negative value yields a huge usize and we'd read arbitrary
    // GPU memory. We write 0 on OOB so the caller sees a deterministic
    // value rather than a silent device memory leak.
    if raw < 0 || (raw as u64) >= vocab_size as u64 {
        unsafe { *out.add(tid as usize) = 0.0_f32 };
        return;
    }
    let row = raw as usize;
    unsafe { *out.add(tid as usize) = *table.add(row * embed_dim as usize + d as usize) };
}

/// Gather along the last dimension: `out[..., j] = src[..., indices[..., j]]`.
///
/// Inputs are contiguous row-major. One thread per output element.
/// `n`          = total output elements.
/// `last_dim_src` = last dim of `src`.
/// `last_dim_idx` = last dim of `indices` (= last dim of output).
/// The "outer" count = n / last_dim_idx.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gather_lastdim_f32(
    src: *const f32,     // (..., last_dim_src)
    indices: *const i64, // (..., last_dim_idx)
    out: *mut f32,       // (..., last_dim_idx)
    n: u32,              // total output elements
    last_dim_src: u32,
    last_dim_idx: u32,
) {
    let tid = elem_idx();
    if tid >= n {
        return;
    }
    // tid = outer_idx * last_dim_idx + j
    let _j = tid % last_dim_idx;
    let outer_idx = tid / last_dim_idx;
    let raw = unsafe { *indices.add(tid as usize) };
    // Bounds check — negative or out-of-range indices write 0 rather
    // than read arbitrary GPU memory (mirrors embedding_f32).
    if raw < 0 || (raw as u64) >= last_dim_src as u64 {
        unsafe { *out.add(tid as usize) = 0.0_f32 };
        return;
    }
    let src_col = raw as usize;
    let src_idx = outer_idx as usize * last_dim_src as usize + src_col;
    unsafe { *out.add(tid as usize) = *src.add(src_idx) };
}

/// im2col for 1D convolution: unroll `(B, C_in, T_in)` into
/// `(B * T_out, C_in * K)` for subsequent GEMM.
///
/// 1 thread per output element.
/// `n` = total elements = `B * T_out * C_in * K`.
/// Layout of output: row = `b * T_out + t_out`, col = `c_in * K + k`.
/// Padding fills with 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn im2col_f32(
    src: *const f32, // (B, C_in, T_in)
    dst: *mut f32,   // (B * T_out, C_in * K)
    batch: u32,
    c_in: u32,
    t_in: u32,
    t_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
    n: u32,
) {
    let tid = elem_idx();
    if tid >= n {
        return;
    }
    // Decode: tid indexes (B * T_out, C_in * K) in row-major.
    let ck = c_in * k_size;
    let bt_out = batch * t_out;
    // row: which (b, t_out) pair
    let row = tid / ck;
    // col: which (c, k) pair
    let col = tid % ck;

    let b = row / t_out;
    let t_o = row % t_out;
    let c = col / k_size;
    let k = col % k_size;

    // Corresponding input position.
    // t_in_pos = t_o * stride - padding + k
    let t_in_signed = (t_o * stride) as i32 + k as i32 - padding as i32;

    let val = if t_in_signed >= 0 && (t_in_signed as u32) < t_in && b < batch && c < c_in {
        // src layout: (B, C_in, T_in) → index = b * C_in * T_in + c * T_in + t
        let src_idx = b as usize * (c_in as usize * t_in as usize)
            + c as usize * t_in as usize
            + t_in_signed as usize;
        unsafe { *src.add(src_idx) }
    } else {
        0.0_f32
    };

    // dst layout: (B * T_out, C_in * K) row-major
    // but we want (B, T_out, C_in * K) then reshape to (B * T_out, C_in * K)
    // which is the same flat order: dst[row * ck + col]
    let _ = bt_out; // suppress unused warning
    unsafe { *dst.add(tid as usize) = val };
}

// ---- B4.3b backward kernels -----------------------------------------------

/// Scatter-add for embedding backward.
///
/// Atomically adds `grad_out[i * D + d]` into `table_grad[indices[i] * D + d]`.
///
/// `n` = `indices_len * embed_dim`.
/// Out-of-range or negative indices are silently skipped (mirrors embedding_f32).
///
/// # Safety
/// Uses `atomicAdd` via PTX `atom.global.add.f32` so concurrent threads
/// accumulating into the same row are handled correctly.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scatter_add_embedding_f32(
    table_grad: *mut f32, // (V, D) — accumulates gradients
    indices: *const i64,  // flat, length = indices_len
    grad_out: *const f32, // (indices_len, D)
    indices_len: u32,
    embed_dim: u32,
    vocab_size: u32,
) {
    let tid = elem_idx();
    let n = indices_len * embed_dim;
    if tid >= n {
        return;
    }
    let d = tid % embed_dim;
    let i = tid / embed_dim;
    let raw = unsafe { *indices.add(i as usize) };
    if raw < 0 || (raw as u64) >= vocab_size as u64 {
        return;
    }
    let row = raw as usize;
    let dst_ptr = unsafe { table_grad.add(row * embed_dim as usize + d as usize) };
    let val = unsafe { *grad_out.add(tid as usize) };
    // PTX atomic add to accumulate gradients from multiple indices pointing
    // at the same vocabulary row.
    // SAFETY: dst_ptr is within the allocated table_grad buffer (validated above).
    unsafe {
        asm!(
            "atom.global.add.f32 _, [{ptr}], {val};",
            ptr = in(reg64) dst_ptr,
            val = in(reg32) val,
            options(nostack),
        );
    }
}

/// Scatter-add for gather backward (last-dim only).
///
/// For each output element `tid` of gather, adds `grad_out[tid]` back into
/// `x_grad[outer_idx * last_dim_src + indices[tid]]`.
///
/// `n`            = total output elements of the forward gather.
/// `last_dim_src` = last dim of the source (x) tensor.
/// `last_dim_idx` = last dim of the indices (= last dim of gather output).
///
/// # Safety
/// Uses atomic add because multiple gather positions can map to the
/// same source element.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scatter_add_gather_lastdim_f32(
    x_grad: *mut f32,     // (..., last_dim_src) — accumulates gradients
    indices: *const i64,  // (..., last_dim_idx)
    grad_out: *const f32, // (..., last_dim_idx)
    n: u32,
    last_dim_src: u32,
    last_dim_idx: u32,
) {
    let tid = elem_idx();
    if tid >= n {
        return;
    }
    let outer_idx = tid / last_dim_idx;
    let raw = unsafe { *indices.add(tid as usize) };
    if raw < 0 || (raw as u64) >= last_dim_src as u64 {
        return;
    }
    let src_col = raw as usize;
    let dst_idx = outer_idx as usize * last_dim_src as usize + src_col;
    let dst_ptr = unsafe { x_grad.add(dst_idx) };
    let val = unsafe { *grad_out.add(tid as usize) };
    // SAFETY: dst_ptr is within the allocated x_grad buffer (validated above).
    unsafe {
        asm!(
            "atom.global.add.f32 _, [{ptr}], {val};",
            ptr = in(reg64) dst_ptr,
            val = in(reg32) val,
            options(nostack),
        );
    }
}

/// col2im for 1D convolution backward.
///
/// Reverses im2col: scatters the gradient from `(B * T_out, C_in * K)`
/// column buffer back into `(B, C_in, T_in)` input gradient using atomic
/// add (since multiple im2col entries can map to the same input position).
///
/// `n` = total elements in the column buffer = `B * T_out * C_in * K`.
///
/// # Safety
/// Uses atomic add to accumulate overlapping kernel-patch contributions.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn col2im_f32(
    x_grad: *mut f32,     // (B, C_in, T_in) — accumulates
    col_grad: *const f32, // (B * T_out, C_in * K)
    batch: u32,
    c_in: u32,
    t_in: u32,
    t_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
    n: u32,
) {
    let tid = elem_idx();
    if tid >= n {
        return;
    }
    let ck = c_in * k_size;
    let row = tid / ck;
    let col = tid % ck;

    let b = row / t_out;
    let t_o = row % t_out;
    let c = col / k_size;
    let k = col % k_size;

    let t_in_signed = (t_o * stride) as i32 + k as i32 - padding as i32;
    if t_in_signed < 0 || (t_in_signed as u32) >= t_in || b >= batch || c >= c_in {
        return;
    }

    let dst_idx = b as usize * (c_in as usize * t_in as usize)
        + c as usize * t_in as usize
        + t_in_signed as usize;
    let dst_ptr = unsafe { x_grad.add(dst_idx) };
    let val = unsafe { *col_grad.add(tid as usize) };
    // SAFETY: dst_ptr is within the allocated x_grad buffer (validated above).
    unsafe {
        asm!(
            "atom.global.add.f32 _, [{ptr}], {val};",
            ptr = in(reg64) dst_ptr,
            val = in(reg32) val,
            options(nostack),
        );
    }
}

/// RMSNorm backward — compute `grad_x` contribution.
///
/// **Geometry**: one thread **block** per row (`grid = (n_rows, 1, 1)`,
/// `block = (BLOCK, 1, 1)` with `BLOCK` a power of two chosen by the host
/// launcher) — replaces an earlier one-thread-per-row scheme that only
/// used `n_rows` scalar threads total and read `x`/`grad_out` with a
/// per-thread stride of `d_model` (non-coalesced: adjacent threads =
/// adjacent *rows*, not adjacent addresses). Here `BLOCK` threads stride
/// across `d_model` together, so adjacent threads touch adjacent
/// addresses (coalesced) and every SM gets work instead of only two.
///
/// Per row:
///   `rms      = sqrt(mean(x^2) + eps)`
///   `inv_rms  = 1 / rms`
///   `dy_dot   = sum_i(grad_out[row, i] * x[row, i] * weight[i])`
///   `dy_scale = dy_dot * inv_rms^3 / d_model`
///   `gx[row, i] = weight[i] * inv_rms * grad_out[row, i]`
///              `- x[row, i] * dy_scale`
/// where `dy_scale = sum(grad_out * x * weight) * inv_rms^3 / d_model`.
///
/// `sum_sq` and `dy_dot` are each accumulated as one partial per thread,
/// then combined with a shared-memory tree reduction (fixed pairing
/// order, no atomics — bit-reproducible across runs).
///
/// # Safety
/// Host must launch with `block_dim.x` a power of two and
/// `shared_mem_bytes >= 2 * block_dim.x * 4` (two f32 partial-sum arrays).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn rmsnorm_backward_x_f32(
    grad_x: *mut f32,     // (n_rows, d_model) — output
    grad_out: *const f32, // (n_rows, d_model)
    x: *const f32,        // (n_rows, d_model)
    weight: *const f32,   // (d_model,)
    n_rows: u32,
    d_model: u32,
    eps: f32,
) {
    // SAFETY: nvptx intrinsics are always valid inside a ptx-kernel.
    let row = unsafe { nvptx::_block_idx_x() };
    // Uniform per-block (blockIdx.x is identical for every thread in the
    // block), so this early return cannot cause intra-block divergence
    // ahead of the syncthreads() calls below.
    if row >= n_rows {
        return;
    }
    let tid = unsafe { nvptx::_thread_idx_x() };
    let n_threads = unsafe { nvptx::_block_dim_x() } as usize;
    let tid_us = tid as usize;
    let d = d_model as usize;
    let base = row as usize * d;

    // ---- Pass 1: strided partial sums (coalesced across the block) ----
    let mut partial_sum_sq = 0f32;
    let mut partial_dy_dot = 0f32;
    let mut i = tid_us;
    while i < d {
        let xi = unsafe { *x.add(base + i) };
        let go_i = unsafe { *grad_out.add(base + i) };
        let w_i = unsafe { *weight.add(i) };
        partial_sum_sq += xi * xi;
        partial_dy_dot += go_i * xi * w_i;
        i += n_threads;
    }

    // SAFETY: `shared_mem_base()` returns this launch's dynamic shared
    // region; the host allocates `2 * n_threads * 4` bytes, so indices
    // `[0, n_threads)` (sum_sq partials) and `[n_threads, 2*n_threads)`
    // (dy_dot partials) are both in bounds.
    let s: *mut f32 = unsafe { shared_mem_base() };
    unsafe { *s.add(tid_us) = partial_sum_sq };
    unsafe { *s.add(n_threads + tid_us) = partial_dy_dot };
    // SAFETY: syncthreads — every partial must be written before any
    // thread starts reading a neighbor's partial in the reduction below.
    unsafe { syncthreads() };

    // Standard log2 tree reduction over both arrays at once. `n_threads`
    // must be a power of two (guaranteed by the host launch config).
    let mut stride = n_threads / 2;
    while stride > 0 {
        if tid_us < stride {
            let a = unsafe { *s.add(tid_us) };
            let b = unsafe { *s.add(tid_us + stride) };
            unsafe { *s.add(tid_us) = a + b };
            let a2 = unsafe { *s.add(n_threads + tid_us) };
            let b2 = unsafe { *s.add(n_threads + tid_us + stride) };
            unsafe { *s.add(n_threads + tid_us) = a2 + b2 };
        }
        // SAFETY: syncthreads — all threads finish this stride's
        // read/write before the next iteration halves the active set.
        unsafe { syncthreads() };
        stride /= 2;
    }

    // s[0] / s[n_threads] now hold the row's full sum_sq / dy_dot; every
    // thread reads the same finished values (nothing writes to `s` again).
    let sum_sq = unsafe { *s };
    let dy_dot = unsafe { *s.add(n_threads) };
    let rms_sq = sum_sq / (d as f32) + eps;
    // inv_rms = 1 / sqrt(rms_sq)
    let inv_rms: f32;
    unsafe {
        asm!(
            "rsqrt.approx.f32 {r}, {v};",
            r = out(reg32) inv_rms,
            v = in(reg32) rms_sq,
            options(pure, nomem, nostack),
        );
    }
    // dy_scale = dy_dot * inv_rms^3 / d
    let dy_scale = dy_dot * inv_rms * inv_rms * inv_rms / (d as f32);

    // ---- Pass 2: strided elementwise write ----
    let mut i = tid_us;
    while i < d {
        let go_i = unsafe { *grad_out.add(base + i) };
        let x_i = unsafe { *x.add(base + i) };
        let w_i = unsafe { *weight.add(i) };
        let gx_i = w_i * inv_rms * go_i - x_i * dy_scale;
        unsafe { *grad_x.add(base + i) = gx_i };
        i += n_threads;
    }
}

/// RMSNorm backward — `grad_weight` **partial sums**, chunked over rows.
///
/// **Geometry**: `grid = (ceil(d_model / block.x), n_chunks, 1)`, one
/// thread per `(column, chunk)` pair — `col = blockIdx.x*blockDim.x +
/// threadIdx.x`, `chunk = blockIdx.y`. Replaces an earlier
/// one-thread-per-column kernel that used only `d_model` threads total
/// (768 at production scale) regardless of `n_rows`; chunking the row
/// axis across `blockIdx.y` multiplies the available parallelism by
/// `n_chunks` without changing the per-row arithmetic.
///
/// Each thread sums `grad_out[row,col] * x[row,col] * inv_rms[row]` over
/// its row range `[chunk*rows_per_chunk, min((chunk+1)*rows_per_chunk,
/// n_rows))`, recomputing `inv_rms[row] = rsqrt(mean(x[row]^2) + eps)`
/// per row — exactly as the prior single-kernel version did (chunking
/// only changes how many rows *one thread* visits, not the formula, so
/// the numerical result is identical).
///
/// Writes into `partial`, shape `(n_chunks, d_model)` row-major. The
/// caller reduces the chunk axis with a **separate, deterministic** pass
/// (`reduce_sum_dim_keepdim_f32`): fixed left-to-right summation, no
/// atomics, so the result reproduces bit-for-bit across runs.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn rmsnorm_backward_w_partial_f32(
    partial: *mut f32,    // (n_chunks, d_model) — output
    grad_out: *const f32, // (n_rows, d_model)
    x: *const f32,        // (n_rows, d_model)
    n_rows: u32,
    d_model: u32,
    rows_per_chunk: u32,
    n_chunks: u32,
    eps: f32,
) {
    // SAFETY: nvptx intrinsics are always valid inside a ptx-kernel.
    let col = unsafe { nvptx::_block_idx_x() * nvptx::_block_dim_x() + nvptx::_thread_idx_x() };
    let chunk = unsafe { nvptx::_block_idx_y() };
    // No shared memory / syncthreads in this kernel, so a per-thread
    // divergent early return here (e.g. the ragged last column block)
    // cannot deadlock anything.
    if col >= d_model || chunk >= n_chunks {
        return;
    }
    let col_us = col as usize;
    let d = d_model as usize;
    let row_start = chunk * rows_per_chunk;
    let row_end = if row_start + rows_per_chunk < n_rows {
        row_start + rows_per_chunk
    } else {
        n_rows
    };

    let mut acc = 0f32;
    let mut row = row_start;
    while row < row_end {
        let base = row as usize * d;
        // Compute inv_rms for this row (same per-row formula as before).
        let mut sum_sq = 0f32;
        for i in 0..d {
            let xi = unsafe { *x.add(base + i) };
            sum_sq += xi * xi;
        }
        let rms_sq = sum_sq / (d as f32) + eps;
        let inv_rms: f32;
        unsafe {
            asm!(
                "rsqrt.approx.f32 {r}, {v};",
                r = out(reg32) inv_rms,
                v = in(reg32) rms_sq,
                options(pure, nomem, nostack),
            );
        }
        let go = unsafe { *grad_out.add(base + col_us) };
        let xi = unsafe { *x.add(base + col_us) };
        acc += go * xi * inv_rms;
        row += 1;
    }
    let out_idx = chunk as usize * d + col_us;
    unsafe { *partial.add(out_idx) = acc };
}

/// Reverse inclusive cumsum along the last dimension.
///
/// `out[row, i] = sum_{j=i}^{last_dim-1} src[row, j]`
///
/// This is the VJP of forward cumsum: given `y = cumsum(x)` and `g = grad_out`,
/// `grad_x[i] = sum_{j >= i} g[j]` = reverse cumsum of `g`.
///
/// One thread per row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn reverse_cumsum_lastdim_f32(
    src: *const f32,
    dst: *mut f32,
    n_rows: u32,
    last_dim: u32,
) {
    let row = elem_idx();
    if row >= n_rows {
        return;
    }
    let row_us = row as usize;
    let ld = last_dim as usize;
    let base = row_us * ld;
    let mut acc = 0f32;
    // Walk from the end, accumulating suffix sum.
    let mut i = ld;
    while i > 0 {
        i -= 1;
        acc += unsafe { *src.add(base + i) };
        unsafe { *dst.add(base + i) = acc };
    }
}

/// Reduce sum along a single axis with keepdim = true.
///
/// Output shape is same as input but with axis dimension set to 1 (kept).
///
/// Parameters:
///   `outer` = product of dims before `axis`
///   `axis_len` = length of the reduction axis
///   `inner` = product of dims after `axis`
///
/// One thread per output element (`outer * 1 * inner` = `outer * inner`).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn reduce_sum_dim_keepdim_f32(
    src: *const f32,
    dst: *mut f32,
    outer: u32,
    axis_len: u32,
    inner: u32,
) {
    let tid = elem_idx();
    let n_out = outer * inner;
    if tid >= n_out {
        return;
    }
    // tid indexes the (outer, inner) output space.
    let i_inner = tid % inner;
    let i_outer = tid / inner;

    // Sum over axis.
    let src_base = i_outer as usize * axis_len as usize * inner as usize;
    let mut acc = 0f32;
    for a in 0..(axis_len as usize) {
        acc += unsafe { *src.add(src_base + a * inner as usize + i_inner as usize) };
    }
    // Output is in row-major with axis set to 1 (keepdim):
    // dst index = i_outer * 1 * inner + i_inner
    unsafe { *dst.add(tid as usize) = acc };
}

/// Pass 1 of the deterministic two-pass `sum_all`/`mean_all` reduce (B'.2f).
///
/// One thread per output chunk. Thread `c` sums `src[c], src[c + n_chunks],
/// src[c + 2*n_chunks], ...]` (grid-stride order) into `partial[c]`.
///
/// This grid-stride partitioning — rather than requiring `numel` to be an
/// exact multiple of `n_chunks` — means every element is visited exactly
/// once with no out-of-bounds reads regardless of the `numel`/`n_chunks`
/// ratio. For a **fixed** `n_chunks`, the partitioning and each thread's
/// left-to-right accumulation order are both fixed, so the result is
/// deterministic and reproducible run-to-run (no atomics anywhere in this
/// pass). Pass 2 (reducing `partial` to a single scalar) reuses the
/// existing `reduce_sum_dim_keepdim_f32` above, called with
/// `(outer=1, axis_len=n_chunks, inner=1)`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_reduce_partial_f32(
    src: *const f32,
    partial: *mut f32, // (n_chunks,) — output
    numel: u32,
    n_chunks: u32,
) {
    let c = elem_idx();
    if c >= n_chunks {
        return;
    }
    let mut acc = 0f32;
    let mut i = c;
    while i < numel {
        acc += unsafe { *src.add(i as usize) };
        i += n_chunks;
    }
    unsafe { *partial.add(c as usize) = acc };
}

// ---- B4.4e-conv1d  depthwise grouped-conv GPU kernels -----------------------
//
// Replaces the 1024-iteration host GEMM loop in `conv1d_f32` for the
// production depthwise shape [1,1024,512] w [1024,1,4] groups=1024.
// Each kernel dispatches one block per (batch, channel) pair.

/// Depthwise 1-D convolution forward (groups == channels, C_in/group == 1).
///
/// B4.4e-conv1d — eliminates the 26 ms host GEMM loop for production shape.
///
/// **Grid**: `(B * C, 1, 1)` — one block per `(batch, channel)` pair.
/// **Block**: `(128, 1, 1)`.
/// **Shared memory**: `K * 4` bytes — per-channel weight strip.
///
/// Each thread handles output positions `tid, tid+128, tid+256, …` for its
/// `(b, c)` block.  The K weight values for channel `c` are pre-loaded into
/// shared memory so that the inner K-loop reads from smem (broadcast) rather
/// than global memory.
///
/// # Layout (row-major, last axis fastest)
/// - `x_ptr`:    `(B, C, T_in)`
/// - `w_ptr`:    `(C, 1, K)` — `w[c, 0, k] = *(w_ptr + c*K + k)`
/// - `bias_ptr`: `(C,)`      — valid only when `has_bias != 0`
/// - `y_ptr`:    `(B, C, T_out)` — written, pre-allocated to zeros
///
/// # Constraints (validated by host before launch)
/// - `K ≤ 128` (threads 0..K-1 load the weight strip; larger K needs a
///   different cooperative load).
/// - `padding = t_in_pos convention: t_o * stride + k - padding`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn depthwise_conv1d_fwd_f32(
    x_ptr: *const f32,
    w_ptr: *const f32,
    bias_ptr: *const f32, // valid iff has_bias != 0; host provides dummy if None
    y_ptr: *mut f32,
    batch: u32,
    channels: u32,
    t_in: u32,
    t_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
) {
    const THREADS: usize = 128;

    // SAFETY: nvptx intrinsics are always valid inside a ptx-kernel.
    let bid = unsafe { nvptx::_block_idx_x() } as usize;
    let tid = unsafe { nvptx::_thread_idx_x() } as usize;

    let channels_us = channels as usize;
    let batch_us = batch as usize;

    // Map block index to (b, c).
    let c = bid % channels_us;
    let b = bid / channels_us;
    if b >= batch_us {
        return;
    }

    // ---- Step 1: load K weights into shared memory ----
    // Threads 0..K-1 each load one weight.  K ≤ THREADS=128 is the constraint.
    // SAFETY: SHARED_BUF has k_size*4 bytes (set by host in LaunchConfig).
    //         tid < k_size ≤ THREADS; w_ptr layout (C,1,K); c*K+tid in bounds.
    let k_us = k_size as usize;
    let s_w: *mut f32 = unsafe { shared_mem_base() };
    if tid < k_us {
        unsafe { *s_w.add(tid) = *w_ptr.add(c * k_us + tid) };
    }
    // SAFETY: all threads must see the loaded weight before the compute loop.
    unsafe { syncthreads() };

    // ---- Step 2: compute output positions in a THREADS-stride loop ----
    let t_out_us = t_out as usize;
    let t_in_us = t_in as usize;
    let stride_us = stride as usize;
    let padding_i = padding as i32;

    // Base offsets for this (b, c) row in x and y (row-major (B, C, T)).
    let x_base = b * channels_us * t_in_us + c * t_in_us;
    let y_base = b * channels_us * t_out_us + c * t_out_us;

    let mut t_o = tid;
    while t_o < t_out_us {
        let mut y_acc = 0f32;
        for k in 0..k_us {
            let t_in_signed = (t_o * stride_us) as i32 + k as i32 - padding_i;
            if t_in_signed >= 0 && (t_in_signed as usize) < t_in_us {
                // SAFETY: x_base + t_in_signed < B*C*T_in (all indices validated).
                let x_val = unsafe { *x_ptr.add(x_base + t_in_signed as usize) };
                // SAFETY: s_w[k] is within the K-element shared memory strip.
                let w_val = unsafe { *s_w.add(k) };
                y_acc += x_val * w_val;
            }
        }
        // Bias broadcast: bias[c] is the same for all t_o positions.
        if has_bias != 0 {
            // SAFETY: bias_ptr layout (C,); c < C; has_bias guard prevents
            //         reads when bias_ptr is a dummy 1-element allocation.
            y_acc += unsafe { *bias_ptr.add(c) };
        }
        // SAFETY: y_base + t_o < B*C*T_out (t_o < T_out and base validated).
        unsafe { *y_ptr.add(y_base + t_o) = y_acc };
        t_o += THREADS;
    }
}

/// Depthwise 1-D convolution backward w.r.t. input (groups == channels).
///
/// B4.4e-conv1d.
///
/// **Grid**: `(B * C, 1, 1)`, **Block**: `(128, 1, 1)`.
/// **Shared memory**: `K * 4` bytes (per-channel weight strip).
///
/// For stride == 1 (production):
///   `grad_x[b,c,t_i] = Σ_{k=0}^{K-1} grad_out[b,c, t_i+padding−k] · w[c,0,k]`
/// For general stride:
///   only positions `t_o = (t_i+padding−k) / stride` where the division is exact
///   and `t_o ∈ [0, T_out)` contribute.
///
/// # Layout
/// - `grad_out_ptr`: `(B, C, T_out)`
/// - `w_ptr`:        `(C, 1, K)`
/// - `grad_x_ptr`:   `(B, C, T_in)` — written, pre-allocated to zeros
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn depthwise_conv1d_bwd_x_f32(
    grad_out_ptr: *const f32,
    w_ptr: *const f32,
    grad_x_ptr: *mut f32,
    batch: u32,
    channels: u32,
    t_in: u32,
    t_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
) {
    const THREADS: usize = 128;

    // SAFETY: nvptx intrinsics always valid in ptx-kernel.
    let bid = unsafe { nvptx::_block_idx_x() } as usize;
    let tid = unsafe { nvptx::_thread_idx_x() } as usize;

    let channels_us = channels as usize;
    let batch_us = batch as usize;

    let c = bid % channels_us;
    let b = bid / channels_us;
    if b >= batch_us {
        return;
    }

    // Load K weights into shared memory.
    // SAFETY: SHARED_BUF has k_size*4 bytes; tid < k_size ≤ 128 for active threads.
    let k_us = k_size as usize;
    let s_w: *mut f32 = unsafe { shared_mem_base() };
    if tid < k_us {
        unsafe { *s_w.add(tid) = *w_ptr.add(c * k_us + tid) };
    }
    // SAFETY: syncthreads — weight load must complete before compute loop.
    unsafe { syncthreads() };

    let t_in_us = t_in as usize;
    let t_out_us = t_out as usize;
    let stride_us = stride as usize;
    let padding_i = padding as i32;

    let go_base = b * channels_us * t_out_us + c * t_out_us;
    let gx_base = b * channels_us * t_in_us + c * t_in_us;

    let mut t_i = tid;
    while t_i < t_in_us {
        let mut gx = 0f32;
        for k in 0..k_us {
            // Solve: t_o * stride = t_i + padding - k  →  t_o = (t_i + padding - k) / stride
            // Only valid when (t_i + padding - k) is non-negative and
            // divisible by stride.
            let num = t_i as i32 + padding_i - k as i32;
            if num < 0 {
                continue;
            }
            let num_u = num as usize;
            // For stride == 1 this branch is always taken (no modulo work).
            if stride_us > 1 && num_u % stride_us != 0 {
                continue;
            }
            let t_o = num_u / stride_us;
            if t_o >= t_out_us {
                continue;
            }
            // SAFETY: go_base + t_o < B*C*T_out (all indices validated).
            let go_val = unsafe { *grad_out_ptr.add(go_base + t_o) };
            // SAFETY: s_w[k] in K-element shared memory strip.
            let w_val = unsafe { *s_w.add(k) };
            gx += go_val * w_val;
        }
        // SAFETY: gx_base + t_i < B*C*T_in.
        unsafe { *grad_x_ptr.add(gx_base + t_i) = gx };
        t_i += THREADS;
    }
}

/// Depthwise 1-D convolution backward w.r.t. weight (groups == channels).
///
/// B4.4e-conv1d.
///
/// **Grid**: `(C, K, 1)` where K = `k_size` — one block per `(channel, kernel_pos)`.
/// **Block**: `(128, 1, 1)`.
/// **Shared memory**: `128 * 4 = 512` bytes for the parallel reduction.
///
/// Each block computes `grad_w[c, 0, k]` via a parallel reduction over the
/// `B × T_out` dimension:
///   `grad_w[c,0,k] = Σ_{b,t_o} grad_out[b,c,t_o] · x[b,c, t_o*stride−padding+k]`
/// (zero-padding for out-of-bounds `x` positions).
///
/// # Layout
/// - `grad_out_ptr`: `(B, C, T_out)`
/// - `x_ptr`:        `(B, C, T_in)`
/// - `grad_w_ptr`:   `(C, 1, K)` — written, pre-allocated to zeros
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn depthwise_conv1d_bwd_w_f32(
    grad_out_ptr: *const f32,
    x_ptr: *const f32,
    grad_w_ptr: *mut f32,
    batch: u32,
    channels: u32,
    t_in: u32,
    t_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
) {
    // SAFETY: nvptx intrinsics always valid.
    let c = unsafe { nvptx::_block_idx_x() } as usize;
    let k = unsafe { nvptx::_block_idx_y() } as usize;
    let tid = unsafe { nvptx::_thread_idx_x() } as usize;
    // Read the block size at **runtime** rather than hardcoding a `const`.
    //
    // B'.2e hang post-mortem: an earlier version used `const THREADS_W:
    // usize = 128` for the reduction loop below. Because that bound was a
    // compile-time constant, LLVM fully unrolled the `while red_stride > 0`
    // loop into 7 explicit blocks, and its last iteration's `if tid < 1`
    // (i.e. `tid == 0`) body sat immediately before this function's
    // (then-separate) `if tid == 0 { write output }` tail. LLVM's ordinary
    // tail-merging treated those as the same predicate and produced **two
    // different `bar.sync` instructions reached by disjoint thread
    // subsets** (`tid == 0` vs `tid != 0`) for what must be a single
    // barrier — confirmed on hardware: the kernel hung at 100% GPU
    // utilization indefinitely (`bar.sync` has no `convergent` marker via
    // a bare `asm!` block, so this transform is "legal" from LLVM's point
    // of view even though it breaks CUDA's barrier contract). Reading
    // `n_threads` from `_block_dim_x()` here makes the trip count opaque
    // to LLVM, so the loop is compiled as an actual loop (one `bar.sync`
    // call site reached identically by every thread on every iteration)
    // instead of being unrolled — see `rmsnorm_backward_x_f32` above,
    // which uses the same technique and was verified to compile to just 2
    // `bar.sync` sites total. Host launcher always passes a power-of-two
    // `block_dim.x` (128 — see `conv1d_depthwise_bwd_w_gpu`).
    let n_threads = unsafe { nvptx::_block_dim_x() } as usize;

    let channels_us = channels as usize;
    let t_in_us = t_in as usize;
    let t_out_us = t_out as usize;
    let stride_us = stride as usize;
    let padding_i = padding as i32;
    let batch_us = batch as usize;
    let k_size_us = k_size as usize;

    // Guard: host sets grid_dim.y = k_size, so k < k_size normally.
    // Guard: host sets grid_dim.x = channels.
    if c >= channels_us || k >= k_size_us {
        return;
    }

    // ---- Step 1: each thread computes its partial sum ----
    // Thread tid sums over t_o = tid, tid + n_threads, tid + 2*n_threads, …
    let go_chan_base = c * t_out_us;
    let x_chan_base = c * t_in_us;

    let mut acc = 0f32;
    for b in 0..batch_us {
        let go_batch_off = b * channels_us * t_out_us + go_chan_base;
        let x_batch_off = b * channels_us * t_in_us + x_chan_base;
        let mut t_o = tid;
        while t_o < t_out_us {
            let t_in_signed = (t_o * stride_us) as i32 + k as i32 - padding_i;
            if t_in_signed >= 0 && (t_in_signed as usize) < t_in_us {
                let t_i = t_in_signed as usize;
                // SAFETY: offsets within (B,C,T_out) and (B,C,T_in) bounds.
                let go_val = unsafe { *grad_out_ptr.add(go_batch_off + t_o) };
                let x_val = unsafe { *x_ptr.add(x_batch_off + t_i) };
                acc += go_val * x_val;
            }
            t_o += n_threads;
        }
    }

    // ---- Step 2: parallel reduction in shared memory ----
    // SAFETY: SHARED_BUF has n_threads*4 bytes (host sizes it to match); tid < n_threads.
    let s_partial: *mut f32 = unsafe { shared_mem_base() };
    unsafe { *s_partial.add(tid) = acc };
    // SAFETY: syncthreads — all threads must write partial sum before reduction.
    unsafe { syncthreads() };

    // Standard log2 tree reduction (n_threads must be a power of 2). See the
    // doc comment above `n_threads` for why this must stay a runtime value.
    let mut red_stride = n_threads / 2;
    while red_stride > 0 {
        if tid < red_stride {
            let v = unsafe { *s_partial.add(tid + red_stride) };
            let old = unsafe { *s_partial.add(tid) };
            unsafe { *s_partial.add(tid) = old + v };
        }
        // SAFETY: syncthreads — all threads complete current reduction step
        // before the next stride halves the active set.
        unsafe { syncthreads() };
        red_stride /= 2;
    }

    // ---- Step 3: thread 0 writes the result ----
    if tid == 0 {
        let total = unsafe { *s_partial };
        // SAFETY: grad_w layout (C, 1, K); c < C and k < K (guarded above).
        //         flat index c * k_size + k (C_in_per_group == 1 so dim-1 is 1).
        unsafe { *grad_w_ptr.add(c * k_size_us + k) = total };
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
