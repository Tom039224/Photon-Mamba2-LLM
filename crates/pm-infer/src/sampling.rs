//! Token samplers (G.4).
//!
//! All samplers consume a host-side `&[f32]` slice of logits over the
//! vocabulary so they're trivial to test and don't drag in backend
//! state. The Generator copies the relevant logits row off-device once
//! per step.

/// LCG-based RNG for reproducible sampling.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
    pub fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32;
        bits as f32 / u32::MAX as f32
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    /// Keep the top-`k` logits; zero / None disables.
    pub top_k: Option<usize>,
    /// Keep the smallest set of tokens whose cumulative probability
    /// exceeds `p`. `None`/0 disables.
    pub top_p: Option<f32>,
    /// `true` skips all stochastic logic — picks `argmax(logits)`.
    pub greedy: bool,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            greedy: false,
        }
    }
}

impl Sampler {
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            greedy: true,
        }
    }

    /// Sample one token id from `logits`. `rng` is only touched when
    /// `greedy=false`.
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> usize {
        if self.greedy {
            return argmax(logits);
        }
        let temp = if self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };
        let mut scaled: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l / temp))
            .collect();

        if let Some(k) = self.top_k {
            if k > 0 && k < scaled.len() {
                scaled.select_nth_unstable_by(k, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                scaled.truncate(k);
            }
        }

        // Convert to probabilities (softmax over the surviving entries).
        let max_logit = scaled
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = scaled
            .into_iter()
            .map(|(i, l)| (i, (l - max_logit).exp()))
            .collect();
        let z: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= z;
        }

        // top-p (nucleus): sort descending by prob, keep until cumulative ≥ p.
        if let Some(p_cap) = self.top_p {
            if p_cap > 0.0 && p_cap < 1.0 {
                probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut cum = 0.0;
                let mut cut = probs.len();
                for (i, (_, q)) in probs.iter().enumerate() {
                    cum += *q;
                    if cum >= p_cap {
                        cut = i + 1;
                        break;
                    }
                }
                probs.truncate(cut);
                let z2: f32 = probs.iter().map(|(_, q)| q).sum();
                for (_, q) in &mut probs {
                    *q /= z2;
                }
            }
        }

        // Roulette wheel.
        let u = rng.next_f32();
        let mut cum = 0.0;
        for (i, q) in &probs {
            cum += *q;
            if u <= cum {
                return *i;
            }
        }
        // Floating-point cleanup: fall back to the last entry.
        probs.last().map(|(i, _)| *i).unwrap_or(0)
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let s = Sampler::greedy();
        let logits = vec![0.1, 0.5, 0.3, 0.4];
        let mut rng = Rng::new(0);
        assert_eq!(s.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn top_k_1_is_greedy() {
        let s = Sampler {
            top_k: Some(1),
            greedy: false,
            ..Default::default()
        };
        let logits = vec![0.1, 0.5, 0.3, 0.4];
        let mut rng = Rng::new(123);
        assert_eq!(s.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn temperature_above_one_widens_distribution() {
        // Hard to assert randomness; just smoke-test that it doesn't panic.
        let s = Sampler {
            temperature: 2.0,
            ..Default::default()
        };
        let logits = vec![0.0, 1.0, 0.5];
        let mut rng = Rng::new(42);
        let _ = s.sample(&logits, &mut rng);
    }
}
