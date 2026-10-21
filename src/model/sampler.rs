//! Choosing the next token from the logits.
//!
//! - **Greedy** (temperature 0): always the most likely token. Deterministic;
//!   used for every correctness comparison.
//! - **Temperature**: divide the logits by T before softmax. T < 1 sharpens
//!   the distribution, T > 1 flattens it.
//! - **Top-k**: keep only the k most likely tokens, renormalise, sample.
//! - **Top-p** (nucleus): keep the smallest set of most likely tokens whose
//!   probabilities add up to more than p, renormalise, sample. Unlike top-k
//!   the set grows when the model is unsure and shrinks when it is confident.
//!
//! The random number generator and the top-p procedure reproduce llama2.c's
//! (`xorshift*`, same cutoff and tie handling), so for the same seed this
//! sampler picks the same tokens as `run -s SEED` — a sampled run can be
//! checked against the reference too, not just greedy decoding.

use crate::kernels::scalar::{argmax, softmax};

pub struct Sampler {
    pub temperature: f32,
    /// Nucleus threshold; `<= 0` or `>= 1` disables top-p.
    pub top_p: f32,
    /// Keep only the k most likely tokens; 0 disables top-k.
    pub top_k: usize,
    rng: u64,
    /// Scratch for top-k / top-p: (probability, token id).
    candidates: Vec<(f32, usize)>,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Sampler {
        Sampler { temperature, top_p, top_k, rng: seed, candidates: Vec::new() }
    }

    /// llama2.c's `random_u32`: xorshift with a multiplicative output step.
    fn next_u32(&mut self) -> u32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in [0, 1): the top 24 bits of a u32, which an f32 holds exactly.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Picks the next token. Modifies `logits` in place (scaled, then turned
    /// into probabilities), as llama2.c does.
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }
        for l in logits.iter_mut() {
            *l /= self.temperature;
        }
        softmax(logits);
        let coin = self.next_f32();
        if self.top_k > 0 && self.top_k < logits.len() {
            return self.sample_top_k(logits, coin);
        }
        if self.top_p <= 0.0 || self.top_p >= 1.0 {
            return sample_cdf(logits.iter().copied().enumerate().map(|(i, p)| (p, i)), 1.0, coin);
        }
        self.sample_top_p(logits, coin)
    }

    fn sample_top_k(&mut self, probs: &[f32], coin: f32) -> usize {
        self.candidates.clear();
        self.candidates.extend(probs.iter().copied().enumerate().map(|(i, p)| (p, i)));
        let k = self.top_k;
        // Partition so the k largest come first, then order just those.
        self.candidates.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        self.candidates.truncate(k);
        self.candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mass: f32 = self.candidates.iter().map(|c| c.0).sum();
        sample_cdf(self.candidates.iter().copied(), mass, coin)
    }

    /// llama2.c's `sample_topp`.
    fn sample_top_p(&mut self, probs: &[f32], coin: f32) -> usize {
        let n = probs.len();
        // Tokens below this probability cannot be in the nucleus (if they
        // were, the n-1 others would hold less than p), so they are dropped
        // before sorting: a 32000-token sort becomes a few-hundred-token one.
        let cutoff = (1.0 - self.top_p) / (n - 1) as f32;
        self.candidates.clear();
        self.candidates.extend(probs.iter().copied().enumerate().filter(|&(_, p)| p >= cutoff).map(|(i, p)| (p, i)));
        self.candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        // Shortest prefix whose mass exceeds top_p.
        let mut mass = 0.0f32;
        let mut last = self.candidates.len() - 1;
        for (i, c) in self.candidates.iter().enumerate() {
            mass += c.0;
            if mass > self.top_p {
                last = i;
                break;
            }
        }
        sample_cdf(self.candidates[..=last].iter().copied(), mass, coin)
    }
}

/// Walks the cumulative distribution of `(prob, token)` pairs until it
/// passes `coin * mass`. Falls back to the last token if rounding leaves the
/// sum just short.
fn sample_cdf(items: impl Iterator<Item = (f32, usize)>, mass: f32, coin: f32) -> usize {
    let r = coin * mass;
    let mut cdf = 0.0f32;
    let mut last = 0;
    for (p, i) in items {
        cdf += p;
        last = i;
        if r < cdf {
            return i;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_is_argmax() {
        let mut s = Sampler::new(0.0, 0.9, 0, 1);
        assert_eq!(s.sample(&mut [0.1, 3.0, 2.9, -1.0]), 1);
    }

    /// k = 1 leaves only the most likely token, whatever the coin.
    #[test]
    fn top_k_one_is_argmax() {
        let mut s = Sampler::new(1.0, 0.0, 1, 7);
        for _ in 0..20 {
            assert_eq!(s.sample(&mut [0.5, 0.1, 2.0, 1.9]), 2);
        }
    }

    /// One token with almost all the mass: top-p = 0.5 keeps only it.
    #[test]
    fn top_p_keeps_only_the_nucleus() {
        let mut s = Sampler::new(1.0, 0.5, 0, 3);
        for _ in 0..20 {
            assert_eq!(s.sample(&mut [0.0, 10.0, 0.0, 0.0]), 1);
        }
    }

    /// The generator's output is in [0, 1) and not constant.
    #[test]
    fn rng_range() {
        let mut s = Sampler::new(1.0, 0.0, 0, 42);
        let v: Vec<f32> = (0..1000).map(|_| s.next_f32()).collect();
        assert!(v.iter().all(|&x| (0.0..1.0).contains(&x)));
        assert!(v.iter().any(|&x| x != v[0]));
    }
}
