//! Text scoring: perplexity, and KL divergence against a reference model.
//!
//! **Perplexity** measures how surprised a model is by real text: the
//! exponential of the average negative log-probability it gives each actual
//! next token. Lower is better; a model that guessed uniformly among V tokens
//! would score V.
//!
//! **KL divergence** compares two models token by token: KL(P‖Q) =
//! Σ p(v)·(log p(v) − log q(v)) over the vocabulary, where P is the f32
//! model and Q the quantized one. Zero means identical predictions. Unlike
//! perplexity it also counts changes that happen not to affect the actual
//! next token. "Same top" is how often both models' most likely token agree.
//!
//! The procedure follows llama.cpp's `llama-perplexity` exactly, so the
//! numbers can be compared with it: split the token stream into chunks of
//! `n_ctx`, evaluate each chunk from an empty cache, and score only the
//! second half of each chunk (positions `n_ctx/2 .. n_ctx-1` predicting the
//! next token), where every prediction has at least `n_ctx/2` tokens of
//! context.

use crate::model::Transformer;

#[derive(Debug, Default, Clone)]
pub struct Score {
    /// Tokens scored.
    pub count: usize,
    /// Sum of −log p(actual next token), natural log.
    pub nll: f64,
    pub nll2: f64,
    /// KL divergence sum against the base model (if one was given).
    pub kld: f64,
    /// Predictions whose argmax equals the base model's.
    pub same_top: usize,
    /// Sum of −log p under the base model.
    pub nll_base: f64,
}

impl Score {
    pub fn ppl(&self) -> f64 {
        (self.nll / self.count as f64).exp()
    }

    /// ± one standard error of the perplexity, as llama.cpp prints it.
    pub fn ppl_err(&self) -> f64 {
        let n = self.count as f64;
        let (mean, mean2) = (self.nll / n, self.nll2 / n);
        ((mean2 - mean * mean).max(0.0) / (n - 1.0)).sqrt() * self.ppl()
    }

    pub fn ppl_base(&self) -> f64 {
        (self.nll_base / self.count as f64).exp()
    }

    pub fn mean_kld(&self) -> f64 {
        self.kld / self.count as f64
    }

    pub fn same_top_pct(&self) -> f64 {
        100.0 * self.same_top as f64 / self.count as f64
    }
}

/// Natural-log probabilities of one row of logits, in f64. Sums `exp` over
/// the row after subtracting the max (so nothing overflows), as llama.cpp's
/// `log_softmax` does.
pub fn log_probs(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let lse = sum.ln();
    logits.iter().map(|&l| (l - max) as f64 - lse).collect()
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Scores `tokens` in `n_chunks` chunks of `n_ctx` (or as many as fit).
/// With `bos`, each chunk's first token is replaced by it (llama.cpp does
/// this for models whose tokenizer adds a BOS token).
/// With `base`, also accumulates KL divergence and top-token agreement of
/// `model` against `base` (both must be able to hold `n_ctx` positions).
/// `progress(chunk, running)` is called after every chunk.
pub fn perplexity(
    model: &mut Transformer,
    mut base: Option<&mut Transformer>,
    tokens: &[usize],
    bos: Option<usize>,
    n_ctx: usize,
    n_chunks: usize,
    mut progress: impl FnMut(usize, &Score),
) -> Score {
    let first = n_ctx / 2;
    let chunks = (tokens.len() / n_ctx).min(n_chunks);
    let vocab = model.config.vocab_size;
    let mut score = Score::default();
    for ch in 0..chunks {
        let mut chunk = tokens[ch * n_ctx..(ch + 1) * n_ctx].to_vec();
        if let Some(bos) = bos {
            chunk[0] = bos;
        }
        let chunk = &chunk[..];
        // Rows predict the tokens after positions first..n_ctx-1; the last
        // row has no next token inside the chunk and is not scored.
        let ours = model.forward_batch_logits(chunk, 0, first).to_vec();
        let reference = base.as_deref_mut().map(|b| b.forward_batch_logits(chunk, 0, first).to_vec());
        for r in 0..(n_ctx - 1 - first) {
            let next = chunk[first + r + 1];
            let lp = log_probs(&ours[r * vocab..(r + 1) * vocab]);
            let nll = -lp[next];
            score.nll += nll;
            score.nll2 += nll * nll;
            score.count += 1;
            if let Some(reference) = &reference {
                let lb = log_probs(&reference[r * vocab..(r + 1) * vocab]);
                score.nll_base -= lb[next];
                score.kld += lb.iter().zip(&lp).map(|(&b, &q)| b.exp() * (b - q)).sum::<f64>();
                score.same_top += usize::from(argmax(&lb) == argmax(&lp));
            }
        }
        progress(ch + 1, &score);
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_probs_normalise() {
        let lp = log_probs(&[1.0, 2.0, 3.0, 1000.0]);
        let total: f64 = lp.iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(lp[3] > -1e-9); // the huge logit takes (almost) all the mass
    }

    /// A model compared with itself: KL exactly 0, same top 100%.
    #[test]
    fn self_comparison_is_zero() {
        let Some(path) = crate::testutil::golden("models/stories15M/stories15M.bin") else { return };
        let (c, w) = crate::model::llama2c::load(&path).unwrap();
        let (c2, w2) = crate::model::llama2c::load(&path).unwrap();
        let (mut a, mut b) = (Transformer::new(c, w), Transformer::new(c2, w2));
        let tokens: Vec<usize> = (0..128).map(|i| 100 + (i * 37) % 5000).collect();
        let s = perplexity(&mut a, Some(&mut b), &tokens, Some(1), 64, 2, |_, _| {});
        assert_eq!(s.count, 2 * 31);
        assert!(s.mean_kld().abs() < 1e-12 && s.same_top == s.count);
        assert!((s.ppl() - s.ppl_base()).abs() < 1e-9);
    }
}
