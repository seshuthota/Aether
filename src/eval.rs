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

    /// Scores one prediction: `lp` the model's log-probabilities, `next`
    /// the actual next token, `base` the base model's log-probabilities
    /// for the same position (if there is a base model).
    pub fn add(&mut self, lp: &[f64], next: usize, base: Option<&[f64]>) {
        let nll = -lp[next];
        self.nll += nll;
        self.nll2 += nll * nll;
        self.count += 1;
        if let Some(lb) = base {
            self.nll_base -= lb[next];
            self.kld += lb.iter().zip(lp).map(|(&b, &q)| b.exp() * (b - q)).sum::<f64>();
            self.same_top += usize::from(argmax(lb) == argmax(lp));
        }
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
            let lb = reference.as_ref().map(|reference| log_probs(&reference[r * vocab..(r + 1) * vocab]));
            score.add(&lp, next, lb.as_deref());
        }
        progress(ch + 1, &score);
    }
    score
}

/// The log-probabilities `llama-perplexity --kl-divergence-base FILE`
/// saves, read a chunk at a time (`tools/perplexity/perplexity.cpp`):
/// `"_logits_"`, `n_ctx: u32`, `n_vocab: i32`, `n_chunk: i32`, the chunks'
/// tokens (`n_chunk · n_ctx` i32, before the BOS substitution), then per
/// chunk the `n_ctx − 1 − n_ctx/2` scored rows. A row is `nv = 2⌈n_vocab/2⌉
/// + 4` u16: two f32 (`scale`, `min_log_prob`), then one u16 `q` per token
/// with `log p = scale · q + min_log_prob`. llama.cpp keeps only the 16
/// below the row's largest logit; lower logits are stored as 0, so their
/// log-probability is lifted to the row's floor.
pub struct BaseFile {
    file: std::io::BufReader<std::fs::File>,
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub n_chunk: usize,
    /// Every chunk's tokens, as llama.cpp tokenized the text.
    pub tokens: Vec<usize>,
    data_at: u64,
}

impl BaseFile {
    pub fn open(path: &std::path::Path) -> std::io::Result<BaseFile> {
        use std::io::Read;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 20];
        file.read_exact(&mut head)?;
        if &head[..8] != b"_logits_" {
            return Err(bad("not a llama-perplexity --kl-divergence-base file"));
        }
        let word = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().unwrap()) as usize;
        let (n_ctx, n_vocab, n_chunk) = (word(8), word(12), word(16));
        let mut raw = vec![0u8; n_chunk * n_ctx * 4];
        file.read_exact(&mut raw)?;
        let tokens = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).collect();
        Ok(BaseFile { file, n_ctx, n_vocab, n_chunk, tokens, data_at: 20 + raw.len() as u64 })
    }

    /// u16 values per row.
    fn nv(&self) -> usize {
        2 * self.n_vocab.div_ceil(2) + 4
    }

    /// Chunk `ch`'s rows, raw (`[n_ctx − 1 − n_ctx/2][nv]`).
    pub fn chunk(&mut self, ch: usize) -> std::io::Result<Vec<u16>> {
        use std::io::{Read, Seek, SeekFrom};
        let rows = self.n_ctx - 1 - self.n_ctx / 2;
        let len = rows * self.nv();
        self.file.seek(SeekFrom::Start(self.data_at + (ch * len * 2) as u64))?;
        let mut raw = vec![0u8; len * 2];
        self.file.read_exact(&mut raw)?;
        Ok(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }
}

impl Score {
    /// Scores one prediction against a row of a [`BaseFile`], as
    /// llama.cpp's `--kl-divergence` does: the base's log-probabilities are
    /// the 16-bit values, KL sums only the tokens whose base log-probability
    /// is above −16 (the rest are the lifted floor), and the base's top token
    /// is its first largest value.
    pub fn add_base_row(&mut self, lp: &[f64], next: usize, row: &[u16]) {
        let nll = -lp[next];
        self.nll += nll;
        self.nll2 += nll * nll;
        self.count += 1;
        let f = |i: usize| f32::from_bits(row[2 * i] as u32 | (row[2 * i + 1] as u32) << 16);
        let (scale, min) = (f(0), f(1));
        let q = &row[4..4 + lp.len()];
        self.nll_base -= (scale * q[next] as f32 + min) as f64;
        let (mut kld, mut top, mut top_lp) = (0.0f64, 0, f32::NEG_INFINITY);
        for (i, (&qi, &l)) in q.iter().zip(lp).enumerate() {
            let b = scale * qi as f32 + min;
            if b > top_lp {
                (top, top_lp) = (i, b);
            }
            if b > -16.0 {
                kld += (b as f64).exp() * (b as f64 - l);
            }
        }
        self.kld += kld;
        self.same_top += usize::from(top == argmax(lp));
    }
}

/// Perplexity and KL of a model against llama.cpp's base file, on the
/// file's own tokens: `logits(chunk, first)` returns the logits of rows
/// `first..n_ctx` for one chunk (any backend). The chunk's first token is
/// replaced by `bos`, as llama.cpp did when it wrote the file.
pub fn perplexity_vs_base(
    base: &mut BaseFile,
    bos: Option<usize>,
    n_chunks: usize,
    mut logits: impl FnMut(&[usize], usize) -> Vec<f32>,
    mut progress: impl FnMut(usize, &Score),
) -> std::io::Result<Score> {
    let (n_ctx, vocab) = (base.n_ctx, base.n_vocab);
    let (first, nv) = (n_ctx / 2, base.nv());
    let mut score = Score::default();
    for ch in 0..base.n_chunk.min(n_chunks) {
        let mut chunk = base.tokens[ch * n_ctx..(ch + 1) * n_ctx].to_vec();
        if let Some(bos) = bos {
            chunk[0] = bos;
        }
        let ours = logits(&chunk, first);
        let rows = base.chunk(ch)?;
        for r in 0..(n_ctx - 1 - first) {
            let lp = log_probs(&ours[r * vocab..(r + 1) * vocab]);
            score.add_base_row(&lp, chunk[first + r + 1], &rows[r * nv..(r + 1) * nv]);
        }
        progress(ch + 1, &score);
    }
    Ok(score)
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
