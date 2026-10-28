//! A loaded model ready to generate text: the transformer plus the way its
//! family turns text into tokens and back. Shared by the `aether` CLI and
//! the C ABI (`ffi`) the Android app calls.

use crate::format::gguf::Value;
use crate::format::json::Json;
use crate::model::sampler::Sampler;
use crate::model::{Transformer, gguf, hf, llama2c};
use crate::pool::ThreadPool;
use crate::tokenizer::{bytelevel, llama2c as l2c_tok};
use crate::weight::Format;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// How a model family turns text into tokens and back, and which tokens
/// end a generation.
pub enum Text {
    /// llama2.c: BOS first; BOS again means "story over".
    Llama2c(l2c_tok::Tokenizer),
    /// Hugging Face byte-level BPE (SmolLM2, Qwen2): no BOS added; stops at
    /// end-of-text or the chat end-of-turn token.
    Hf { tok: Box<bytelevel::Tokenizer>, start: usize, stops: Vec<usize> },
}

impl Text {
    pub fn encode(&self, prompt: &str) -> Vec<usize> {
        match self {
            Text::Llama2c(t) => t.encode(prompt, true, false),
            Text::Hf { tok, start, .. } => {
                let ids = tok.encode(prompt);
                if ids.is_empty() { vec![*start] } else { ids }
            }
        }
    }

    /// Bytes to print for `next` following `prev`. (Bytes, not text: one
    /// character can be split across tokens.)
    pub fn piece(&self, prev: usize, next: usize) -> Vec<u8> {
        match self {
            Text::Llama2c(t) => {
                let p = t.decode(prev, next);
                // Like llama2.c's safe_printf: skip lone non-printable bytes.
                if p.len() == 1 && !(p[0].is_ascii_graphic() || p[0].is_ascii_whitespace()) { Vec::new() } else { p }
            }
            Text::Hf { tok, .. } => tok.token_bytes(next),
        }
    }

    /// Bytes for the first prompt token, which has no predecessor: nothing
    /// for llama2.c (it is BOS), the token itself otherwise.
    pub fn first_piece(&self, id: usize) -> Vec<u8> {
        match self {
            Text::Llama2c(_) => Vec::new(),
            Text::Hf { tok, .. } => tok.token_bytes(id),
        }
    }

    /// The token llama.cpp puts at the start of each perplexity chunk, if
    /// the tokenizer adds one: BOS for llama2.c, none for SmolLM2 / Qwen2.
    pub fn chunk_bos(&self) -> Option<usize> {
        match self {
            Text::Llama2c(_) => Some(l2c_tok::BOS),
            Text::Hf { .. } => None,
        }
    }

    pub fn is_stop(&self, id: usize) -> bool {
        match self {
            Text::Llama2c(_) => id == l2c_tok::BOS,
            Text::Hf { stops, .. } => stops.contains(&id),
        }
    }
}

/// Timings of one generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Prompt tokens processed in the batched prefill.
    pub prefill_tokens: usize,
    pub prefill: Duration,
    /// Forward passes after the prefill (one per generated token after the
    /// first; the first comes from the prefill's logits).
    pub decode_tokens: usize,
    pub decode: Duration,
}

/// Counts from [`Session::generate_lookup`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Lookup {
    /// Batched verification passes (each also yields one certain token).
    pub passes: usize,
    /// Draft tokens proposed, and how many of them the model agreed with.
    pub drafted: usize,
    pub accepted: usize,
}

/// The tokens that followed the most recent earlier occurrence of the last
/// `ngram` tokens of `hist`, at most `k` of them (prompt lookup decoding:
/// text often repeats what came before, and then the past is a free draft).
fn lookup_draft(hist: &[usize], ngram: usize, k: usize) -> &[usize] {
    if hist.len() <= ngram {
        return &[];
    }
    let tail = &hist[hist.len() - ngram..];
    // Latest match first; a match must leave at least one token after it.
    for i in (0..hist.len() - ngram).rev() {
        if &hist[i..i + ngram] == tail {
            let from = i + ngram;
            return &hist[from..(from + k).min(hist.len())];
        }
    }
    &[]
}

pub struct Session {
    pub model: Transformer,
    pub text: Text,
}

impl Session {
    /// Loads a `.gguf` file, a Hugging Face directory, or a llama2.c `.bin`
    /// (with its `tokenizer.bin` at `l2c_tokenizer`, default next to it),
    /// converts the matrices to `format`, and spreads work over `cores`.
    pub fn open(path: &Path, l2c_tokenizer: Option<&Path>, max_seq: usize, format: Format, cores: Option<&[usize]>) -> io::Result<Session> {
        let (config, weights, text) = if path.extension().is_some_and(|e| e == "gguf") {
            let (config, weights, g) = gguf::load(path, max_seq)?;
            let tok = bytelevel::Tokenizer::from_gguf(&g)?;
            let id = |k: &str| g.get(k).and_then(Value::as_u64).map(|v| v as usize);
            let mut stops: Vec<usize> = id("tokenizer.ggml.eos_token_id").into_iter().collect();
            stops.extend(tok.token_id("<|im_end|>"));
            stops.extend(tok.token_id("<|endoftext|>"));
            let start = id("tokenizer.ggml.bos_token_id").or(stops.first().copied()).unwrap_or(0);
            (config, weights, Text::Hf { tok: Box::new(tok), start, stops })
        } else if path.is_dir() {
            let (config, weights) = hf::load(path, max_seq)?;
            let tok = bytelevel::Tokenizer::load(path.join("tokenizer.json"))?;
            let cfg = Json::parse(&std::fs::read_to_string(path.join("config.json"))?).unwrap_or(Json::Null);
            let id = |k: &str| cfg.get(k).and_then(Json::as_usize);
            let mut stops: Vec<usize> = id("eos_token_id").into_iter().collect();
            stops.extend(tok.token_id("<|im_end|>"));
            let start = id("bos_token_id").or(stops.first().copied()).unwrap_or(0);
            (config, weights, Text::Hf { tok: Box::new(tok), start, stops })
        } else {
            let (mut config, weights) = llama2c::load(path)?;
            config.seq_len = config.seq_len.min(max_seq);
            let tok_path = l2c_tokenizer.map(Path::to_path_buf).unwrap_or_else(|| path.with_file_name("tokenizer.bin"));
            let tok = l2c_tok::Tokenizer::load(&tok_path, config.vocab_size)?;
            (config, weights, Text::Llama2c(tok))
        };
        let weights = weights.convert(&config, format);
        let mut model = Transformer::new(config, weights);
        if let Some(cores) = cores {
            // One thread per core; the calling thread becomes the first.
            model.set_pool(ThreadPool::new(cores)?);
        }
        Ok(Session { model, text })
    }

    /// Generates from `prompt` for at most `steps` positions (prompt
    /// included) or until a stop token, passing each piece of text to
    /// `emit` as soon as it exists; `emit` returning `false` stops early.
    /// With `echo_prompt`, the prompt's own pieces are emitted first.
    pub fn generate(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, echo_prompt: bool, mut emit: impl FnMut(&[u8]) -> bool) -> Stats {
        let steps = steps.min(self.model.config.seq_len);
        let tokens = self.text.encode(prompt);
        let mut stats = Stats::default();
        if echo_prompt && !emit(&self.text.first_piece(tokens[0])) {
            return stats;
        }
        // Prefill: the whole prompt in one batched pass. Its logits are those
        // after the last prompt token, so they already give the first
        // generated token: time to first token is the prefill time.
        let np = tokens.len().min(steps);
        let t0 = Instant::now();
        let mut logits = self.model.forward_batch(&tokens[..np], 0).to_vec();
        stats.prefill = t0.elapsed();
        stats.prefill_tokens = np;
        if echo_prompt {
            for p in 0..np - 1 {
                if !emit(&self.text.piece(tokens[p], tokens[p + 1])) {
                    return stats;
                }
            }
        }
        // Decode: one forward pass per new token.
        let t1 = Instant::now();
        let mut pos = np;
        if np == tokens.len() {
            let mut token = tokens[np - 1];
            loop {
                let next = sampler.sample(&mut logits);
                if self.text.is_stop(next) || !emit(&self.text.piece(token, next)) {
                    break;
                }
                token = next;
                if pos >= steps {
                    break;
                }
                logits.copy_from_slice(self.model.forward(token, pos));
                pos += 1;
            }
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        stats
    }
}

impl Session {
    /// Greedy generation with prompt lookup decoding: after each certain
    /// token, up to `k` draft tokens are copied from the most recent earlier
    /// place where the last `ngram` tokens occurred ([`lookup_draft`]), and
    /// the certain token plus the drafts go through the model in one batched
    /// pass. Row r of its logits predicts the token after input r, so drafts
    /// are accepted while each equals the argmax of the row before it; the
    /// first disagreement's argmax is the next certain token. All weights
    /// are read once per pass, so every accepted draft is a token for (almost)
    /// free. Cache entries of rejected drafts are overwritten later.
    ///
    /// The tokens are those of greedy decoding, up to rounding: a batched
    /// pass computes the same logits by another summation order, so where two
    /// tokens are within rounding of each other the choice can differ.
    pub fn generate_lookup(&mut self, prompt: &str, steps: usize, k: usize, ngram: usize, mut emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        let steps = steps.min(self.model.config.seq_len);
        let mut hist = self.text.encode(prompt);
        let (mut stats, mut look) = (Stats::default(), Lookup::default());
        let np = hist.len().min(steps);
        let t0 = Instant::now();
        let mut next = crate::kernels::argmax(self.model.forward_batch(&hist[..np], 0));
        stats.prefill = t0.elapsed();
        stats.prefill_tokens = np;
        let t1 = Instant::now();
        // `pos`: positions in the cache; `next` is certain but not yet in it.
        let mut pos = np;
        if np == hist.len() {
            'tokens: loop {
                if self.text.is_stop(next) || !emit(&self.text.piece(hist[pos - 1], next)) || pos >= steps {
                    break;
                }
                hist.push(next);
                let draft: Vec<usize> = lookup_draft(&hist, ngram, k.min(steps - pos - 1)).to_vec();
                if draft.is_empty() {
                    next = crate::kernels::argmax(self.model.forward(next, pos));
                    pos += 1;
                    continue;
                }
                let input: Vec<usize> = std::iter::once(next).chain(draft.iter().copied()).collect();
                let vocab = self.model.config.vocab_size;
                let logits = self.model.forward_batch_logits(&input, pos, 0);
                let best: Vec<usize> = logits.chunks_exact(vocab).map(crate::kernels::argmax).collect();
                look.passes += 1;
                look.drafted += draft.len();
                pos += 1;
                // best[r] follows input[r]; draft[r] = input[r + 1].
                let mut r = 0;
                while r < draft.len() && draft[r] == best[r] {
                    let d = draft[r];
                    if self.text.is_stop(d) || !emit(&self.text.piece(hist[pos - 1], d)) {
                        break 'tokens;
                    }
                    look.accepted += 1;
                    hist.push(d);
                    pos += 1;
                    r += 1;
                }
                next = best[r];
            }
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        (stats, look)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;

    /// The session reproduces llama2.c's greedy text byte for byte, and
    /// stops early when `emit` says so.
    #[test]
    fn generates_llama2c_text_and_stops_on_request() {
        let (Some(model), Some(want)) = (golden("models/stories15M/stories15M.bin"), golden("stories15M/greedy.txt")) else { return };
        let mut s = Session::open(&model, None, 256, Format::F32, None).unwrap();
        let mut out = Vec::new();
        let stats = s.generate("Once upon a time", &mut Sampler::new(0.0, 0.9, 0, 1), 64, true, |b| {
            out.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8(out).unwrap() + "\n", std::fs::read_to_string(want).unwrap());
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 64);
        let mut pieces = 0;
        s.generate("Once upon a time", &mut Sampler::new(0.0, 0.9, 0, 1), 64, false, |_| {
            pieces += 1;
            pieces < 3
        });
        assert_eq!(pieces, 3);
    }

    /// Prompt lookup decoding gives llama2.c's greedy text too (the story
    /// repeats names and phrases, so drafts are proposed and some accepted).
    #[test]
    fn lookup_decoding_gives_llama2c_greedy_text() {
        let (Some(model), Some(want)) = (golden("models/stories15M/stories15M.bin"), golden("stories15M/greedy.txt")) else { return };
        let mut s = Session::open(&model, None, 256, Format::F32, None).unwrap();
        let mut out = s.text.first_piece(s.text.encode("Once upon a time")[0]);
        let prompt_tokens = s.text.encode("Once upon a time");
        for p in 0..prompt_tokens.len() - 1 {
            out.extend(s.text.piece(prompt_tokens[p], prompt_tokens[p + 1]));
        }
        let (stats, look) = s.generate_lookup("Once upon a time", 64, 4, 1, |b| {
            out.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8(out).unwrap() + "\n", std::fs::read_to_string(want).unwrap());
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 64);
        assert!(look.drafted > 0 && look.accepted > 0, "{look:?}");
    }
}
