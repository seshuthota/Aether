//! A loaded model ready to generate text: the transformer plus the way its
//! family turns text into tokens and back. Shared by the `aether` CLI and
//! the C ABI (`ffi`) the Android app calls.

use crate::format::gguf::Value;
use crate::format::imatrix::Imatrix;
use crate::format::json::Json;
use crate::kernels::argmax;
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
    /// The tokenizer and stop tokens a GGUF file's metadata describes.
    pub fn from_gguf(g: &crate::format::gguf::Gguf) -> io::Result<Text> {
        let tok = bytelevel::Tokenizer::from_gguf(g)?;
        let id = |k: &str| g.get(k).and_then(Value::as_u64).map(|v| v as usize);
        let mut stops: Vec<usize> = id("tokenizer.ggml.eos_token_id").into_iter().collect();
        stops.extend(tok.token_id("<|im_end|>"));
        stops.extend(tok.token_id("<|endoftext|>"));
        let start = id("tokenizer.ggml.bos_token_id").or(stops.first().copied()).unwrap_or(0);
        Ok(Text::Hf { tok: Box::new(tok), start, stops })
    }

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
        Session::open_imatrix(path, l2c_tokenizer, max_seq, format, None, cores)
    }

    /// [`Session::open`], quantizing with an importance matrix (see
    /// [`Weights::convert`](crate::model::transformer::Weights::convert)).
    pub fn open_imatrix(path: &Path, l2c_tokenizer: Option<&Path>, max_seq: usize, format: Format, imatrix: Option<&Imatrix>, cores: Option<&[usize]>) -> io::Result<Session> {
        let (config, weights, text) = if path.extension().is_some_and(|e| e == "gguf") {
            let (config, weights, g) = gguf::load(path, max_seq)?;
            (config, weights, Text::from_gguf(&g)?)
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
        let weights = weights.convert(&config, format, imatrix)?;
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
    /// Generation with prompt lookup decoding: after each certain token, up
    /// to `k` draft tokens are copied from the most recent earlier place
    /// where the last `ngram` tokens occurred ([`lookup_draft`]), and the
    /// certain token plus the drafts go through the model in one batched
    /// pass. Row r of its logits predicts the token after input r; each
    /// draft is checked against its row with [`Sampler::sample_draft`]
    /// (greedy: equal to the argmax; sampling: accepted with its
    /// probability, which keeps the sampler's distribution), and the first
    /// rejection's replacement is the next certain token. All weights are
    /// read once per pass, so every accepted draft is a token for (almost)
    /// free. Cache entries of rejected drafts are overwritten later.
    ///
    /// The batched pass computes the same logits by another summation
    /// order, and on quantized models that moves them by up to ~0.6
    /// (results/phase-6.md, "Prompt lookup decoding"), so a greedy run can
    /// fork from plain greedy decoding where two tokens are that close.
    pub fn generate_lookup(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, k: usize, ngram: usize, emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        self.generate_speculative(prompt, sampler, steps, |_, hist, max| lookup_draft(hist, ngram, k.min(max)).to_vec(), emit)
    }

    /// Speculative decoding with a small draft model of the same family
    /// (same vocabulary): after each certain token the draft model runs up
    /// to `k` greedy steps on its own cache, and the target checks them in
    /// one batched pass as in [`Session::generate_lookup`]. The draft's
    /// guesses only decide how much work is skipped; which tokens come out
    /// is decided by the target, so greedy text is the target's greedy text
    /// (under its batched numbers) and sampled text keeps its distribution.
    pub fn generate_draft(&mut self, draft: &mut Drafter, prompt: &str, sampler: &mut Sampler, steps: usize, k: usize, emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        assert_eq!(draft.model.config.vocab_size, self.model.config.vocab_size, "draft and target models need the same vocabulary");
        assert!(draft.model.config.seq_len >= steps.min(self.model.config.seq_len), "draft model's context is too short");
        // The draft reads the prompt up front, all but its last token (the
        // first proposal feeds that), and this counts as prefill time.
        let t0 = Instant::now();
        let tokens = self.text.encode(prompt);
        let np = tokens.len().min(steps);
        draft.fed.clear();
        if np > 1 {
            self.model.swap_pool(&mut draft.model);
            draft.model.forward_batch(&tokens[..np - 1], 0);
            self.model.swap_pool(&mut draft.model);
            draft.fed.extend_from_slice(&tokens[..np - 1]);
        }
        let ahead = t0.elapsed();
        let (mut stats, look) = self.generate_speculative(prompt, sampler, steps, |target, hist, max| draft.propose(target, hist, k.min(max)), emit);
        stats.prefill += ahead;
        (stats, look)
    }

    /// The loop shared by lookup and draft-model decoding. `propose(target,
    /// hist, max)` returns at most `max` draft tokens to follow `hist` (all
    /// certain tokens so far); the target model is passed so a draft model
    /// can borrow its thread pool.
    fn generate_speculative(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, mut propose: impl FnMut(&mut Transformer, &[usize], usize) -> Vec<usize>, mut emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        let steps = steps.min(self.model.config.seq_len);
        let mut hist = self.text.encode(prompt);
        let (mut stats, mut look) = (Stats::default(), Lookup::default());
        let np = hist.len().min(steps);
        let t0 = Instant::now();
        let mut next = sampler.sample(&mut self.model.forward_batch(&hist[..np], 0).to_vec());
        stats.prefill = t0.elapsed();
        stats.prefill_tokens = np;
        let t1 = Instant::now();
        let vocab = self.model.config.vocab_size;
        // `pos`: positions in the cache; `next` is certain but not yet in it.
        let mut pos = np;
        if np == hist.len() {
            'tokens: loop {
                if self.text.is_stop(next) || !emit(&self.text.piece(hist[pos - 1], next)) || pos >= steps {
                    break;
                }
                hist.push(next);
                let draft = propose(&mut self.model, &hist, steps - pos - 1);
                if draft.is_empty() {
                    next = sampler.sample(&mut self.model.forward(next, pos).to_vec());
                    pos += 1;
                    continue;
                }
                let input: Vec<usize> = std::iter::once(next).chain(draft.iter().copied()).collect();
                let mut rows = self.model.forward_batch_logits(&input, pos, 0).to_vec();
                look.passes += 1;
                look.drafted += draft.len();
                pos += 1;
                // Row r follows input[r]; draft[r] = input[r + 1].
                let mut rows = rows.chunks_exact_mut(vocab);
                for &d in &draft {
                    let (t, accepted) = sampler.sample_draft(rows.next().unwrap(), d);
                    if !accepted {
                        next = t;
                        continue 'tokens;
                    }
                    if self.text.is_stop(d) || !emit(&self.text.piece(hist[pos - 1], d)) {
                        break 'tokens;
                    }
                    look.accepted += 1;
                    hist.push(d);
                    pos += 1;
                }
                // Every draft accepted: the last row gives the next token.
                next = sampler.sample(rows.next().unwrap());
            }
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        (stats, look)
    }
}

/// A draft model for [`Session::generate_draft`], with the tokens its KV
/// cache holds (`fed[i]` at position i).
pub struct Drafter {
    pub model: Transformer,
    fed: Vec<usize>,
}

impl Drafter {
    /// The draft model of a loaded session (its tokenizer is not needed:
    /// draft and target share the vocabulary). It runs on the target's
    /// threads, borrowed for each proposal, so it should be opened without
    /// cores of its own: two pools spinning on the same cores would fight.
    pub fn new(session: Session) -> Drafter {
        Drafter { model: session.model, fed: Vec::new() }
    }

    /// Up to `k` greedy tokens to follow `hist`. Cache entries past the
    /// longest prefix `fed` shares with `hist` are stale (rejected drafts)
    /// and get overwritten: the missing part of `hist` is fed first (the
    /// whole prompt, the first time, in one batched pass).
    fn propose(&mut self, target: &mut Transformer, hist: &[usize], k: usize) -> Vec<usize> {
        if k == 0 {
            return Vec::new();
        }
        let keep = self.fed.iter().zip(hist).take_while(|(a, b)| a == b).count().min(hist.len() - 1);
        self.fed.truncate(keep);
        target.swap_pool(&mut self.model);
        let mut logits = self.model.forward_batch(&hist[keep..], keep).to_vec();
        self.fed.extend_from_slice(&hist[keep..]);
        let mut draft = vec![argmax(&logits)];
        while draft.len() < k {
            let d = *draft.last().unwrap();
            logits.copy_from_slice(self.model.forward(d, self.fed.len()));
            self.fed.push(d);
            draft.push(argmax(&logits));
        }
        target.swap_pool(&mut self.model);
        draft
    }
}

/// A model decoding on the GPU (`--features vulkan`, a Q4_0 or Q4_K_M
/// GGUF; see `model::gpu`), the prompt batched on the GPU or on a CPU
/// copy of the model; sampling runs on the CPU.
#[cfg(feature = "vulkan")]
pub struct GpuSession {
    // Declared before `gpu`: its buffers must be freed while the device lives.
    pub model: crate::model::gpu::GpuModel,
    pub gpu: crate::kernels::vulkan::Gpu,
    pub text: Text,
    /// With [`GpuSession::open_hybrid`]: the CPU model that runs the
    /// prompt (batched), whose KV cache is then copied to the GPU.
    pub cpu: Option<Transformer>,
}

#[cfg(feature = "vulkan")]
impl GpuSession {
    pub fn open(path: &Path, max_seq: usize) -> io::Result<GpuSession> {
        GpuSession::open_with(path, max_seq, crate::model::gpu::Options::default())
    }

    fn open_with(path: &Path, max_seq: usize, opt: crate::model::gpu::Options) -> io::Result<GpuSession> {
        let (config, weights, g) = gguf::load(path, max_seq)?;
        let text = Text::from_gguf(&g)?;
        let gpu = crate::kernels::vulkan::Gpu::new().map_err(io::Error::other)?;
        let model = crate::model::gpu::GpuModel::with(&gpu, config, weights, opt).map_err(io::Error::other)?;
        Ok(GpuSession { model, gpu, text, cpu: None })
    }

    /// [`GpuSession::open`] plus a CPU copy of the model on `cores` for the
    /// prompt, whose KV cache is then copied over (~24 KB per token for
    /// Qwen2.5-0.5B). On the phone the CPU's batched prompt pass is as fast
    /// as the GPU's f16 one for chat-length prompts (64 tokens: 777 against
    /// 691 tok/s for Qwen2.5-0.5B Q4_0; at 128, 669 against 813;
    /// results/backends). Costs a second copy of the weights in memory.
    /// The GPU's prompt pass never runs, so its f16 copy of the matrices
    /// is not built (`f32_gemm`): ~280 MB and ~2.5 s of loading less for
    /// Qwen2.5-0.5B Q4_0 on the phone.
    pub fn open_hybrid(path: &Path, max_seq: usize, cores: &[usize]) -> io::Result<GpuSession> {
        let mut s = GpuSession::open_with(path, max_seq, crate::model::gpu::Options { f32_gemm: true, ..Default::default() })?;
        let (config, weights, _) = gguf::load(path, max_seq)?;
        let mut cpu = Transformer::new(config, weights);
        cpu.set_pool(ThreadPool::new(cores)?);
        s.cpu = Some(cpu);
        Ok(s)
    }

    /// As [`Session::generate`] without echoing the prompt. The prompt runs
    /// on the CPU model if there is one (batched, then its KV cache is
    /// copied to the GPU), otherwise batched on the GPU
    /// ([`crate::model::gpu::GpuModel::forward_batch`]).
    pub fn generate(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, mut emit: impl FnMut(&[u8]) -> bool) -> Stats {
        let steps = steps.min(self.model.config.seq_len);
        let tokens = self.text.encode(prompt);
        let np = tokens.len().min(steps);
        let mut stats = Stats { prefill_tokens: np, ..Stats::default() };
        let t0 = Instant::now();
        let mut logits = self.prompt(&tokens[..np]);
        stats.prefill = t0.elapsed();
        let (t1, mut pos, mut token) = (Instant::now(), np, tokens[np - 1]);
        // Greedy decoding takes the GPU's argmax instead of reading the
        // logits (~1.1 ms per token on the phone for Qwen2.5's 151,936).
        let mut gpu_next = None;
        while np == tokens.len() {
            let next = gpu_next.unwrap_or_else(|| sampler.sample(&mut logits));
            if self.text.is_stop(next) || !emit(&self.text.piece(token, next)) || pos >= steps {
                break;
            }
            token = next;
            let l = self.model.forward(&self.gpu, token, pos).expect("GPU step");
            if sampler.temperature == 0.0 {
                gpu_next = Some(self.model.greedy());
            } else {
                logits.copy_from_slice(l);
            }
            pos += 1;
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        stats
    }

    /// Runs the prompt from position 0 on the CPU model if there is one
    /// (batched, then its KV cache is copied to the GPU), otherwise
    /// batched on the GPU; returns the last token's logits.
    fn prompt(&mut self, tokens: &[usize]) -> Vec<f32> {
        if let Some(cpu) = &mut self.cpu {
            let logits = cpu.forward_batch(tokens, 0).to_vec();
            let (k, v) = cpu.kv_cache();
            self.model.load_kv(k, v, cpu.config.seq_len, tokens.len());
            return logits;
        }
        let b = self.model.batch;
        let mut logits = Vec::new();
        for (i, part) in tokens.chunks(b).enumerate() {
            logits = self.model.forward_batch(&self.gpu, part, i * b).expect("GPU prompt pass").to_vec();
        }
        logits
    }

    /// Prompt lookup decoding as [`Session::generate_lookup`]: up to `k`
    /// draft tokens (at most [`crate::model::gpu::GpuModel::few`] − 1)
    /// copied from the last earlier match of the last `ngram` tokens,
    /// checked in one pass of the decode shader for several tokens
    /// ([`crate::model::gpu::GpuModel::forward_few`]). Greedy text is the
    /// GPU's greedy text under those products' numbers; sampled text keeps
    /// the sampler's distribution.
    pub fn generate_lookup(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, k: usize, ngram: usize, mut emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        let steps = steps.min(self.model.config.seq_len);
        let k = k.min(self.model.few.saturating_sub(1));
        let mut hist = self.text.encode(prompt);
        let (mut stats, mut look) = (Stats::default(), Lookup::default());
        let np = hist.len().min(steps);
        let t0 = Instant::now();
        let mut next = sampler.sample(&mut self.prompt(&hist[..np]));
        stats.prefill = t0.elapsed();
        stats.prefill_tokens = np;
        let t1 = Instant::now();
        let vocab = self.model.config.vocab_size;
        // `pos`: positions in the cache; `next` is certain but not yet in it.
        let mut pos = np;
        if np == hist.len() {
            'tokens: loop {
                if self.text.is_stop(next) || !emit(&self.text.piece(hist[pos - 1], next)) || pos >= steps {
                    break;
                }
                hist.push(next);
                let draft = lookup_draft(&hist, ngram, k.min(steps - pos - 1)).to_vec();
                if draft.is_empty() {
                    next = sampler.sample(&mut self.model.forward(&self.gpu, next, pos).expect("GPU step").to_vec());
                    pos += 1;
                    continue;
                }
                let input: Vec<usize> = std::iter::once(next).chain(draft.iter().copied()).collect();
                let mut rows = self.model.forward_few(&self.gpu, &input, pos).expect("GPU check").to_vec();
                look.passes += 1;
                look.drafted += draft.len();
                pos += 1;
                // Row r follows input[r]; draft[r] = input[r + 1].
                let mut rows = rows.chunks_exact_mut(vocab);
                for &d in &draft {
                    let (t, accepted) = sampler.sample_draft(rows.next().unwrap(), d);
                    if !accepted {
                        next = t;
                        continue 'tokens;
                    }
                    if self.text.is_stop(d) || !emit(&self.text.piece(hist[pos - 1], d)) {
                        break 'tokens;
                    }
                    look.accepted += 1;
                    hist.push(d);
                    pos += 1;
                }
                // Every draft accepted: the last row gives the next token.
                next = sampler.sample(rows.next().unwrap());
            }
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        (stats, look)
    }
}

/// A model on the Hexagon DSP (`--features hexagon`, a `--pure` Q4_0
/// GGUF; see `model::dsp`): the prompt by prefills of up to
/// [`DspSession::BATCH`] tokens (products on HMX), then one decode step
/// per token; sampling on the CPU.
#[cfg(feature = "hexagon")]
pub struct DspSession {
    // Declared before `dsp`: its shared buffers must be freed while the
    // DSP session lives.
    pub model: crate::model::dsp::DspModel,
    pub dsp: crate::kernels::hexagon::Dsp,
    pub text: Text,
}

#[cfg(feature = "hexagon")]
impl DspSession {
    /// Prompt tokens per prefill call (what Qwen2.5-0.5B's activations
    /// leave room for in VTCM is ~320).
    pub const BATCH: usize = 128;

    /// `dir`: a writable directory the DSP library is put in and loaded
    /// from (see [`crate::kernels::hexagon::Dsp::open`]); `threads`: DSP
    /// threads (6 is the fastest measured).
    pub fn open(path: &Path, max_seq: usize, dir: &Path, threads: u32) -> io::Result<DspSession> {
        let (config, weights, g) = gguf::load(path, max_seq)?;
        let text = Text::from_gguf(&g)?;
        let dsp = crate::kernels::hexagon::Dsp::open(dir).map_err(io::Error::other)?;
        let batch = Self::BATCH.min(config.seq_len);
        let model = crate::model::dsp::DspModel::new(&dsp, config, weights, threads, batch, false).map_err(io::Error::other)?;
        Ok(DspSession { model, dsp, text })
    }

    /// As [`Session::generate`] without echoing the prompt.
    pub fn generate(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, mut emit: impl FnMut(&[u8]) -> bool) -> Stats {
        let steps = steps.min(self.model.config.seq_len);
        let tokens = self.text.encode(prompt);
        let np = tokens.len().min(steps);
        let mut stats = Stats { prefill_tokens: np, ..Stats::default() };
        let t0 = Instant::now();
        let mut logits = self.prompt(&tokens[..np]);
        stats.prefill = t0.elapsed();
        let (t1, mut pos, mut token) = (Instant::now(), np, tokens[np - 1]);
        // Greedy decoding takes the DSP's argmax instead of reading the logits.
        let mut dsp_next = (sampler.temperature == 0.0).then(|| self.model.greedy());
        while np == tokens.len() {
            let next = dsp_next.unwrap_or_else(|| sampler.sample(&mut logits));
            if self.text.is_stop(next) || !emit(&self.text.piece(token, next)) || pos >= steps {
                break;
            }
            token = next;
            let l = self.model.forward(&self.dsp, token, pos).expect("DSP step");
            if sampler.temperature == 0.0 {
                dsp_next = Some(self.model.greedy());
            } else {
                logits.copy_from_slice(l);
            }
            pos += 1;
        }
        stats.decode = t1.elapsed();
        stats.decode_tokens = pos - np;
        stats
    }

    /// Runs the prompt from position 0 in prefills of up to
    /// [`DspSession::BATCH`] tokens; returns the last token's logits.
    fn prompt(&mut self, tokens: &[usize]) -> Vec<f32> {
        let b = self.model.batch;
        let mut logits = Vec::new();
        for (i, part) in tokens.chunks(b).enumerate() {
            logits = self.model.prefill(&self.dsp, part, i * b).expect("DSP prefill").to_vec();
        }
        logits
    }

    /// Prompt lookup decoding as [`Session::generate_lookup`]: up to `k`
    /// draft tokens (at most [`crate::kernels::hexagon::MAX_FEW`] − 1)
    /// checked in one few-token pass ([`crate::model::dsp::DspModel::few`]),
    /// whose products are those of the decode step. Greedy text is the
    /// DSP's greedy text up to near-ties (the products add in another
    /// order); sampled text keeps the sampler's distribution.
    pub fn generate_lookup(&mut self, prompt: &str, sampler: &mut Sampler, steps: usize, k: usize, ngram: usize, mut emit: impl FnMut(&[u8]) -> bool) -> (Stats, Lookup) {
        let steps = steps.min(self.model.config.seq_len);
        let k = k.min(crate::kernels::hexagon::MAX_FEW - 1);
        let mut hist = self.text.encode(prompt);
        let (mut stats, mut look) = (Stats::default(), Lookup::default());
        let np = hist.len().min(steps);
        let t0 = Instant::now();
        let mut next = sampler.sample(&mut self.prompt(&hist[..np]));
        stats.prefill = t0.elapsed();
        stats.prefill_tokens = np;
        let t1 = Instant::now();
        let vocab = self.model.config.vocab_size;
        // `pos`: positions in the cache; `next` is certain but not yet in it.
        let mut pos = np;
        if np == hist.len() {
            'tokens: loop {
                if self.text.is_stop(next) || !emit(&self.text.piece(hist[pos - 1], next)) || pos >= steps {
                    break;
                }
                hist.push(next);
                let draft = lookup_draft(&hist, ngram, k.min(steps - pos - 1)).to_vec();
                if draft.is_empty() {
                    next = sampler.sample(&mut self.model.forward(&self.dsp, next, pos).expect("DSP step").to_vec());
                    pos += 1;
                    continue;
                }
                let input: Vec<usize> = std::iter::once(next).chain(draft.iter().copied()).collect();
                let mut rows = self.model.few(&self.dsp, &input, pos).expect("DSP check").to_vec();
                look.passes += 1;
                look.drafted += draft.len();
                pos += 1;
                // Row r follows input[r]; draft[r] = input[r + 1].
                let mut rows = rows.chunks_exact_mut(vocab);
                for &d in &draft {
                    let (t, accepted) = sampler.sample_draft(rows.next().unwrap(), d);
                    if !accepted {
                        next = t;
                        continue 'tokens;
                    }
                    if self.text.is_stop(d) || !emit(&self.text.piece(hist[pos - 1], d)) {
                        break 'tokens;
                    }
                    look.accepted += 1;
                    hist.push(d);
                    pos += 1;
                }
                // Every draft accepted: the last row gives the next token.
                next = sampler.sample(rows.next().unwrap());
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
        let (stats, look) = s.generate_lookup("Once upon a time", &mut Sampler::new(0.0, 0.9, 0, 1), 64, 4, 1, |b| {
            out.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8(out).unwrap() + "\n", std::fs::read_to_string(want).unwrap());
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 64);
        assert!(look.drafted > 0 && look.accepted > 0, "{look:?}");
    }

    /// Draft-model decoding gives llama2.c's greedy text: the target is
    /// stories15M in f32 and the draft the same model in Q4_0, which
    /// disagrees now and then, so stale draft cache entries are exercised.
    #[test]
    fn draft_model_decoding_gives_llama2c_greedy_text() {
        let (Some(model), Some(want)) = (golden("models/stories15M/stories15M.bin"), golden("stories15M/greedy.txt")) else { return };
        let mut s = Session::open(&model, None, 256, Format::F32, None).unwrap();
        let mut draft = Drafter::new(Session::open(&model, None, 256, Format::Q4_0, None).unwrap());
        let prompt = s.text.encode("Once upon a time");
        let mut out = s.text.first_piece(prompt[0]);
        for p in 0..prompt.len() - 1 {
            out.extend(s.text.piece(prompt[p], prompt[p + 1]));
        }
        let (stats, look) = s.generate_draft(&mut draft, "Once upon a time", &mut Sampler::new(0.0, 0.9, 0, 1), 64, 4, |b| {
            out.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8(out).unwrap() + "\n", std::fs::read_to_string(want).unwrap());
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 64);
        assert!(look.accepted > 0 && look.accepted < look.drafted, "{look:?}");
    }

    /// The GPU's argmax (greedy decoding) is the index the CPU's argmax
    /// finds in the same logits, after a prompt batch and after each step.
    #[cfg(feature = "vulkan")]
    #[test]
    fn gpu_argmax_matches_the_logits() {
        let Some(model) = golden("gguf/smollm2-135m-q4_0.gguf") else { return };
        let Ok(mut s) = GpuSession::open(&model, 64) else {
            assert!(!cfg!(target_os = "android"), "no GPU");
            return;
        };
        let tokens = s.text.encode("The capital of France is");
        let mut next = argmax(s.model.forward_batch(&s.gpu, &tokens, 0).unwrap());
        assert_eq!(s.model.greedy(), next);
        for pos in tokens.len()..tokens.len() + 16 {
            next = argmax(s.model.forward(&s.gpu, next, pos).unwrap());
            assert_eq!(s.model.greedy(), next, "at {pos}");
        }
    }

    /// Prompt lookup on the GPU leaves greedy text as it is: the same text
    /// as plain GPU decoding of the same file, with drafts proposed (the
    /// reply repeats itself) and some accepted, checked by the few-token
    /// pass. A different text would mean a wrong KV position or row.
    #[cfg(feature = "vulkan")]
    #[test]
    fn gpu_lookup_keeps_the_greedy_text() {
        let Some(model) = golden("gguf/smollm2-135m-q4_0.gguf") else { return };
        let Ok(mut s) = GpuSession::open(&model, 96) else {
            assert!(!cfg!(target_os = "android"), "no GPU");
            return;
        };
        let prompt = "The capital of France is";
        let mut plain = Vec::new();
        s.generate(prompt, &mut Sampler::new(0.0, 0.9, 0, 1), 96, |b| {
            plain.extend_from_slice(b);
            true
        });
        let mut looked = Vec::new();
        let (stats, look) = s.generate_lookup(prompt, &mut Sampler::new(0.0, 0.9, 0, 1), 96, 4, 2, |b| {
            looked.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8_lossy(&looked), String::from_utf8_lossy(&plain));
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 96);
        assert!(look.drafted > 0 && look.accepted > 0, "{look:?}");
    }

    /// Prompt lookup on the DSP leaves greedy text as it is: the few-token
    /// pass's logits are the decode step's (same products, same order; see
    /// `model::dsp`'s test), so the text equals plain DSP decoding's, with
    /// drafts proposed and some accepted.
    #[cfg(feature = "hexagon")]
    #[test]
    fn dsp_lookup_keeps_the_greedy_text() {
        let Some(model) = golden("gguf/smollm2-135m-q4_0.gguf") else { return };
        let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        let Ok(mut s) = DspSession::open(&model, 96, &dir, 6) else {
            assert!(!cfg!(target_os = "android"), "no DSP");
            return;
        };
        let prompt = "The capital of France is";
        let mut plain = Vec::new();
        s.generate(prompt, &mut Sampler::new(0.0, 0.9, 0, 1), 96, |b| {
            plain.extend_from_slice(b);
            true
        });
        let mut looked = Vec::new();
        let (stats, look) = s.generate_lookup(prompt, &mut Sampler::new(0.0, 0.9, 0, 1), 96, 4, 2, |b| {
            looked.extend_from_slice(b);
            true
        });
        assert_eq!(String::from_utf8_lossy(&looked), String::from_utf8_lossy(&plain));
        assert_eq!(stats.prefill_tokens + stats.decode_tokens, 96);
        assert!(look.drafted > 0 && look.accepted > 0, "{look:?}");
    }
}
