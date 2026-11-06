//! SentencePiece-style BPE: Gemma 4's tokenizer (`tokenizer.ggml.model =
//! gemma4` in GGUF, a `BPE` model with `byte_fallback` in tokenizer.json).
//!
//! Unlike GPT-2's byte-level BPE ([`super::bytelevel`]) the vocabulary holds
//! real UTF-8 text, with `▁` (U+2581) standing for a space:
//!
//! 1. special tokens (`<bos>`, `<|turn>`, …) are cut out first, verbatim;
//! 2. every space becomes `▁` (HF's normalizer; its pre-tokenizer then splits
//!    on spaces, of which none are left, so the whole text is one word);
//! 3. each character is a symbol; a character with no token becomes its
//!    UTF-8 bytes as `<0xXX>` tokens (byte fallback);
//! 4. adjacent symbols are merged, lowest merge rank first (leftmost on
//!    ties), until no pair has a merge.
//!
//! Decoding reverses it: `▁` → space, `<0xXX>` runs → bytes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub struct Tokenizer {
    id_to_token: Vec<String>,
    vocab: HashMap<String, usize>,
    /// (left id, right id) → (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Special tokens matched verbatim before anything else, longest first.
    added: Vec<(String, usize)>,
    is_added: Vec<bool>,
    /// `byte_token[b]` is the id of `<0xBB>`.
    byte_token: [Option<usize>; 256],
    /// The byte a `<0xXX>` token stands for.
    token_byte: HashMap<usize, u8>,
}

impl Tokenizer {
    /// The tokenizer embedded in a GGUF file: tokens, their types (3
    /// control and 4 user-defined are special, 6 byte) and the merges.
    pub fn from_gguf(g: &crate::format::gguf::Gguf) -> io::Result<Tokenizer> {
        use crate::format::gguf::Value;
        let arr = |k: &str| g.get(k).and_then(Value::as_array).ok_or_else(|| bad(format!("GGUF has no {k}")));
        if g.get("tokenizer.ggml.model").and_then(Value::as_str) != Some("gemma4") {
            return Err(bad("not a gemma4 tokenizer".into()));
        }
        let id_to_token: Vec<String> = arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let types: Vec<u64> = arr("tokenizer.ggml.token_type")?.iter().map(|v| v.as_u64().unwrap_or(1)).collect();
        if types.len() != id_to_token.len() {
            return Err(bad("token types and tokens differ in length".into()));
        }
        let vocab: HashMap<String, usize> = id_to_token.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        let mut added: Vec<(String, usize)> = types.iter().enumerate().filter(|&(_, &t)| t == 3 || t == 4).map(|(i, _)| (id_to_token[i].clone(), i)).collect();
        added.sort_by_key(|(s, _)| Reverse(s.len()));
        let mut is_added = vec![false; id_to_token.len()];
        for &(_, id) in &added {
            is_added[id] = true;
        }
        let mut byte_token = [None; 256];
        let mut token_byte = HashMap::new();
        for (id, t) in id_to_token.iter().enumerate().filter(|&(i, _)| types[i] == 6) {
            let b = t.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()).ok_or_else(|| bad(format!("byte token {t:?}")))?;
            byte_token[b as usize] = Some(id);
            token_byte.insert(id, b);
        }
        let id = |s: &str| vocab.get(s).map(|&i| i as u32).ok_or_else(|| bad(format!("merge uses unknown token {s:?}")));
        let mut merges = HashMap::new();
        for (rank, v) in arr("tokenizer.ggml.merges")?.iter().enumerate() {
            let (a, b) = v.as_str().and_then(|s| s.split_once(' ')).ok_or_else(|| bad("bad merge".into()))?;
            merges.insert((id(a)?, id(b)?), (rank as u32, id(&format!("{a}{b}"))?));
        }
        Ok(Tokenizer { id_to_token, vocab, merges, added, is_added, byte_token, token_byte })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// The id of a token string (e.g. a special token), if it exists.
    pub fn token_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    /// Token ids for `text` (no BOS: the caller adds it).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            // Earliest special token in `rest` (the longest one at that position).
            let hit = self.added.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id))).min_by_key(|&(p, len, _)| (p, Reverse(len)));
            match hit {
                Some((p, len, id)) => {
                    self.bpe(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.bpe(rest, &mut out);
                    return out;
                }
            }
        }
    }

    /// Steps 2–4 for text without special tokens.
    fn bpe(&self, text: &str, out: &mut Vec<usize>) {
        if text.is_empty() {
            return;
        }
        let text = text.replace(' ', "\u{2581}");
        let mut syms: Vec<u32> = Vec::with_capacity(text.len());
        let mut buf = [0u8; 4];
        for c in text.chars() {
            match self.vocab.get(c.encode_utf8(&mut buf) as &str) {
                Some(&id) => syms.push(id as u32),
                None => syms.extend(c.encode_utf8(&mut buf).bytes().filter_map(|b| self.byte_token[b as usize]).map(|id| id as u32)),
            }
        }
        // A linked list of symbols and a heap of candidate merges by (rank,
        // position); entries made stale by an earlier merge are skipped.
        let n = syms.len();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<(u32, usize, u32, u32)>>, syms: &[u32], i: usize, j: usize| {
            if let Some(&(rank, _)) = self.merges.get(&(syms[i], syms[j])) {
                heap.push(Reverse((rank, i, syms[i], syms[j])));
            }
        };
        for i in 0..n.saturating_sub(1) {
            push(&mut heap, &syms, i, i + 1);
        }
        while let Some(Reverse((_, i, a, b))) = heap.pop() {
            let j = next[i];
            if !alive[i] || j >= n || syms[i] != a || syms[j] != b {
                continue;
            }
            syms[i] = self.merges[&(a, b)].1;
            alive[j] = false;
            next[i] = next[j];
            if next[j] < n {
                prev[next[j]] = i;
            }
            if prev[i] < n {
                push(&mut heap, &syms, prev[i], i);
            }
            if next[i] < n {
                push(&mut heap, &syms, i, next[i]);
            }
        }
        out.extend((0..n).filter(|&i| alive[i]).map(|i| syms[i] as usize));
    }

    /// The raw bytes of one token (for streaming output: a multi-byte
    /// character may be split across byte tokens, so bytes, not text).
    pub fn token_bytes(&self, id: usize) -> Vec<u8> {
        if let Some(&b) = self.token_byte.get(&id) {
            return vec![b];
        }
        let t = &self.id_to_token[id];
        if self.is_added[id] { t.as_bytes().to_vec() } else { t.replace('\u{2581}', " ").into_bytes() }
    }

    /// Text for `ids` (invalid UTF-8 from byte tokens becomes U+FFFD).
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;

    /// 440 strings (the shared corpus, Gemma's chat markup, newline runs,
    /// byte fallback) must encode to exactly the ids HF `tokenizers` gives
    /// for `google/gemma-4-E2B-it`, and decode to exactly its text.
    #[test]
    fn gemma4_matches_hf_tokenizers_on_corpus() {
        let (Some(m), Some(c), Some(e), Some(d)) = (
            golden("gguf/gemma-4-E2B-it-Q4_0.gguf"),
            golden("gemma4/corpus.txt"),
            golden("gemma4/encoded.txt"),
            golden("gemma4/decoded.txt"),
        ) else {
            return;
        };
        let t = Tokenizer::from_gguf(&crate::format::gguf::Gguf::open(m).unwrap()).unwrap();
        let (corpus, enc, dec) = (std::fs::read_to_string(c).unwrap(), std::fs::read_to_string(e).unwrap(), std::fs::read_to_string(d).unwrap());
        let (strings, lines, decoded): (Vec<&str>, Vec<&str>, Vec<&str>) = (corpus.split('\x1e').collect(), enc.lines().collect(), dec.split('\x1e').collect());
        assert_eq!((strings.len(), lines.len()), (decoded.len(), decoded.len()));
        let mut bad = 0;
        for ((s, line), want_text) in strings.iter().zip(&lines).zip(&decoded) {
            let want: Vec<usize> = line.split(' ').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
            let got = t.encode(s);
            if got != want {
                bad += 1;
                eprintln!("MISMATCH {s:?}\n  ours {got:?}\n  hf   {want:?}");
            }
            assert_eq!(t.decode(&want), *want_text, "decoding {want:?}");
        }
        assert_eq!(bad, 0, "{bad} of {} strings encode differently from HF", strings.len());
        println!("gemma4: {} strings, encode and decode identical to HF tokenizers", strings.len());
    }
}
