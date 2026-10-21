//! llama2.c's tokenizer (`tokenizer.bin`): the Llama 2 SentencePiece
//! vocabulary, encoded with score-ordered BPE merges.
//!
//! File format: `i32 max_token_length`, then for each of the 32000 tokens:
//! `f32 score, i32 len, len bytes`. Token 0 is `<unk>`, 1 is `<s>` (BOS),
//! 2 is `</s>` (EOS), 3..=258 are the byte tokens `<0x00>`..`<0xFF>`.
//!
//! Encoding a string:
//!
//! 1. BOS if asked; then, unless the text is empty, the token for `" "`
//!    (SentencePiece's "dummy prefix": every word, even the first, is
//!    preceded by a space).
//! 2. Split the UTF-8 bytes into codepoints; each codepoint that is a vocab
//!    entry becomes that token, any other becomes one byte token per byte
//!    ("byte fallback", so no text is ever unrepresentable).
//! 3. Repeatedly merge the adjacent pair whose concatenation is in the vocab
//!    with the highest score (leftmost on ties), until no pair merges.
//!
//! Merging by score rather than by a merge list is what makes this
//! "SentencePiece BPE"; GPT-2-style byte-level BPE (Phase 5) ranks merges
//! from an explicit list instead.

use std::collections::HashMap;
use std::io;
use std::path::Path;

pub const BOS: usize = 1;
pub const EOS: usize = 2;

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    lookup: HashMap<Vec<u8>, usize>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>, vocab_size: usize) -> io::Result<Tokenizer> {
        let bytes = std::fs::read(path)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "tokenizer.bin ends early");
        let mut at = 4; // skip max_token_length
        let mut take = |n: usize| -> io::Result<&[u8]> {
            let s = bytes.get(at..at + n).ok_or_else(bad)?;
            at += n;
            Ok(s)
        };
        let (mut vocab, mut scores) = (Vec::with_capacity(vocab_size), Vec::with_capacity(vocab_size));
        for _ in 0..vocab_size {
            scores.push(f32::from_le_bytes(take(4)?.try_into().unwrap()));
            let len = i32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            vocab.push(take(len)?.to_vec());
        }
        // Several ids could in principle share bytes; like llama2.c's sorted
        // lookup we keep one id per string (the first).
        let mut lookup = HashMap::with_capacity(vocab_size);
        for (i, v) in vocab.iter().enumerate() {
            lookup.entry(v.clone()).or_insert(i);
        }
        Ok(Tokenizer { vocab, scores, lookup })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() + 3);
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            tokens.push(self.lookup[&b" "[..]]);
        }

        // Step 2: codepoints, with byte fallback. Like run.c this works on
        // raw bytes: a codepoint is a lead byte plus up to 3 continuation
        // bytes (a 4th continuation byte would start a new group).
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let mut end = i + 1;
            while end < b.len() && b[end] & 0xC0 == 0x80 && end - i < 4 {
                end += 1;
            }
            match self.lookup.get(&b[i..end]) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(b[i..end].iter().map(|&byte| byte as usize + 3)),
            }
            i = end;
        }

        // Step 3: greedy merges by score.
        let mut buf = Vec::new();
        loop {
            let mut best: Option<(f32, usize, usize)> = None; // (score, position, merged id)
            for p in 0..tokens.len().saturating_sub(1) {
                buf.clear();
                buf.extend_from_slice(&self.vocab[tokens[p]]);
                buf.extend_from_slice(&self.vocab[tokens[p + 1]]);
                if let Some(&id) = self.lookup.get(&buf)
                    && best.is_none_or(|(s, _, _)| self.scores[id] > s)
                {
                    best = Some((self.scores[id], p, id));
                }
            }
            let Some((_, p, id)) = best else { break };
            tokens[p] = id;
            tokens.remove(p + 1);
        }

        if eos {
            tokens.push(EOS);
        }
        tokens
    }

    /// The bytes token `token` stands for when it follows `prev`: the
    /// leading space of the first word after BOS is dropped (it was only
    /// SentencePiece's dummy prefix), and `<0xNN>` tokens become byte NN.
    pub fn decode(&self, prev: usize, token: usize) -> Vec<u8> {
        let mut piece: &[u8] = &self.vocab[token];
        if prev == BOS && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        if let Some(hex) = piece.strip_prefix(b"<0x").and_then(|r| r.strip_suffix(b">"))
            && let Some(byte) = std::str::from_utf8(hex).ok().and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            return vec![byte];
        }
        piece.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;

    fn tokenizer() -> Option<Tokenizer> {
        Some(Tokenizer::load(golden("models/stories15M/tokenizer.bin")?, 32000).unwrap())
    }

    /// Every corpus string, encoded with BOS, must give exactly the ids
    /// llama2.c's own `encode` gives (tools/llama2c_encode.c).
    #[test]
    fn encode_matches_llama2c_on_corpus() {
        let (Some(t), Some(corpus), Some(expected)) =
            (tokenizer(), golden("stories15M/corpus.txt"), golden("stories15M/encoded.txt"))
        else {
            return;
        };
        let corpus = std::fs::read_to_string(corpus).unwrap();
        let expected = std::fs::read_to_string(expected).unwrap();
        let (strings, lines): (Vec<&str>, Vec<&str>) = (corpus.split('\x1e').collect(), expected.lines().collect());
        assert_eq!(strings.len(), lines.len());
        for (s, line) in strings.iter().zip(&lines) {
            let want: Vec<usize> = line.split(' ').map(|x| x.parse().unwrap()).collect();
            assert_eq!(t.encode(s, true, false), want, "encoding {s:?}");
        }
        println!("{} strings encode identically to llama2.c", strings.len());
    }

    /// Decoding what was encoded gives the text back, byte for byte
    /// (byte-fallback tokens included).
    #[test]
    fn decode_inverts_encode() {
        let (Some(t), Some(corpus)) = (tokenizer(), golden("stories15M/corpus.txt")) else { return };
        for s in std::fs::read_to_string(corpus).unwrap().split('\x1e') {
            let ids = t.encode(s, true, false);
            let bytes: Vec<u8> = ids.windows(2).flat_map(|w| t.decode(w[0], w[1])).collect();
            assert_eq!(String::from_utf8_lossy(&bytes), *s);
        }
    }
}
