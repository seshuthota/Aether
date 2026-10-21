//! GPT-2-style byte-level BPE, loaded from a Hugging Face `tokenizer.json`
//! (SmolLM2; Qwen2 adds its own pre-tokenizer in Phase 7).
//!
//! Encoding a string takes four steps:
//!
//! 1. **Special tokens** (`<|im_start|>`, `<|endoftext|>`, …) are cut out
//!    first and become their ids directly; only the text between them is
//!    tokenized.
//! 2. **Pre-tokenization** splits the text into "words" that BPE never
//!    merges across. SmolLM2 declares two pre-tokenizers in sequence:
//!    `Digits` (every digit on its own) and `ByteLevel`'s GPT-2 regex
//!    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!    — contractions, an optional space glued to the following letters /
//!    numbers / symbols, and whitespace runs. [`gpt2_split`] implements that
//!    regex by hand: at each position the first alternative that matches wins.
//! 3. **Byte-level mapping**: each word's UTF-8 bytes become 256 printable
//!    stand-in characters (space → `Ġ`, newline → `Ċ`, …), so every possible
//!    byte has a token and nothing is ever "unknown".
//! 4. **BPE**: start from one symbol per byte and repeatedly merge the
//!    adjacent pair with the best (lowest) rank in the merges list, leftmost
//!    first on ties, until no listed pair remains.
//!
//! Decoding concatenates the tokens' stand-in characters and maps them back
//! to bytes.

use super::unicode_tables::LETTER;
use crate::format::json::Json;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// `\p{L}`: binary search in the generated letter ranges.
pub fn is_letter(c: char) -> bool {
    let c = c as u32;
    LETTER
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{N}` (general categories Nd, Nl, No).
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// `[^\s\p{L}\p{N}]`
fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// The GPT-2 pre-tokenizer regex, as byte ranges into `s`.
pub fn gpt2_split(s: &str) -> Vec<(usize, usize)> {
    let cs: Vec<(usize, char)> = s.char_indices().collect();
    let end = s.len();
    let at = |i: usize| cs.get(i).map(|&(b, _)| b).unwrap_or(end);
    let ch = |i: usize| cs.get(i).map(|&(_, c)| c);
    let run = |mut i: usize, f: &dyn Fn(char) -> bool| {
        while ch(i).is_some_and(f) {
            i += 1;
        }
        i
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i].1;
        // 1. contractions: 's 't 'm 'd 're 've 'll (lowercase only)
        if c == '\'' {
            let (a, b) = (ch(i + 1), ch(i + 2));
            let len = match (a, b) {
                (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
                (Some('s' | 't' | 'm' | 'd'), _) => 2,
                _ => 0,
            };
            if len > 0 {
                out.push((at(i), at(i + len)));
                i += len;
                continue;
            }
        }
        // 2-4. ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`
        let start = if c == ' ' { i + 1 } else { i };
        let mut matched = false;
        for class in [is_letter as fn(char) -> bool, is_number, is_other] {
            if ch(start).is_some_and(class) {
                let j = run(start, &class);
                out.push((at(i), at(j)));
                i = j;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        // 5. `\s+(?!\S)`: a whitespace run, minus its last character when a
        //    non-space follows (that character then prefixes the next word).
        // 6. `\s+`: otherwise the whole run.
        if c.is_whitespace() {
            let j = run(i, &|c: char| c.is_whitespace());
            let j = if j < cs.len() && j - i > 1 { j - 1 } else { j };
            out.push((at(i), at(j)));
            i = j;
            continue;
        }
        unreachable!("every character is a letter, number, other or whitespace");
    }
    out
}

/// GPT-2's byte → stand-in character table: printable Latin-1 bytes map to
/// themselves, the other 68 (controls, space, …) to U+0100 onwards.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut extra = 0;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    table
}

pub struct Tokenizer {
    id_to_token: Vec<String>,
    vocab: HashMap<String, usize>,
    /// (left id, right id) → (rank, merged id)
    merges: HashMap<(usize, usize), (usize, usize)>,
    /// Added tokens matched verbatim before pre-tokenization, longest first.
    added: Vec<(String, usize)>,
    split_digits: bool,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Tokenizer> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let j = Json::parse(&text).map_err(|e| bad(format!("tokenizer.json: {e}")))?;
        let model = j.get("model").ok_or_else(|| bad("no model".into()))?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return Err(bad("only BPE tokenizers are supported".into()));
        }
        let vocab_obj = model.get("vocab").and_then(Json::as_object).ok_or_else(|| bad("no vocab".into()))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut id_to_token = vec![String::new(); vocab_obj.len()];
        for (tok, id) in vocab_obj {
            let id = id.as_usize().ok_or_else(|| bad(format!("bad id for {tok:?}")))?;
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = tok.clone();
            vocab.insert(tok.clone(), id);
        }
        let mut added = Vec::new();
        for a in j.get("added_tokens").and_then(Json::as_array).unwrap_or(&[]) {
            let (Some(content), Some(id)) = (a.get("content").and_then(Json::as_str), a.get("id").and_then(Json::as_usize)) else {
                continue;
            };
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = content.to_string();
            added.push((content.to_string(), id));
        }
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        // Merges are "a b" strings (older files) or ["a", "b"] pairs (newer).
        let mut merges = HashMap::new();
        for (rank, m) in model.get("merges").and_then(Json::as_array).unwrap_or(&[]).iter().enumerate() {
            let (a, b) = match m {
                Json::Str(s) => s.split_once(' ').ok_or_else(|| bad(format!("bad merge {s:?}")))?,
                Json::Arr(p) if p.len() == 2 => (p[0].as_str().unwrap_or(""), p[1].as_str().unwrap_or("")),
                _ => return Err(bad("bad merge entry".into())),
            };
            let id = |s: &str| vocab.get(s).copied().ok_or_else(|| bad(format!("merge uses unknown token {s:?}")));
            merges.insert((id(a)?, id(b)?), (rank, id(&format!("{a}{b}"))?));
        }

        // Pre-tokenizer: Digits(individual) + ByteLevel(regex) is what is
        // supported; anything else is refused rather than silently ignored.
        let pre = j.get("pre_tokenizer").ok_or_else(|| bad("no pre_tokenizer".into()))?;
        let steps: Vec<&Json> = match pre.get("type").and_then(Json::as_str) {
            Some("Sequence") => pre.get("pretokenizers").and_then(Json::as_array).unwrap_or(&[]).iter().collect(),
            _ => vec![pre],
        };
        let mut split_digits = false;
        for step in steps {
            match step.get("type").and_then(Json::as_str) {
                Some("Digits") if step.get("individual_digits").and_then(Json::as_bool) == Some(true) => split_digits = true,
                Some("ByteLevel")
                    if step.get("use_regex").and_then(Json::as_bool) != Some(false)
                        && step.get("add_prefix_space").and_then(Json::as_bool) != Some(true) => {}
                other => return Err(bad(format!("unsupported pre-tokenizer step {other:?}"))),
            }
        }

        let byte_char = bytes_to_unicode();
        let char_byte = byte_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Ok(Tokenizer { id_to_token, vocab, merges, added, split_digits, byte_char, char_byte })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// The id of a token string (e.g. a special token), if it exists.
    pub fn token_id(&self, token: &str) -> Option<usize> {
        self.added.iter().find(|(s, _)| s == token).map(|&(_, id)| id).or_else(|| self.vocab.get(token).copied())
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            // Earliest added token in `rest` (the longest one at that position).
            let hit = self
                .added
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id)))
                .min_by_key(|&(p, len, _)| (p, std::cmp::Reverse(len)));
            match hit {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    return out;
                }
            }
        }
    }

    /// The raw bytes of one token (for streaming output: a multi-byte
    /// character may be split across tokens, so bytes, not text).
    pub fn token_bytes(&self, id: usize) -> Vec<u8> {
        if self.added.iter().any(|&(_, a)| a == id) {
            return self.id_to_token[id].as_bytes().to_vec();
        }
        self.id_to_token[id].chars().map(|c| self.char_byte.get(&c).copied().unwrap_or(b'?')).collect()
    }

    /// Steps 2–4 for text without special tokens.
    fn encode_plain(&self, text: &str, out: &mut Vec<usize>) {
        // Digits pre-tokenizer: each numeric character becomes its own piece.
        let mut pieces = Vec::new();
        if self.split_digits {
            let mut start = 0;
            for (b, c) in text.char_indices() {
                if c.is_numeric() {
                    pieces.push(&text[start..b]);
                    pieces.push(&text[b..b + c.len_utf8()]);
                    start = b + c.len_utf8();
                }
            }
            pieces.push(&text[start..]);
        } else {
            pieces.push(text);
        }
        for piece in pieces.into_iter().filter(|p| !p.is_empty()) {
            for (a, b) in gpt2_split(piece) {
                self.bpe(&piece[a..b], out);
            }
        }
    }

    /// Byte-level mapping and merges for one pre-tokenized word.
    fn bpe(&self, word: &str, out: &mut Vec<usize>) {
        let mut syms: Vec<usize> = word
            .bytes()
            .map(|b| {
                let mut buf = [0u8; 4];
                self.vocab[self.byte_char[b as usize].encode_utf8(&mut buf) as &str]
            })
            .collect();
        loop {
            // The best-ranked adjacent pair, leftmost on ties.
            let best = syms.windows(2).enumerate().filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id))).min();
            let Some((_, i, id)) = best else { break };
            syms[i] = id;
            syms.remove(i + 1);
        }
        out.extend(syms);
    }

    /// Text for `ids`: stand-in characters mapped back to bytes, then UTF-8
    /// (invalid sequences, e.g. half an emoji, become U+FFFD).
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = &self.id_to_token[id];
            if self.added.iter().any(|&(_, a)| a == id) {
                bytes.extend_from_slice(tok.as_bytes());
                continue;
            }
            for c in tok.chars() {
                match self.char_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;

    fn words(s: &str) -> Vec<&str> {
        gpt2_split(s).into_iter().map(|(a, b)| &s[a..b]).collect()
    }

    /// Hand-checked splits of the GPT-2 regex.
    #[test]
    fn gpt2_regex_cases() {
        assert_eq!(words("Hello world"), ["Hello", " world"]);
        assert_eq!(words("it's  ok!!"), ["it", "'s", " ", " ok", "!!"]);
        assert_eq!(words("a   b"), ["a", "  ", " b"]);
        assert_eq!(words("end   "), ["end", "   "]);
        assert_eq!(words("x\n\ny"), ["x", "\n", "\n", "y"]);
        assert_eq!(words("café 中文"), ["café", " 中文"]);
        assert_eq!(words("'x 'LL"), ["'", "x", " '", "LL"]);
    }

    fn tok() -> Option<Tokenizer> {
        Some(Tokenizer::load(golden("models/SmolLM2-135M/tokenizer.json")?).unwrap())
    }

    /// 433 strings (whitespace, Unicode scripts, emoji, code, chat markup
    /// with special tokens, random mixes) must encode to exactly the ids HF
    /// `tokenizers` produces, and decode to exactly its text.
    #[test]
    fn matches_hf_tokenizers_on_corpus() {
        let (Some(t), Some(c), Some(e), Some(d)) =
            (tok(), golden("smollm2-135m/corpus.txt"), golden("smollm2-135m/encoded.txt"), golden("smollm2-135m/decoded.txt"))
        else {
            return;
        };
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
        println!("{} strings: encode and decode identical to HF tokenizers", strings.len());
    }
}
