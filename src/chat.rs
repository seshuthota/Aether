//! Chat prompts. Instruction-tuned models are trained on conversations in a
//! fixed text format; a plain question without it gets plain continuation
//! instead of an answer.
//!
//! Qwen2.5 (and SmolLM2-Instruct) use **ChatML**: every turn is
//! `<|im_start|>{role}\n{text}<|im_end|>\n`, and the prompt ends with an
//! open assistant turn for the model to complete. `<|im_start|>` and
//! `<|im_end|>` are special tokens (one id each), which the tokenizer
//! matches before splitting the text. The model ends its answer by
//! producing `<|im_end|>`.

/// The system message Qwen2.5's template uses when none is given.
pub const QWEN_DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// A one-question ChatML prompt, ready for the assistant's answer.
pub fn chatml(system: &str, user: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// LFM3's ChatML: a BOS first, an optional system turn, then the user turn
/// and an open assistant turn. With thinking on (the model's default) the
/// assistant turn starts with `<think>`; with it off it starts with an empty
/// `<think></think>` block.
pub fn lfm3(system: Option<&str>, user: &str, thinking: bool) -> String {
    let mut s = String::from("<|startoftext|>");
    if let Some(system) = system.filter(|s| !s.is_empty()) {
        s.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }
    s.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"));
    s.push_str(if thinking { "<think>" } else { "<think></think>" });
    s
}

/// Gemma 4's template (in the GGUF; checked against HF
/// `apply_chat_template`): an optional system turn, the user turn and an
/// open model turn. Thinking is switched on by `<|think|>` at the top of the
/// system turn (one is opened for it if there is no system message). The
/// BOS is not part of the text: the tokenizer adds it.
pub fn gemma4(system: Option<&str>, user: &str, thinking: bool) -> String {
    let mut s = String::new();
    let system = system.filter(|s| !s.is_empty());
    if thinking || system.is_some() {
        s.push_str("<|turn>system\n");
        if thinking {
            s.push_str("<|think|>\n");
        }
        s.push_str(system.unwrap_or(""));
        s.push_str("<turn|>\n");
    }
    s.push_str(&format!("<|turn>user\n{user}<turn|>\n<|turn>model\n"));
    s
}

/// Hides a `<think>...</think>` span (Gemma 4: `<|channel>...<channel|>`)
/// from a streamed reply. Pieces may split the tags, so it keeps the last
/// few bytes until it knows whether they start a tag.
pub struct ThinkFilter {
    hide: bool,
    buf: Vec<u8>,
    hiding: bool,
    open: &'static [u8],
    close: &'static [u8],
}

impl ThinkFilter {
    pub fn new(hide: bool) -> ThinkFilter {
        ThinkFilter::tags(hide, b"<think>", b"</think>")
    }

    /// A filter for the span between `open` and `close`.
    pub fn tags(hide: bool, open: &'static [u8], close: &'static [u8]) -> ThinkFilter {
        ThinkFilter { hide, buf: Vec::new(), hiding: false, open, close }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<u8> {
        if !self.hide {
            return bytes.to_vec();
        }
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            if self.hiding {
                match find(&self.buf, self.close) {
                    Some(p) => {
                        self.buf.drain(..p + self.close.len());
                        self.hiding = false;
                    }
                    None => break,
                }
            } else {
                match find(&self.buf, self.open) {
                    Some(p) => {
                        out.extend_from_slice(&self.buf[..p]);
                        self.buf.drain(..p + self.open.len());
                        self.hiding = true;
                    }
                    None => {
                        let keep = (self.open.len() - 1).min(self.buf.len());
                        let emit = self.buf.len() - keep;
                        out.extend_from_slice(&self.buf[..emit]);
                        self.buf.drain(..emit);
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<u8> {
        if !self.hide || self.hiding {
            self.buf.clear();
            return Vec::new();
        }
        std::mem::take(&mut self.buf)
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;
    use crate::tokenizer::bytelevel::Tokenizer;

    /// The four prompts HF `apply_chat_template` (transformers 5.20.2,
    /// `google/gemma-4-E2B-it`, `add_generation_prompt=True`) renders, less
    /// the `<bos>` the tokenizer adds.
    #[test]
    fn gemma4_template_matches_hf() {
        let (sys, q) = ("You are terse.", "What is 2+2?");
        assert_eq!(gemma4(None, q, false), "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
        assert_eq!(gemma4(Some(sys), q, false), "<|turn>system\nYou are terse.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
        assert_eq!(gemma4(None, q, true), "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
        assert_eq!(gemma4(Some(sys), q, true), "<|turn>system\n<|think|>\nYou are terse.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
    }

    #[test]
    fn think_filter_hides_split_spans() {
        let mut f = ThinkFilter::new(true);
        let mut out = Vec::new();
        for piece in [b"a<th".as_slice(), b"ink>b</thi", b"nk>c"] {
            out.extend(f.feed(piece));
        }
        out.extend(f.finish());
        assert_eq!(out, b"ac");
        assert_eq!(ThinkFilter::new(false).feed(b"x"), b"x");
        let mut g = ThinkFilter::tags(true, b"<|channel>", b"<channel|>");
        let mut out = Vec::new();
        for piece in [b"<|chan".as_slice(), b"nel>thought\nhm<chan", b"nel|>4"] {
            out.extend(g.feed(piece));
        }
        out.extend(g.finish());
        assert_eq!(out, b"4");
    }

    /// LFM3's template: BOS, optional system, and `<think>` (or an empty
    /// think block) on the assistant's generation prompt.
    #[test]
    fn lfm3_matches_transformers_chat_template() {
        let (Some(tok), Some(cases)) = (crate::testutil::golden("models/LFM3-1.7B/tokenizer.json"), crate::testutil::golden("lfm3/chat.txt")) else { return };
        let tok = Tokenizer::load(tok).unwrap();
        for line in std::fs::read_to_string(cases).unwrap().lines() {
            let (case, ids) = line.split_once('\t').unwrap();
            let case = crate::format::json::Json::parse(case).unwrap();
            let system = case.get("system").and_then(crate::format::json::Json::as_str);
            let user = case.get("user").and_then(crate::format::json::Json::as_str).unwrap();
            let think = case.get("think").and_then(crate::format::json::Json::as_bool).unwrap();
            let want: Vec<usize> = ids.split(' ').map(|x| x.parse().unwrap()).collect();
            assert_eq!(tok.encode(&lfm3(system, user, think)), want, "case {case:?}");
        }
    }

    /// Rendering + tokenizing gives exactly the ids of transformers'
    /// `apply_chat_template(..., add_generation_prompt=True)`.
    #[test]
    fn matches_transformers_chat_template() {
        let (Some(tok), Some(cases)) = (golden("models/Qwen2.5-0.5B-Instruct/tokenizer.json"), golden("qwen2.5-0.5b/chat.txt")) else { return };
        let tok = Tokenizer::load(tok).unwrap();
        for line in std::fs::read_to_string(cases).unwrap().lines() {
            let (msg, ids) = line.split_once('\t').unwrap();
            let msg = crate::format::json::Json::parse(msg).unwrap();
            let want: Vec<usize> = ids.split(' ').map(|x| x.parse().unwrap()).collect();
            assert_eq!(tok.encode(&chatml(QWEN_DEFAULT_SYSTEM, msg.as_str().unwrap())), want);
        }
    }
}
