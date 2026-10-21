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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::golden;
    use crate::tokenizer::bytelevel::Tokenizer;

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
