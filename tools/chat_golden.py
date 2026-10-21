"""Chat-template goldens: the token ids transformers' apply_chat_template
produces for a few user messages (with the model's default system prompt).

    tools/.venv/bin/python -I tools/chat_golden.py testdata/models/Qwen2.5-0.5B-Instruct testdata/qwen2.5-0.5b/chat.txt

One line per message: the message (JSON string), a tab, the ids.
"""

import json
import sys

from transformers import AutoTokenizer

MESSAGES = [
    "Hello!",
    "What is the capital of France? Answer in one word.",
    "Write a haiku about a phone that runs a language model.",
    "Multi-line\ninput with  spaces and émojis 🎉",
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for m in MESSAGES:
            ids = tok.apply_chat_template([{"role": "user", "content": m}], add_generation_prompt=True, tokenize=True)
            if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            f.write(json.dumps(m) + "\t" + " ".join(map(str, ids)) + "\n")
    print(f"{len(MESSAGES)} chat prompts")


if __name__ == "__main__":
    main()
