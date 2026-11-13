set -u
cd /home/curious/Documents/hermes-projects/aether-infer-kv
cargo test --release -p aether-infer --lib -- f16_kv_drift --nocapture > /tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/tts-f16.log 2>&1; echo "drift test exit $?"
grep -E "qwen3-tts|test result|panicked" /tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/tts-f16.log
cargo build --release -p aether-cli 2>&1 | tail -1
R=/home/curious/Documents/hermes-projects/aether-infer
for v in gguf-f16:$R/hostdata/qwen3-tts-0.6b-q4_k_m.gguf f32-f16:$R/hostdata/Qwen3-TTS-12Hz-0.6B-CustomVoice; do
  d=/tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/ttsq/${v%%:*}; m=${v#*:}; mkdir -p $d
  echo "== $d $(date +%T)"
  python3 tools/tts_quality.py list | while IFS=$'\t' read -r name voice lang seed text; do
    [ -s $d/$name.wav ] || target/release/aether speak -m $m -ctk f16 --voice "$voice" -l "$lang" -s "$seed" -p "$text" -o $d/$name.wav </dev/null >/dev/null 2>&1
  done
  ls $d | wc -l
done
tools/.venv/bin/python -I tools/tts_quality.py score $R/hostdata/whisper-small /tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/ttsq/gguf-f16 /tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/ttsq/f32-f16 2>&1 | grep -v "Warning\|Loading\|transformers\|Ignoring\|Passing"
echo "== done $(date +%T)"
