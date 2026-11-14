# T2: LFM3-Audio-2B, speech in and speech out (plans/next-six.md in aether-infer)

Branch `lfm3-audio`. Model `LiquidAI/LFM3-Audio-2B` at `2d6c1f9` (HF) and its
GGUF repo at `39670cb`. Reference: transformers f32 on the CPU with the
checkpoint's remote code (`tools/hf_dump_lfm3_audio.py`). PC: i7-13620H.

## H1–H8 on the PC (goldens from the reference)

| Step | Check | Measured |
| --- | --- | --- |
| H1 log-mel | the processor's features, 4 clips | ≤ 2e-3 worst (the f32 reference's own looseness at near-silent bins; f64 numpy differs from it by 1.27e-3) |
| H2–H3 encoder | every layer and the projector | F16 mmproj ≤ 1.3e-6; Q8_0 ≤ 2e-2 |
| H4 ASR | prompt ids, last logits, greedy text | ids identical; logits ≤ 3e-6 (BF16 LM); text identical on 4 of 4 with the BF16, Q8_0 and Q4_K_M LMs |
| H5 API | `Session::chat` with clips and `Transcriber::from_model` | the H4 transcripts; the `openai` package (3.49) against `aether serve -m LFM3-Audio-2B-Q8_0 --mmproj …Q8_0`: JFK transcribed exactly; `/v1/audio/speech` answers a WAV (`h5-openai-check.py`) |
| H6 depth decoder | every pass's logits, the reference's codes fed in | ≤ 9.6e-7 |
| H7 codec | waveform for the reference's codes; streamed against whole | SNR 114–118 dB; stream = whole ≤ 1e-6 |
| H8 generation | greedy speech (3 prompts) and interleaved replies (2) with the reference's MaskGIT orders | every frame's codes and every text token identical |
| H8 API | sampled speech (Q8_0) through `Synthesizer::from_model` and `Session::chat_speech`, transcribed by Whisper base | the words of the text; the interleaved reply "The capital of France is Paris. Would you like to know more about the city?" heard exactly |

On the PC (x86, no tuned kernels) any LFM3 runs at about 8 tokens a second,
LFM3-1.7B's text as well; speed is the phone's question.

## H9 on the phone

iQOO 13 (SM8750), `--cores 2-7` (6 threads), what an app ships:
`LFM3-Audio-2B-Q4_K_M.gguf` (1.10 GB) + `mmproj-LFM3-Audio-2B-Q8_0.gguf`
(304 MB). On battery (72%, not charging), screen kept on, wireless adb, CPU
42–44 °C at the start of each batch. Raw output: `h9-phone-*.txt`. Peak RSS
from `toybox time -v`; times from the CLI (the model load, about 0.9 s from
a warm page cache, excluded).

### Speech to text (`aether transcribe -l LANG`)

| Clip | Audio | Time | Real-time factor | Peak RSS | Transcript |
| --- | --- | --- | --- | --- | --- |
| jfk (en) | 11.0 s | 1.84 s | 6.0× | 1.42 GB | identical to the reference |
| fr | 4.2 s | 0.91 s | 4.7× | 1.35 GB | identical |
| zh | 6.6 s | 1.25 s | 5.3× | 1.40 GB | identical |

### Text to speech (`aether speak`, seed 7, the card's sampling)

| Text | Voice, language | Audio | Time | Real-time factor | First audio | Peak RSS | Whisper base hears |
| --- | --- | --- | --- | --- | --- | --- | --- |
| The quick brown fox jumps over the lazy dog. | female, en | 3.36 s | 1.67 s | 2.0× | 363 ms | 1.47 GB | exactly |
| Please call me back after three o'clock. | male, en-gb | 2.40 s | 1.26 s | 1.9× | 283 ms | 1.46 GB | exactly ("3 o'clock") |
| 今天天气很好，我们去公园散步吧。 | female, zh | 3.36 s | 1.60 s | 2.1× | 296 ms | 1.47 GB | exactly (traditional characters) |

### Voice chat, text and speech interleaved (`aether speak --chat`, seed 5)

| Question | Prompt | First text | First audio | Speech | Time | Real-time factor | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| "What is the capital of France?" (text) | 29 tokens, 311 ms | 311 ms | 443 ms | 4.24 s | 2.26 s | 1.9× | 1.47 GB |
| the same, spoken (Qwen3-TTS, 3 s at 16 kHz) | 59 tokens, 417 ms | 418 ms | 550 ms | 2.40 s | 1.50 s | 1.6× | 1.58 GB |
| "Tell me one fact about the moon." (text) | 30 tokens, 211 ms | 212 ms | 348 ms | 7.12 s | 3.44 s | 2.1× | 1.48 GB |

Replies: "The capital of France is Paris. Would you like to know more about the
city?"; "The capital of France is Paris."; "The Moon is Earth's natural
satellite, the only celestial body we can see without a telescope. Would you
like another fact?". Whisper base hears each reply's speech as its text.
The CPU reached 57–69 °C by the end of each batch.

### Over HTTP on the phone (`aether serve`, `h9-phone-server.py`)

The `openai` package (3.49) on the PC against `aether serve -m
LFM3-Audio-2B-Q4_K_M.gguf --mmproj mmproj-LFM3-Audio-2B-Q8_0.gguf --cores
2-7` on the phone (adb-forwarded), jfk.wav, temperature 0:

| Request | Answer | Wall time |
| --- | --- | --- |
| chat, "Perform ASR in English." + `input_audio` | the reference's transcript, 158 prompt tokens | 2.0 s |
| chat, "Who said this, and when?" + `input_audio` | "A male speaker with a deep, resonant voice says this in a formal, public setting, likely during a political speech. The exact date or event is not mentioned in the audio." | 2.0 s |
| `/v1/audio/transcriptions`, `language=en` | the reference's transcript | 1.7 s |
| `/v1/audio/speech`, female, "Hello from the phone." | an 84 KB WAV | 1.1 s |

### The AAR on the phone

`AudioChatTest` (android/aether/src/androidTest): `hearsAClip` (chat with
the clip: 158 prompt tokens, the transcript; `Transcriber.fromSession`)
6.4 s, `speaks` (`Synthesizer.fromSession`, `chatSpeech`: "The capital of
France is Paris", END_OF_TURN) 6.9 s, `refusesWhatItCannotHear` (44.1 kHz,
a session without the encoder) 3.0 s; all pass. Run as one test (all of
it, about 15 s) it never reported: the vendor's `fast_freezer` freezes the
instrumentation process 10 s after it starts (`am_app_frozen` in the events
log, the threads in `do_freezer_trap`) and `am unfreeze --sticky` does not
stop it; so each test runs as its own `am instrument -e class
dev.aether.AudioChatTest#NAME`.

## H8: intelligibility of sampled speech (24 clips)

`tools/tts_quality.py`: 6 English and 2 Chinese texts, seeds 1–3, voices as
genders, the card's sampling. The reference: `ref-lfm3` (transformers f32,
CPU, 110 s of audio). The engine: `aether speak` on the phone, Q4_K_M LM +
Q8_0 mmproj (108 s; timings in `h8-phone-tts-24.txt`: 1.5–2.2× real time,
first audio 297–488 ms). Both scored by Whisper small (transformers):

| Audio from | English letter errors | Chinese character errors |
| --- | --- | --- |
| reference, f32 | 0.18% (en3-s3 drops "to") | 1.11% (one: "3" for 三) |
| engine, Q4_K_M, phone | 0% | 1.11% (one: 报告告) |

Per clip: `h8-intelligibility-scores.json`.
