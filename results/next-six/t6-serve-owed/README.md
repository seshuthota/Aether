# T6: `aether serve`'s owed items (plan `next-six.md`, branch `serve-owed`)

Host for the PC work: i7-13620H (16 threads), 23 GB RAM, Linux x86_64,
Docker 29.9.3, `rustc` 1.99.0 on the host, Rust 1.98 inside the image.
The phone rows (W3's second half, W2's device checks) are recorded when the
iQOO 13 is connected.

## W1: Docker image

- `Dockerfile` and `.dockerignore` at the repository root. Build stage
  `rust:1.98-bookworm` (the MSRV, not the host's 1.99), runtime
  `debian:bookworm-slim`, non-root user `aether` (uid 10001), models at
  `/models`, `EXPOSE 8080`, entrypoint
  `aether serve --host 0.0.0.0`. CPU only. The binary is stripped in the
  build stage (Debian slim has no `strip`).
- Image: `aether:serve-owed`, amd64, **118,478,841 bytes (118 MB)**, id
  `sha256:7459235c85872bcf5e80d01a8089911694357e9944ff3ab805b2e27a5758ac3b`.
- `linux/arm64`: **not built**. The buildx builder advertises only
  `linux/amd64`, and no `qemu-aarch64` is registered in
  `/proc/sys/fs/binfmt_misc` (only `python3.12`), so buildx cannot run the
  arm64 Rust toolchain. Untested.
- CI: a `docker` job in `.github/workflows/ci.yml` builds the image and runs
  `--help`; no push (the repository is private, publishing waits).

Check (`w1-check.txt`): the image and the native binary on the same GGUF
`testdata/gguf/smollm2-360m-instruct-q8_0.gguf`
(sha256 `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`),
both `-t 8`, `/v1/models` byte for byte equal, and an `openai` 3.49.0
greedy chat (`temperature 0`, `max_tokens 48`, `cache_prompt false`) gives
the same text, `finish_reason` and usage on both:

> The sky appears blue because of a phenomenon called Rayleigh scattering,
> which occurs when sunlight enters Earth's atmosphere and encounters tiny
> molecules of gases such as nitrogen and oxygen. These molecules scatter
> the shorter wavelengths of light, like blue and violet, more

`length`, 41 prompt tokens, 48 completion tokens, on both. PASS.

## W2: Open WebUI v0.13.3

`ghcr.io/open-webui/open-webui:v0.13.3` (released 2026-11-08), its OpenAI
connection at `aether serve` with Qwen2.5-0.5B Q4_0 (`-np 2 -t 8`,
`--whisper hostdata/whisper-small`, `--tts
hostdata/qwen3-tts-0.6b-q4_k_m.gguf`). Driven with Playwright and Google
Chrome; logs in `w2-log.txt`, `w2b-log.txt`, `diag-log.txt`, screenshots
`webui-*.png`.

**Required setting.** v0.13.3 sends a `tools` array (its built-in tools) in
every chat request. `aether serve` answers 400 for `tools` (a known but
unsupported field is named, as the plan decides), so the model's **Builtin
Tools must be turned off** (Admin Panel > Settings > Models > the model >
Builtin Tools, all twelve categories). Then the upstream request is just
`{"stream":true,"model":…,"messages":…}` and chat works. This is a
configuration note in `docs/serve.md`; it is not an engine change.

| Check | Result |
| --- | --- |
| Model list | pass |
| Streamed chat | pass: first text 2.7 s, deltas over the websocket, 7.96 cores while generating |
| Stop mid-reply | pass: 65 characters, no growth over 5 s, Stop gone; the proxy logged the upstream client went away after 6144 bytes; a direct socket close drops `aether serve` from 4.0 cores to 0.38 and then 0.0 |
| Regenerate | pass, through the ↻ menu's **Try Again** (a bare click only opens the menu) |
| System prompt | pass: `params.system` stored; the reply changed to "Hello, mate!" |
| Two chats at once | pass: both requests answered with `-np 2` |
| Title generation | pass |
| Text to speech | pass: `/v1/audio/speech` returned 200, Open WebUI transcoded the 24 kHz WAV to an 18,668-byte MP3 |
| Speech to text | **fail**: Open WebUI calls `convert_audio_to_mp3` before `/v1/audio/transcriptions`; Aether accepts 16 kHz WAV only and answered `not a WAV file (no RIFF / WAVE header)`. `owui-tts.mp3` is the TTS result. |
| Image in a chat | pending T1 |

The STT failure is Open WebUI's format choice against Aether's documented
WAV-only input. Fixing it means an MP3 decoder in Aether (a new dependency
and a new decision) or a change in Open WebUI, so it is left open here.

Small caveat on the host: Open WebUI's first start downloads its
`sentence-transformers/all-MiniLM-L6-v2` embedding model from Hugging Face;
with `--network=host` that start hung at "application startup" on this
machine, so the final runs used a bridge network and
`--add-host=host.docker.internal:host-gateway`.
