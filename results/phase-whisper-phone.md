# Whisper on the phone: Aether against whisper.cpp (plan F7)

aether-infer `595813f` (`whisper-stt`), whisper.cpp
`d2c6f05a46dacf663e357bcf0ed40f542a72825d` (the revision in
`tools/fetch_models.sh`). iQOO 13 (SM8750), Android 16
(`PD2408BF_EX_A_16.1.24.1.W30`), 2026-11-11.

**Power and clocks.** The phone was on battery (`AC powered: false`,
`USB powered: false`), not on AC as the earlier D / E runs were. The screen
was held on, which keeps the vendor's screen-on cap: `scaling_max_freq`
2 400 000 kHz on cores 0-5 and 2 438 400 kHz on 6-7, against the 3.53 /
4.32 GHz on AC. Both engines ran under the same caps, so the comparison is
fair, but the absolute RTFs are the battery numbers, not the AC ones.

**Thermal.** One timed run per configuration, each started at a cooled
floor (max CPU zone 38.8-44.2 °C; the idle floor is ~42 °C). A run heats
the cores to 49-66 °C (small to 66 °C), so later runs in a sweep are
slower; the values below are floor starts. The hottest zone is a
`cpu-0-*` sensor, never the charger's `pmih010x_lite_tz`.

## Build and flags

whisper.cpp: CMake 4, NDK 27.3.13750724, `android-31`, `arm64-v8a`,
Release, `BUILD_SHARED_LIBS=OFF`, examples on, server/tests off, OpenMP on
(libomp.so from the NDK pushed beside the binary). Models converted with
its own `models/convert-h5-to-ggml.py` to f16 (`ggml-model.bin`): tiny
74 MB, base 141 MB, small 465 MB.

whisper-cli: `taskset 3f` (cores 0-5), `-t 6 -l en -bs 1 -bo 1 -nf`
(greedy, one beam, no temperature fallback, timestamps on) so the decoder
rules match Aether's greedy decoder. It has no CPU-affinity flag; the
process was pinned with `taskset`.

Aether: `aether transcribe -m <model> -f jfk.wav -t 6 --cores 0-5 -l en`.
The encoder / cross-K/V / step split is from a temporary example (removed
after the runs) calling `encode`, `decoder` and 30 `step`s on the
reference's features, so the fixed-token step cost is measured, not the
clip's exact token sequence.

## jfk.wav, 11.0 s, 6 threads (cores 0-5)

| Model | Engine | Total | RTF | mel | encode | cross-K/V | decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | Aether | **0.59 s** | **18.6×** | — | 329.2 ms | 27.3 ms | 5.59 ms/token |
| tiny | whisper.cpp | 0.896 s | 12.3× | 12.7 ms | 641.4 ms | — | 2.44 ms/token |
| base | Aether | **1.34 s** | **8.2×** | — | 727.5 ms | 76.9 ms | 11.81 ms/token |
| base | whisper.cpp | 1.825 s | 6.0× | 14.3 ms | 1431.3 ms | — | 4.26 ms/token |
| small | Aether | **3.87 s** | **2.8×** | — | 2527.7 ms | 304.8 ms | 27.41 ms/token |
| small | whisper.cpp | 6.594 s | 1.7× | 11.6 ms | 5631.3 ms | — | 15.57 ms/token |

Aether's total is the CLI's transcription timer: mel + encode + decode,
without model load. whisper.cpp's total includes model load (180 ms for
base on an earlier run), mel, encode and decode. Its decode time is over
26 tokens.

**Encoder:** Aether 1.95× (tiny), 1.97× (base) and 2.23× (small) faster.
**Decoder:** whisper.cpp is faster per token: 2.3× (tiny), 2.8× (base),
1.8× (small). The decoder is now Aether's gap.

Two threads on the prime cores (6-7) are slower than six on 0-5 for Aether
at the same caps: tiny 0.77 s, base 1.51 s, small 4.91 s (step-2 runs,
language detected rather than given).

## Transcripts, base, auto language, timestamped decoding

All five clips give the same text in both engines, including the Spanish
one:

| Clip | Text (both engines) |
| --- | --- |
| jfk | And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. |
| librispeech-1 | He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce. |
| librispeech-2 | Stuff it into you, his belly countled him. |
| librispeech-3 | After early nightfall, the yellow lamps would light up here and there, the squalid quarter of the brothels. |
| librispeech-4 | Y en las ramas medios emergidas revoloteaban algunos pájaros de quimédico y legendario plumaké. |

## The rest of the phone work (step 2)

The aarch64 suite: 203 library tests pass (1 ignored; small skips as
host-only), and the C ABI's 11 including `c_abi_transcribe`. The AAR's
`Transcriber` instrumented test
(`android/aether/src/androidTest/java/dev/aether/TranscriberTest.kt`)
passes on the phone against the pushed base checkpoint: the CLI's text,
`language: "en"`, one segment, 2.47 s including the model load.

Small's checkpoint is host-only and was pushed to
`/data/local/tmp/aether/models/whisper-small` for these runs.
