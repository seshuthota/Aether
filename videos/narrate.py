"""Voice-over for the explainer films: the audio sets the timing.

A film's scene.py holds its SCRIPT, a list of (id, text) lines, and mixes in
Narrated. self.vo("id") starts a line: animations played after it run while
the voice speaks, and self.hold() waits until the clip has finished plus a
short breath. The length comes from the recorded clip in <film>/audio/
(id.wav, .mp3, ...) when it exists, otherwise from a words-per-second
estimate, so a film can be drafted before any audio is recorded. Rendering
again after recording re-times every beat to the real voice.

Each render writes, next to scene.py:
  script.md / script.txt  the lines to record (one clip per line)
  timeline.json           when each line starts in the rendered video

Commands (run with any python 3; ffmpeg/ffprobe on PATH):
  python videos/narrate.py mux   videos/<film> silent.mp4 out.mp4
  python videos/narrate.py vtt   videos/<film> out.vtt
  python videos/narrate.py check videos/<film>   # Whisper on Groq, needs .env
"""

import json
import os
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")
WORDS_PER_SECOND = 2.6  # estimate for lines not recorded yet
BREATH = 0.45  # pause after each line before the next one starts


def find_clip(folder, key):
    for ext in AUDIO_EXTS:
        path = Path(folder) / "audio" / f"{key}{ext}"
        if path.exists():
            return path
    return None


def clip_seconds(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def estimate_seconds(text):
    return len(text.split()) / WORDS_PER_SECOND + 0.3


class Narrated:
    """Mix into a Scene. Call self.start_voice(__file__, SCRIPT) first and
    self.finish_voice() at the end of construct()."""

    def start_voice(self, scene_file, script):
        self.voice_dir = Path(scene_file).parent
        self.script = dict(script)
        self.script_order = [key for key, _ in script]
        assert len(self.script) == len(script), "duplicate line id in SCRIPT"
        self.timeline = []
        self.deadline = 0.0

    def vo(self, key, after=BREATH):
        text = self.script[key]
        clip = find_clip(self.voice_dir, key)
        length = clip_seconds(clip) if clip else estimate_seconds(text)
        self.timeline.append({"id": key, "start": round(self.time, 3),
                              "seconds": round(length, 3), "recorded": clip is not None,
                              "text": text})
        self.deadline = self.time + length + after

    def hold(self, extra=0.0):
        left = self.deadline + extra - self.time
        if left > 0.02:
            self.wait(left)

    def say(self, key, extra=0.0):
        self.vo(key)
        self.hold(extra)

    def finish_voice(self):
        used = [item["id"] for item in self.timeline]
        missing = [key for key in self.script_order if key not in used]
        assert not missing, f"SCRIPT lines never spoken: {missing}"
        recorded = sum(item["recorded"] for item in self.timeline)
        (self.voice_dir / "timeline.json").write_text(json.dumps(
            {"duration": round(self.time, 3), "recorded": recorded,
             "lines": self.timeline}, indent=2, ensure_ascii=False) + "\n")
        self._write_script()

    def _write_script(self):
        name = self.voice_dir.name
        rows = ["# Narration script", "",
                f"Film: `videos/{name}/`. Record one clip per line and save it as",
                f"`videos/{name}/audio/<id>.wav` (or .mp3). The id is the file name.",
                "Then re-render: every beat stretches or shrinks to fit the real voice.", "",
                "| id | line |", "|---|---|"]
        rows += [f"| `{key}` | {self.script[key]} |" for key in self.script_order]
        words = sum(len(self.script[k].split()) for k in self.script_order)
        rows += ["", f"{len(self.script_order)} lines, {words} words "
                     f"(about {words / WORDS_PER_SECOND / 60:.1f} min of speech).", ""]
        (self.voice_dir / "script.md").write_text("\n".join(rows))
        (self.voice_dir / "script.txt").write_text(
            "".join(f"{key}\t{self.script[key]}\n" for key in self.script_order))


# ---------------------------------------------------------------- commands
def load_timeline(folder):
    return json.loads((Path(folder) / "timeline.json").read_text())


def mux(folder, video, out):
    """Lay every recorded clip at its start time under the silent video."""
    tl = load_timeline(folder)
    clips = [(item, find_clip(folder, item["id"])) for item in tl["lines"]]
    clips = [(item, path) for item, path in clips if path]
    if not clips:
        sys.exit("no clips in audio/ yet")
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(video)]
    parts = []
    for k, (item, path) in enumerate(clips):
        args += ["-i", str(path)]
        ms = int(item["start"] * 1000)
        parts.append(f"[{k + 1}:a]aresample=48000,aformat=channel_layouts=mono,"
                     f"adelay={ms}:all=1[a{k}]")
    mix = "".join(f"[a{k}]" for k in range(len(clips)))
    parts.append(f"{mix}amix=inputs={len(clips)}:normalize=0:dropout_transition=0,"
                 f"apad[aout]")
    # The render holds still frames instead of repeating them, so its last frame
    # has no length. Re-encode at a constant 30 fps, padded to the scene's full
    # duration, so the final line is not cut off.
    end = tl["duration"]
    parts.append(f"[0:v]tpad=stop_mode=clone:stop_duration=30,fps=30[vout]")
    args += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]",
             "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-t", f"{end:.3f}", "-movflags", "+faststart",
             str(out)]
    subprocess.run(args, check=True)
    print(f"{out}: {len(clips)} of {len(tl['lines'])} lines voiced")


def stamp(t):
    h, rest = divmod(t, 3600)
    mins, sec = divmod(rest, 60)
    return f"{int(h):02d}:{int(mins):02d}:{sec:06.3f}"


def vtt(folder, out):
    """Subtitles from the script, for viewers who cannot play sound."""
    rows = ["WEBVTT", ""]
    for item in load_timeline(folder)["lines"]:
        end = item["start"] + item["seconds"]
        rows += [f"{stamp(item['start'])} --> {stamp(end)}", item["text"], ""]
    Path(out).write_text("\n".join(rows))
    print(out)


def env_settings():
    env = dict(os.environ)
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    base = env.get("OPENAI_COMPAT_BASE_URL", "")
    base = base.split("/chat/completions")[0].rstrip("/")
    keys = [k.strip() for k in env.get("OPENAI_COMPAT_API_KEY", "").split(",") if k.strip()]
    model = env.get("OPENAI_COMPAT_MODEL", "whisper-large-v3-turbo")
    if not base or not keys:
        sys.exit("set OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY in .env")
    return base, keys, model


def transcribe(path, base, keys, model):
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in (("model", model), ("response_format", "json"), ("language", "en")):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
             ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    last = None
    for key in keys:  # several keys: move to the next one when one is rate limited
        req = urllib.request.Request(
            f"{base}/audio/transcriptions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "User-Agent": "aether-narrate/1.0",
                     "Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["text"]
        except urllib.error.HTTPError as err:
            last = err
            if err.code not in (401, 429):
                raise
    raise last


def words(text):
    text = text.lower().replace("\u2019", "'")
    for short, full in (("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"),
                        ("'m", " am"), ("'s", "s")):
        text = text.replace(short, full)
    keep = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return keep.split()


def check(folder):
    """Transcribe each recorded clip and flag lines that do not match the script."""
    import difflib
    base, keys, model = env_settings()
    script = dict(line.split("\t", 1) for line in
                  (Path(folder) / "script.txt").read_text().splitlines() if "\t" in line)
    bad = 0
    for key, text in script.items():
        path = find_clip(folder, key)
        if not path:
            print(f"  missing  {key}")
            continue
        heard = transcribe(path, base, keys, model)
        ratio = difflib.SequenceMatcher(None, words(text), words(heard)).ratio()
        flag = "ok  " if ratio >= 0.85 else "DIFF"
        bad += flag == "DIFF"
        print(f"  {flag} {ratio:4.2f}  {key}")
        if flag == "DIFF":
            print(f"        script: {text}\n        heard:  {heard.strip()}")
    print(f"{bad} clip(s) to look at")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "mux" and len(rest) == 3:
        mux(*rest)
    elif cmd == "vtt" and len(rest) == 2:
        vtt(*rest)
    elif cmd == "check" and len(rest) == 1:
        check(rest[0])
    else:
        print(__doc__)
