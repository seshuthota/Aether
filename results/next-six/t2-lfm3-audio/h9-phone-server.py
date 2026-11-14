"""H5 over HTTP: the openai package against `aether serve` with LFM3-Audio (on the phone, adb-forwarded)."""
import base64, sys, time
from openai import OpenAI
c = OpenAI(base_url=sys.argv[1], api_key="none")
clip = base64.b64encode(open(sys.argv[2], "rb").read()).decode()
t = time.time()
r = c.chat.completions.create(model="x", temperature=0, messages=[
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}}]},
])
print("chat, ASR prompt:", repr(r.choices[0].message.content), r.usage.prompt_tokens, f"{time.time() - t:.1f}s")
t = time.time()
r = c.chat.completions.create(model="x", temperature=0, messages=[
    {"role": "user", "content": [{"type": "text", "text": "Who said this, and when?"}, {"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}}]},
])
print("chat, question:", repr(r.choices[0].message.content), f"{time.time() - t:.1f}s")
t = time.time()
r = c.audio.transcriptions.create(model="x", file=open(sys.argv[2], "rb"), language="en")
print("transcription:", repr(r.text), f"{time.time() - t:.1f}s")
t = time.time()
s = c.audio.speech.create(model="x", voice="female", input="Hello from the phone.", response_format="wav")
open(sys.argv[3], "wb").write(s.content)
print("speech:", len(s.content), "bytes", f"{time.time() - t:.1f}s")
