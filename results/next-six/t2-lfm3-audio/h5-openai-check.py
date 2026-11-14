import sys, time
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:18093/v1", api_key="none")
wav = sys.argv[1]
t = time.time()
r = c.audio.transcriptions.create(model="x", file=open(wav, "rb"), language="en")
print("transcription:", repr(r.text), f"{time.time()-t:.1f}s")
r = c.audio.transcriptions.create(model="x", file=open(wav, "rb"), language="en", response_format="verbose_json")
print("verbose:", r.language, r.duration, len(r.segments))
t = time.time()
speech = c.audio.speech.create(model="x", voice="male", input="Hello from the engine.", response_format="wav")
open(sys.argv[2], "wb").write(speech.content)
print("speech bytes:", len(speech.content), f"{time.time()-t:.1f}s")
r = c.audio.transcriptions.create(model="x", file=open(sys.argv[2], "rb"), language="en") if False else None
