# -*- coding: utf-8 -*-
"""Desk-side TTS clients for the Egyptian voice work (sampling, judging, the clip bank).

Everything here runs on the DESK, never on the phone: the app ships the pre-rendered clip bank
(assets/voice/ar/), so the phone never needs these credentials.

Auth: the developer's gcloud login (`gcloud auth print-access-token` + x-goog-user-project).
Gemini-TTS is the only Google engine with a real `ar-EG` locale and it is OAuth-only (the
app's restricted API key gets 403 — docs/voice-samples/DECISION.md, 2026-09-07); the same
bearer reaches Chirp3-HD and Vertex AI (the judge), so one login covers the whole method.
"""
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request

_TOKEN = {"v": None, "t": 0.0}
_PROJECT = {"v": None}

TTS_URL = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"

# The one style prompt every Gemini-TTS render gets. Gemini-TTS takes a natural-language
# direction next to the text; this one names the accent by its sounds (the judge scores the
# sounds), the register (a passenger giving directions, not an announcer) and the pace.
EGY_STYLE_PROMPT = (
    "Speak in natural Egyptian Arabic with a Cairo accent, the way a Cairene friend sitting "
    "next to the driver gives directions: pronounce ج as a hard g (as in 'gamda'), ق as a "
    "glottal stop (as in ''uddam'), ث as t, ذ as d or z; Egyptian vowels and Egyptian "
    "intonation, never formal Fusha and never a Gulf or Levantine accent. Warm, calm, clear "
    "and confident, a natural conversational pace, not rushed."
)


def _gcloud(args):
    return subprocess.run("gcloud " + args, shell=True, capture_output=True, text=True).stdout.strip()


def token():
    if not _TOKEN["v"] or time.time() - _TOKEN["t"] > 45 * 60:
        _TOKEN["v"] = _gcloud("auth print-access-token")
        _TOKEN["t"] = time.time()
    return _TOKEN["v"]


def project():
    if not _PROJECT["v"]:
        _PROJECT["v"] = _gcloud("config get-value project")
    return _PROJECT["v"]


def _post(url, body, timeout=120, retries=6):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": "Bearer " + token(),
            "x-goog-user-project": project(),
            "Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = "HTTP %d %s" % (e.code, e.read()[:300].decode("utf-8", "replace"))
            if e.code == 401:
                # gcloud hands back its CACHED token, which can expire long before our 45 min
                _TOKEN["v"] = None
                continue
            if e.code == 429:
                time.sleep(10 * (attempt + 1))   # a per-minute quota: wait it out
                continue
            if e.code in (499, 500, 502, 503, 504):
                time.sleep(4 * (attempt + 1))
                continue
            raise RuntimeError(last)
        except Exception as e:  # timeouts, resets
            last = str(e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(last)


def gemini_tts(text, voice="Kore", model="gemini-2.5-pro-tts", prompt=EGY_STYLE_PROMPT,
               encoding="LINEAR16", lang="ar-EG", rate=None, gain_db=None):
    """Gemini-TTS through Cloud TTS v1beta1. LINEAR16 = 24 kHz mono WAV (for the bank's
    processing); MP3 = what a listener gets in the samples."""
    inp = {"text": text}
    if prompt:
        inp["prompt"] = prompt
    cfg = {"audioEncoding": encoding}
    if encoding == "LINEAR16":
        cfg["sampleRateHertz"] = 24000
    if rate:
        cfg["speakingRate"] = rate
    if gain_db is not None:
        cfg["volumeGainDb"] = gain_db   # Gemini-TTS renders hot (RMS -14.5 dBFS, clipped peaks): -6 dB clears it
    body = {"input": inp, "voice": {"languageCode": lang, "name": voice, "modelName": model},
            "audioConfig": cfg}
    return base64.b64decode(_post(TTS_URL, body)["audioContent"])


def chirp_tts(text, voice="ar-XA-Chirp3-HD-Aoede", encoding="LINEAR16", rate=1.05):
    """The Arabic voice the app shipped until 2026-09-23 (TTS_VOICE_AR / GOOGLE_TTS_VOICE_AR; still the copilot's)."""
    cfg = {"audioEncoding": encoding, "speakingRate": rate}
    if encoding == "LINEAR16":
        cfg["sampleRateHertz"] = 24000
    body = {"input": {"text": text}, "voice": {"languageCode": voice[:5], "name": voice},
            "audioConfig": cfg}
    return base64.b64decode(_post(TTS_URL, body)["audioContent"])


def vertex_generate(model, parts, json_out=True, location="global", timeout=180):
    """Vertex AI generateContent — the audio judge."""
    host = "aiplatform.googleapis.com" if location == "global" else location + "-aiplatform.googleapis.com"
    url = "https://%s/v1/projects/%s/locations/%s/publishers/google/models/%s:generateContent" % (
        host, project(), location, model)
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0}}
    if json_out:
        body["generationConfig"]["responseMimeType"] = "application/json"
    r = _post(url, body, timeout=timeout)
    return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])


def audio_part(data, mime="audio/mpeg"):
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}}
