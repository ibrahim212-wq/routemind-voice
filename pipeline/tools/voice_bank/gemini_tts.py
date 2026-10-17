# -*- coding: utf-8 -*-
"""The bank's Gemini TTS call, STREAMED — the audition's request exactly (tools/voice_audition/audition.py `synth`:
the same body, the same cache key and files, so nothing is ever asked twice between the two), sent to
`streamGenerateContent?alt=sse` so every request also measures its TIME TO FIRST AUDIO (the first audio chunk's
arrival) at no extra request. Measured 2026-10-17 on gemini-3.8-flash-lite-tts (Despina, 3 lines): the first chunk
after 1.79 s, 83 chunks, the whole 6.7 s of audio after 2.88 s.

A model that refuses the stream (HTTP 400 naming it) is asked once more the audition's way (generateContent); quota
answers are the audition's (`Limits`: a per-day 429 → Exhausted, the day's stop; per-minute → wait it out).
"""
import base64
import datetime
import io
import json
import os
import time
import urllib.error
import urllib.request

from tools.masri_voice import acoustics
from tools.voice_audition import audition as A

STREAM_API = "https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse"
_NO_STREAM = set()       # models that refused the stream this run


def _stream(model, body, timeout=600):
    """One streamed request → (response dict shaped like generateContent's, seconds to the first audio chunk)."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(STREAM_API % model, data=data, headers={
        "x-goog-api-key": A.api_key(), "Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    first = None
    audio, mime, usage, finish, texts = [], None, None, None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            usage = d.get("usageMetadata") or usage
            c = (d.get("candidates") or [{}])[0]
            finish = c.get("finishReason") or finish
            for p in (c.get("content") or {}).get("parts") or []:
                if "inlineData" in p:
                    if first is None:
                        first = time.time() - t0
                    mime = mime or p["inlineData"].get("mimeType")
                    audio.append(base64.b64decode(p["inlineData"]["data"]))
                elif p.get("text"):
                    texts.append(p["text"])
    parts = [{"inlineData": {"mimeType": mime or "", "data": base64.b64encode(b"".join(audio)).decode()}}] \
        if audio else [{"text": "".join(texts)}] if texts else []
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}], "usageMetadata": usage}, first


def call_stream(model, body, max5xx=None):
    """audition.call's protocol (pacing, 429 per-minute / per-day, 5xx back-off) over the stream → (r, first_s)."""
    n5xx = 0
    soft429 = 0
    for attempt in range(10):
        if A.STATS["max_calls"] is not None and A.STATS["calls"] >= A.STATS["max_calls"]:
            raise A.Exhausted("--max-calls reached")
        A.LIM.wait(model)
        A.STATS["calls"] += 1
        t0 = time.time()
        try:
            r, first = _stream(model, body)
            A.LIM.ok(model, time.time() - t0, r.get("usageMetadata"))
            return r, first
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(raw)
                err = (err[0] if isinstance(err, list) else err).get("error", {})
            except (ValueError, AttributeError, IndexError):
                err = {"message": raw[:300]}
            if e.code == 429:
                per_day, retry, viol = A.LIM.note_429(model, err)
                ids = ",".join("%s=%s" % (v.get("quotaId"), v.get("quotaValue")) for v in viol) or "no details"
                A.log("  429 %s (%s) retry=%s" % (model, ids, retry))
                if per_day:
                    raise A.Exhausted("%s: per-day quota (%s)" % (model, ids))
                soft429 += 1
                if not viol and soft429 >= 5:
                    A.LIM.m(model)["blocked_until"] = time.time() + 3600
                    A.LIM.save()
                    raise A.Exhausted("%s: repeated 429 without details" % model)
                time.sleep(max(retry or 0.0, A.LIM.m(model)["gap_s"], 15.0) + 1.0)
                continue
            if e.code in (500, 502, 503, 504):
                A.LIM.m(model)["e5xx"] += 1
                A.LIM.save()
                n5xx += 1
                if max5xx is not None and n5xx >= max5xx:
                    raise A.Overloaded(model)
                wait = min(120.0, 6.0 * (2 ** min(attempt, 4)))
                A.log("  %d %s — %s; retry in %.0fs" % (e.code, model, A.scrub(err.get("message", ""))[:120], wait))
                time.sleep(wait)
                continue
            if e.code == 400 and "stream" in (err.get("message") or "").lower():
                _NO_STREAM.add(model)
                A.log("  %s refused the stream (%s) — generateContent" % (model, A.scrub(err.get("message"))[:120]))
                return A.call(model, body, "tts"), None
            A.LIM.m(model)["e_other"] += 1
            A.LIM.save()
            raise RuntimeError(A.scrub("HTTP %d %s: %s" % (e.code, model, err.get("message", raw[:300]))))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            A.log("  network %s: %s" % (model, A.scrub(e)))
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError("%s: gave up after retries" % model)


def synth(model, voice, prompt, take):
    """= audition.synth (cache key, files, meta) + meta["first_audio_s"] / ["streamed"]. → (key, wav, meta)."""
    key = A.sha(model, voice, prompt, *(["take=%d" % take] if take > 1 else []))
    wav_p = os.path.join(A.CACHE, key + ".wav")
    meta_p = os.path.join(A.CACHE, key + ".json")
    if os.path.exists(wav_p) and os.path.exists(meta_p):
        return key, open(wav_p, "rb").read(), json.load(io.open(meta_p, encoding="utf-8"))
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for attempt in range(2):
        t0 = time.time()
        if model in _NO_STREAM:
            r, first = A.call(model, body, "tts"), None
        else:
            r, first = call_stream(model, body)
        lat = time.time() - t0
        parts = ((r.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        aud = [p["inlineData"] for p in parts if "inlineData" in p and p["inlineData"].get("data")]
        if aud:
            mime = aud[0].get("mimeType", "")
            wav_b = A.to_wav(mime, b"".join(base64.b64decode(a["data"]) for a in aud))
            x, rate = acoustics.pcm(wav_b)
            meta = {"model": model, "voice": voice, "take": take, "mime": mime, "latency_s": round(lat, 2),
                    "first_audio_s": round(first, 2) if first is not None else None, "streamed": first is not None,
                    "usage": r.get("usageMetadata"), "audio_s": round(len(x) / float(rate), 2),
                    "at": datetime.datetime.now().isoformat(timespec="seconds")}
            A.LIM.m(model)["audio_s"] += meta["audio_s"]
            A.LIM.save()
            os.makedirs(A.CACHE, exist_ok=True)
            open(wav_p, "wb").write(wav_b)
            io.open(meta_p, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False))
            return key, wav_b, meta
        A.LIM.m(model)["no_audio"] += 1
        A.LIM.save()
        fr = ((r.get("candidates") or [{}])[0]).get("finishReason")
        A.log("  no audio from %s/%s (finishReason=%s)" % (model, voice, fr))
    raise RuntimeError("no audio from %s/%s" % (model, voice))
