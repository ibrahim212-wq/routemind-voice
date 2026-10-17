# -*- coding: utf-8 -*-
"""The bank's quota-free listener: an Azure speech-to-text READ-BACK (the same free F0 Speech resource as the English
voice — tools/voice_audition/azure.py's key and region). It needs no Gemini quota, so the free Gemini TTS requests are
the bank's only Gemini spend (docs/voice-bank-publish-2026-10.md §3, 2026-10-17).

    python -m tools.voice_bank.stt hear <clip.mp3|wav> [--lang ar|en] [--context]   # one read-back (prints it)
    python -m tools.voice_bank.stt ledger                                          # the audio seconds spent

WHAT IT IS: the "REST API for short audio" (real-time recognition, ≤ 60 s a request; F0: ONE concurrent request, so
every call here is sequential), `format=detailed`. It is BLIND — it is never told the text; the comparison with the
segment is the bank's (build.read_back). English is compared on the LEXICAL form (numbers as the words said: "In 400
meters," is heard "in four hundred meters"); Arabic (ar-EG) on the lexical form through the bank's folds.

THE BUDGET IS AUDIO SECONDS: F0 = 5 audio hours of speech to text a month, billed per second (the pricing page). Every
request is counted before it is sent (whole seconds, rounded UP) in $VOICE_BANK_OUT/stt_ledger.json and refused past
STT_MONTH_CEILING_S. Nothing is asked twice (the cache keys a read-back by the exact 16 kHz audio + locale).
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
CACHE = os.path.join(OUT, "stt_cache")
LEDGER = os.path.join(OUT, "stt_ledger.json")
LOCALES = {"en": "en-US", "ar": "ar-EG"}
MONTHLY_F0_S = 5 * 3600
STT_MONTH_CEILING_S = 4 * 3600 + 30 * 60      # the bank's share; 30 min stay for anything else on the resource
MAX_AUDIO_S = 55.0                            # the short-audio API takes ≤ 60 s
MIN_GAP_S = 0.25


class Exhausted(Exception):
    pass


class SttError(Exception):
    pass


def _ffmpeg():
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def wav16(audio):
    """Any audio (WAV / MP3 bytes) → 16 kHz mono 16-bit WAV, what the short-audio API takes."""
    r = subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-sample_fmt",
                        "s16", "-f", "wav", "-"], input=audio, capture_output=True, check=True)
    return r.stdout


def _seconds(wav):
    # 44-byte header, 32,000 bytes a second
    return max(0.0, (len(wav) - 44) / 32000.0)


class Ledger:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.d = json.load(io.open(LEDGER, encoding="utf-8")) if os.path.exists(LEDGER) else {
            "seconds": 0, "requests": 0, "failed": 0, "by_month": {}, "by_day": {}, "errors": []}

    def month(self):
        return self.d["by_month"].get(datetime.date.today().isoformat()[:7], 0)

    def check(self, s):
        if self.month() + s > STT_MONTH_CEILING_S:
            raise Exhausted("the month's speech-to-text ceiling (%d s of the F0 %d s) would be passed"
                            % (STT_MONTH_CEILING_S, MONTHLY_F0_S))

    def ok(self, s):
        today = datetime.date.today().isoformat()
        self.d["seconds"] += s
        self.d["requests"] += 1
        self.d["by_month"][today[:7]] = self.d["by_month"].get(today[:7], 0) + s
        self.d["by_day"][today] = self.d["by_day"].get(today, 0) + s
        self.save()

    def fail(self, code, msg):
        from tools.voice_audition import azure as AZR
        self.d["failed"] += 1
        self.d["errors"] = (self.d["errors"] + [{"at": datetime.datetime.now().isoformat(timespec="seconds"),
                                                 "code": code, "message": AZR.scrub(msg)[:200]}])[-40:]
        self.save()

    def save(self):
        tmp = LEDGER + ".tmp"
        io.open(tmp, "w", encoding="utf-8").write(json.dumps(self.d, indent=1))
        os.replace(tmp, LEDGER)


_L = {"ledger": None, "last": 0.0}


def ledger():
    if _L["ledger"] is None:
        _L["ledger"] = Ledger()
    return _L["ledger"]


def _post(wav, locale):
    from tools.voice_audition import azure as AZR
    key, region = AZR._load()
    q = urllib.parse.urlencode({"language": locale, "format": "detailed", "profanity": "raw"})
    url = "https://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?%s" % (region, q)
    req = urllib.request.Request(url, data=wav, method="POST", headers={
        "Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Accept": "application/json", "User-Agent": "routemind-voice-bank"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def recognize(audio, lang):
    """One blind read-back → {"status", "lexical", "display", "confidence", "seconds"} (cached; nothing asked twice).
    Exhausted past the month's ceiling; SttError when the service keeps failing."""
    from tools.voice_audition import azure as AZR
    wav = wav16(audio)
    sec = _seconds(wav)
    if sec > MAX_AUDIO_S:
        raise SttError("%.1f s is longer than the short-audio API takes" % sec)
    locale = LOCALES.get(lang, lang)
    k = hashlib.sha256(locale.encode() + b"\n" + wav).hexdigest()
    p = os.path.join(CACHE, k[:2], k + ".json")
    if os.path.exists(p):
        return json.load(io.open(p, encoding="utf-8"))
    billed = int(math.ceil(sec))
    L = ledger()
    L.check(billed)
    last = None
    for attempt in range(6):
        w = _L["last"] + MIN_GAP_S - time.time()
        if w > 0:
            time.sleep(w)
        _L["last"] = time.time()
        try:
            r = _post(wav, locale)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            L.fail(e.code, body or e.reason)
            last = "HTTP %d %s" % (e.code, AZR.scrub(body)[:160])
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else 5.0 * (attempt + 1))
                continue
            if e.code == 403:
                raise Exhausted("azure stt 403: %s" % AZR.scrub(body)[:160])
            if e.code in (401, 500, 502, 503, 504):     # 401 came back once mid-run (2026-10-17), then OK again
                time.sleep(4.0 * (attempt + 1))
                continue
            raise SttError("azure stt %s" % last)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            L.fail(0, str(e))
            last = AZR.scrub(e)[:160]
            time.sleep(4.0 * (attempt + 1))
            continue
        L.ok(billed)
        nb = (r.get("NBest") or [{}])[0]
        out = {"status": r.get("RecognitionStatus"), "lexical": nb.get("Lexical") or "", "itn": nb.get("ITN") or "",
               "display": nb.get("Display") or r.get("DisplayText") or "", "confidence": nb.get("Confidence"),
               "offset_s": round((r.get("Offset") or 0) / 1e7, 2), "duration_s": round((r.get("Duration") or 0) / 1e7, 2),
               "seconds": round(sec, 2), "locale": locale,
               "at": datetime.datetime.now().isoformat(timespec="seconds")}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False))
        return out
    raise SttError("azure stt gave up: %s" % last)


def verdict(audio, lang):
    """The read-back in the judge's verdict shape (build.Build.verdict reads "transcript" and "_judge_model")."""
    r = recognize(audio, lang)
    text = r["lexical"] if lang == "en" else (r["lexical"] or r["display"])
    return {"transcript": text if r["status"] == "Success" else "", "display": r["display"], "itn": r.get("itn", ""),
            "confidence": r["confidence"], "status": r["status"], "_judge_model": "azure-stt/" + r["locale"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hear", "ledger"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--context", action="store_true", help="Arabic: behind the bank's lead clip")
    a = ap.parse_args()
    if a.cmd == "ledger":
        L = ledger()
        print(json.dumps({k: L.d[k] for k in ("seconds", "requests", "failed", "by_month")}, indent=1))
        print("this month: %d of %d s (F0 %d s)" % (L.month(), STT_MONTH_CEILING_S, MONTHLY_F0_S))
        return
    from tools.voice_audition import audition as A
    for f in a.files:
        data = open(f, "rb").read()
        if a.context:
            wav = data if data[:4] == b"RIFF" else A.mp3_to_wav(f)
            data = A.in_context(wav)
        print(os.path.basename(f), json.dumps(recognize(data, a.lang), ensure_ascii=False))


if __name__ == "__main__":
    main()
