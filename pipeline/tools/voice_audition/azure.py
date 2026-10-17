# -*- coding: utf-8 -*-
"""Azure Speech (the FREE F0 resource) — the desk's client for the voice audition and the bank build.

    python -m tools.voice_audition.azure voices [--locale ar-EG]   # the voices this resource lists (no characters)
    python -m tools.voice_audition.azure ledger                    # the characters this desk has spent

THE BUDGET IS CHARACTERS. F0 = 500,000 neural-voice characters a month (the pricing page), 20 requests per 60 s
(docs: speech-services-quotas-and-limits, "20 transactions per 60 seconds", not adjustable). Billable = every
code point of the SSML EXCEPT the <speak> and <voice> tags — so prosody / express-as / break / phoneme / sub markup
COSTS characters (docs: text-to-speech "Billable characters"). Every request this desk makes is counted here before
it is sent (`billable`), recorded in the ledger after it answered, and refused locally once the run's cap
(`--azure-cap`, default AUDITION_CAP) would be passed. The bank shares the month: keep the audition small.

F0 cannot bill: it is a free SKU with a fixed allowance; past it Azure refuses (429 / 403 quota) until the month
resets — only an explicit tier change to S0 could cost money, and nothing here touches the resource.

THE KEY AND REGION live OUTSIDE the repo (C:\\Users\\ibrah\\routemind-secrets\\azure_speech_key.txt / _region.txt,
or $AZURE_SPEECH_KEY_FILE / $AZURE_SPEECH_REGION_FILE). They are read at runtime, sent only as the
Ocp-Apim-Subscription-Key header and the host name, and scrubbed from everything printed or stored (the ledger
and the cache never hold a URL).
"""
import argparse
import datetime
import hashlib
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from xml.sax.saxutils import escape, quoteattr

OUT = os.environ.get("VOICE_AUDITION_OUT", "G:/grad_all/.p5work/voice_audition")
AZ = os.path.join(OUT, "azure")
KEY_FILE = os.environ.get("AZURE_SPEECH_KEY_FILE", r"C:\Users\ibrah\routemind-secrets\azure_speech_key.txt")
REGION_FILE = os.environ.get("AZURE_SPEECH_REGION_FILE", r"C:\Users\ibrah\routemind-secrets\azure_speech_region.txt")

MONTHLY_F0 = 500_000
AUDITION_CAP = 60_000          # this desk's audition + tuning + Sekka tests, all runs (the bank needs the rest)
MIN_GAP_S = 3.2                # 20 requests / 60 s on F0 → one every 3 s (+ a margin)
WAV_FMT = "riff-24khz-16bit-mono-pcm"
MP3_FMT = "audio-24khz-48kbitrate-mono-mp3"   # what Sekka's /speak sends the phones
# The desk asks for 96 kbps MP3 and decodes it: the ~200 KB WAV streams broke mid-transfer on this connection
# (http IncompleteRead on 12 of ~250 requests, 2026-10-17, one line four times in a row); the MP3 is 1/4 the bytes.
HQ_FMT = "audio-24khz-96kbitrate-mono-mp3"


class Exhausted(Exception):
    pass


class AzureError(Exception):
    pass


# ── secrets ─────────────────────────────────────────────────────────────────────────────────────────
_SECRET = {"key": None, "region": None}


def _load():
    if _SECRET["key"] is None:
        with io.open(KEY_FILE, encoding="utf-8") as fh:
            _SECRET["key"] = fh.read().strip()
        with io.open(REGION_FILE, encoding="utf-8") as fh:
            _SECRET["region"] = fh.read().strip()
        if not re.fullmatch(r"[a-z0-9]{3,30}", _SECRET["region"] or ""):
            raise AzureError("the region file does not hold a region name")
    return _SECRET["key"], _SECRET["region"]


def scrub(s):
    s = str(s)
    for v in (_SECRET["key"], _SECRET["region"]):
        if v:
            s = s.replace(v, "<secret>")
    return re.sub(r"(?<![0-9A-Za-z])[0-9A-Za-z]{80,}(?![0-9A-Za-z])", "<secret>", s)   # the 84-character key format


def log(*a):
    print(scrub(" ".join(str(x) for x in a)), flush=True)


# ── SSML ────────────────────────────────────────────────────────────────────────────────────────────
NS = 'xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts"'


def ssml(text, voice, lang, rate=None, pitch=None, volume=None, contour=None, style=None, degree=None,
         body_xml=None, lead_ms=None):
    """ONE compact SSML document. `text` is escaped; `body_xml` (already SSML: breaks, sub, phoneme …) replaces it.
    Every attribute is billable, so only the ones given are written — and ONE prosody wrapper for the whole text."""
    inner = body_xml if body_xml is not None else escape(text)
    pros = []
    if rate:
        pros.append("rate=%s" % quoteattr(rate))
    if pitch:
        pros.append("pitch=%s" % quoteattr(pitch))
    if volume:
        pros.append("volume=%s" % quoteattr(volume))
    if contour:
        pros.append("contour=%s" % quoteattr(contour))
    if pros:
        inner = "<prosody %s>%s</prosody>" % (" ".join(pros), inner)
    if style:
        inner = "<mstts:express-as style=%s%s>%s</mstts:express-as>" % (
            quoteattr(style), (" styledegree=%s" % quoteattr(str(degree))) if degree else "", inner)
    if lead_ms is not None:
        inner = '<mstts:silence type="Leading-exact" value="%dms"/>' % lead_ms + inner
    loc = lang if "-" in lang else {"ar": "ar-EG", "en": "en-US"}[lang]
    return '<speak version="1.0" %s xml:lang="%s"><voice name=%s>%s</voice></speak>' % (NS, loc, quoteattr(voice),
                                                                                       inner)


_FREE_TAG = re.compile(r"</?speak\b[^>]*>|</?voice\b[^>]*>")


def billable(doc):
    """Azure's billable characters: every code point except the <speak> and <voice> tags."""
    return len(_FREE_TAG.sub("", doc))


# ── the ledger ──────────────────────────────────────────────────────────────────────────────────────
class Ledger:
    def __init__(self):
        os.makedirs(AZ, exist_ok=True)
        self.path = os.path.join(AZ, "ledger.json")
        self.d = json.load(io.open(self.path, encoding="utf-8")) if os.path.exists(self.path) else {
            "chars": 0, "requests": 0, "failed": 0, "by_day": {}, "by_purpose": {}, "by_month": {}, "errors": []}
        self.cap = AUDITION_CAP

    def save(self):
        io.open(self.path, "w", encoding="utf-8").write(scrub(json.dumps(self.d, ensure_ascii=False, indent=1)))

    def spent(self, purpose=None):
        return self.d["by_purpose"].get(purpose, 0) if purpose else self.d["chars"]

    def check(self, n, purpose):
        if purpose != "bank" and self.d["chars"] - self.d["by_purpose"].get("bank", 0) + n > self.cap:
            raise Exhausted("this desk's Azure audition cap (%d characters) would be passed" % self.cap)

    def ok(self, n, purpose, lat):
        today = datetime.date.today().isoformat()
        month = today[:7]
        self.d["chars"] += n
        self.d["requests"] += 1
        self.d["by_day"][today] = self.d["by_day"].get(today, 0) + n
        self.d["by_month"][month] = self.d["by_month"].get(month, 0) + n
        self.d["by_purpose"][purpose] = self.d["by_purpose"].get(purpose, 0) + n
        self.d.setdefault("lat", []).append(round(lat, 2))
        self.d["lat"] = self.d["lat"][-200:]
        self.save()

    def fail(self, code, msg):
        self.d["failed"] += 1
        self.d["errors"].append({"at": datetime.datetime.now().isoformat(timespec="seconds"), "code": code,
                                 "message": scrub(msg)[:240]})
        self.d["errors"] = self.d["errors"][-40:]
        self.save()


LEDGER = None
_LAST = {"t": 0.0}


def ledger():
    global LEDGER
    if LEDGER is None:
        LEDGER = Ledger()
    return LEDGER


def _post(doc, fmt, timeout=60):
    key, region = _load()
    req = urllib.request.Request("https://%s.tts.speech.microsoft.com/cognitiveservices/v1" % region,
                                 data=doc.encode("utf-8"), method="POST", headers={
                                     "Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                                     "X-Microsoft-OutputFormat": fmt, "User-Agent": "routemind-voice-audition"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def synth(doc, fmt=WAV_FMT, purpose="audition", cache=True):
    """(audio bytes, meta). Cached by the exact SSML + format: nothing is asked twice."""
    k = hashlib.sha256(("%s\n%s" % (fmt, doc)).encode("utf-8")).hexdigest()
    ext = "wav" if fmt.startswith("riff") else "mp3"
    ap = os.path.join(AZ, "cache", "%s.%s" % (k, ext))
    mp = os.path.join(AZ, "cache", "%s.json" % k)
    if cache and os.path.exists(ap) and os.path.exists(mp):
        return open(ap, "rb").read(), json.load(io.open(mp, encoding="utf-8"))
    L = ledger()
    n = billable(doc)
    L.check(n, purpose)
    for attempt in range(4):
        w = _LAST["t"] + MIN_GAP_S - time.time()
        if w > 0:
            time.sleep(w)
        _LAST["t"] = time.time()
        t0 = time.time()
        try:
            audio = _post(doc, fmt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            L.fail(e.code, body or e.reason)
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else 10.0 * (attempt + 1)
                log("  azure 429 — waiting %.0f s" % wait)
                time.sleep(wait)
                continue
            if e.code == 403:
                raise Exhausted("azure 403: %s" % scrub(body)[:160])
            if e.code in (500, 502, 503, 504):
                time.sleep(5.0 * (attempt + 1))
                continue
            raise AzureError("azure HTTP %d: %s" % (e.code, scrub(body)[:200]))
        except http.client.IncompleteRead as e:
            # the service synthesised (and counted) the text; the stream broke on the way — count it, ask again
            L.ok(n, purpose + "-lost", time.time() - t0)
            L.fail(0, "incomplete read (%d bytes)" % len(e.partial or b""))
            log("  azure: the audio stream broke after %d bytes — asking again" % len(e.partial or b""))
            time.sleep(3.0 * (attempt + 1))
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            L.fail(0, str(e))
            time.sleep(5.0 * (attempt + 1))
            continue
        lat = time.time() - t0
        if not audio or len(audio) < 200:
            L.fail(200, "empty audio (%d bytes)" % len(audio or b""))
            raise AzureError("azure: empty audio")
        L.ok(n, purpose, lat)
        meta = {"chars": n, "latency_s": round(lat, 2), "bytes": len(audio), "fmt": fmt,
                "at": datetime.datetime.now().isoformat(timespec="seconds")}
        if cache:
            os.makedirs(os.path.dirname(ap), exist_ok=True)
            open(ap, "wb").write(audio)
            io.open(mp, "w", encoding="utf-8").write(json.dumps(meta))
        return audio, meta
    raise AzureError("azure: gave up after retries")


def synth_wav(doc, purpose="audition"):
    """synth() through HQ_FMT, decoded to 24 kHz mono 16-bit WAV (ffmpeg) — what the audition and the bank process."""
    import shutil
    import subprocess
    mp3, meta = synth(doc, fmt=HQ_FMT, purpose=purpose)
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    r = subprocess.run([ff, "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "24000", "-f", "wav", "-"],
                       input=mp3, capture_output=True, check=True)
    return r.stdout, meta


def voices(refresh=False):
    p = os.path.join(AZ, "voices.json")
    if os.path.exists(p) and not refresh:
        return json.load(io.open(p, encoding="utf-8"))
    key, region = _load()
    req = urllib.request.Request("https://%s.tts.speech.microsoft.com/cognitiveservices/voices/list" % region,
                                 headers={"Ocp-Apim-Subscription-Key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    os.makedirs(AZ, exist_ok=True)
    io.open(p, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=1))
    return data


# ── Sekka's runtime respellings (supabase/functions/api/copilot/azure_tts.ts RESPELL_AR) ──────────────────────────
# The navigation bank's SAY spellings (tools/masri_voice/pronounce.py, first spelling) + the Azure-only fixes the
# audition found (audition.AZ_SAY), as whole words, for the model's FREE text. Left out: words whose meaning in free
# text is not always the Egyptian one the spelling assumes («قد» is also MSA "may"; «ثانية» also the ordinal).
RUNTIME_EXCLUDE = {"قد", "ثانية", "قرب"}   # «قرب» without its shadda is also «قُرب» (nearness)
_HARAKAT = re.compile("[ً-ْ]")


def runtime_respell():
    from tools.masri_voice import pronounce
    from tools.voice_audition import audition as A
    out = {}
    for k, vs in pronounce.SAY.items():
        if k in RUNTIME_EXCLUDE:
            continue
        for form in (k, _HARAKAT.sub("", k)):
            if form not in RUNTIME_EXCLUDE:
                out.setdefault(form, vs[0])
    for k, v in A.AZ_SAY.items():
        if "<" in v:                       # markup (a phoneme) costs characters: the bank only, never Sekka
            continue
        for form in (k, _HARAKAT.sub("", k)):
            out[form] = v
    return {k: v for k, v in sorted(out.items()) if k != v}


def respell_ts():
    """The TypeScript literal of RESPELL_AR (paste into azure_tts.ts; tests/azure_tts.test.ts checks a sample)."""
    rows = ["  %s: %s," % (json.dumps(k, ensure_ascii=False), json.dumps(v, ensure_ascii=False))
            for k, v in runtime_respell().items()]
    return "export const RESPELL_AR: Readonly<Record<string, string>> = {\n" + "\n".join(rows) + "\n};"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["voices", "ledger", "respell-ts"])
    ap.add_argument("--locale", default=None)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.cmd == "respell-ts":
        print(respell_ts())
        return
    if a.cmd == "voices":
        vs = voices(a.refresh)
        for v in vs:
            if a.locale and not v["Locale"].lower().startswith(a.locale.lower()):
                continue
            print(v["ShortName"], v["Gender"], v["VoiceType"], v.get("Status"), ",".join(v.get("StyleList") or []))
        print(len(vs), "voices")
    else:
        L = ledger()
        print(json.dumps({k: L.d[k] for k in ("chars", "requests", "failed", "by_day", "by_month", "by_purpose")},
                         indent=1))
        print("F0 month: %d of %d characters (%.1f %%)" % (L.d["chars"], MONTHLY_F0, 100.0 * L.d["chars"] / MONTHLY_F0))


if __name__ == "__main__":
    main()
