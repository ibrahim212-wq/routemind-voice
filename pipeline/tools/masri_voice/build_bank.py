# -*- coding: utf-8 -*-
"""Build the Egyptian clip bank the app ships — assets/voice/ar/<fnv64(segment)>.mp3 (the app bundles only
the clips; the manifest lives beside this builder, tools/masri_voice/bank_manifest.json).

  python -m tools.masri_voice.build_bank            # render every missing / stale segment
  python -m tools.masri_voice.build_bank --check    # exit 1 when a segment has no current clip
  python -m tools.masri_voice.build_bank --redo-qa  # re-render the clips the ASR check flagged
  python -m tools.masri_voice.build_bank --context [--all]  # judge flagged [every direction-word] clip INSIDE a line
  python -m tools.masri_voice.build_bank --report   # sizes, CER, every flag and its in-line verdict
  python -m tools.masri_voice.build_bank --pron     # the blind pronunciation listeners over the bank as it is

One call per segment of tools/masri_voice/inventory.all_segments() (every line MasriVoice can
compose is a join of these), rendered ONCE here by the chosen voice, so the phone never synthesises
Arabic guidance: no network at a maneuver, no latency, no per-trip cost.

THE PIPELINE, per segment
  1. Gemini-TTS (ar-EG, the Egyptian style prompt, -6 dB so nothing clips) → 24 kHz LINEAR16.
  2. Trim the lead / tail silence to 25 / 40 ms, 5 ms fade in, 15 ms fade out — the pauses between
     segments are set by the composer (MasriVoice.pauseAfter), not by whatever the engine left.
  3. Loudness: the speech's RMS to -20 dBFS with the peak held under -1.5 dBFS, so consecutive
     segments of one line — separate syntheses — sound like one speaker at one level.
  4. QA: speech rate within bounds (a stall or a hallucinated sentence is re-rendered) and an ASR
     read-back (Gemini on Vertex) — a clip whose transcript loses or swaps a DIRECTION word
     (يمين / شمال / على طول / ارجع) or an exit ordinal is re-rendered; after 3 takes the best is
     kept and named in the report. A wrong direction word is a safety defect, not a style one.
  5. Windows Media Foundation MP3 (MPEG-2 Layer III, 24 kHz mono, 32 kbps CBR — the rate Google serves its own MP3 TTS at), stripped to bare
     frames (no ID3 / Xing — tools/masri_voice/mp3.py explains why).
The manifest records, per clip, the exact text, the voice id and the QA verdict;
tool/masri_voice_parity.py fails when a segment's clip is missing or was rendered from other text.
"""
import concurrent.futures
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from tools.masri_voice import acoustics
from tools.masri_voice import inventory
from tools.masri_voice import masri_voice as mv
from tools.masri_voice import mp3
from tools.masri_voice import tts
from tools.masri_voice import pronounce

ROOT = mv.ROOT
BANK = os.path.join(ROOT, "assets", "voice", "ar")
# build metadata only (text, voice, QA verdicts) — NOT in assets/, so it never ships inside the app
MANIFEST = os.path.join(ROOT, "tools", "masri_voice", "bank_manifest.json")
WORK = os.environ.get("MASRI_SCRATCH") or os.path.join(ROOT, "build", "masri_voice_scratch")
PS = os.path.join(ROOT, "tools", "masri_voice", "transcode_batch.ps1")

VOICE = {
    "engine": "gemini-tts",
    "model": "gemini-2.5-pro-tts",
    "name": "Zephyr",
    "lang": "ar-EG",
    "gain_db": -6.0,
    "rate_hz": 24000,
    "kbps": 32,
}
TARGET_RMS_DB = -20.0
PEAK_CEIL_DB = -1.5
LEAD_S, TAIL_S = 0.025, 0.040
FADE_IN_S, FADE_OUT_S = 0.005, 0.015
MAX_TAKES = 3
QA_MODEL = "gemini-2.5-flash"
# A read-back Flash flags (a lost direction word, CER > CER_RECHECK) gets a second listen by Pro; the
# clip is judged by the more careful listener. On 1-second clips Flash alone heard «بعد مية متر» as
# «بتاع مدينتي» — the check must fail the VOICE, not the transcriber.
QA_MODEL_STRICT = "gemini-2.5-pro"
CER_RECHECK = 0.30

# «رجع», not «ارجع»: «وارجعي» is spoken — and transcribed — «ورجعي» (the alif elides)
DIRECTION = ["يمين", "شمال", "على طول", "رجع", "ورا", "ملف"]   # ملف: the U-turn (since 2026-09-24)
ORDINAL = ["أول", "تاني", "تالت", "رابع", "خامس", "سادس", "سابع", "تامن"]


def voice_id():
    return "%s/%s/%s/%s/g%+d/p%s" % (VOICE["engine"], VOICE["model"], VOICE["name"], VOICE["lang"],
                                    int(VOICE["gain_db"]), mv.fnv64(tts.EGY_STYLE_PROMPT)[:8])


# ── text normalisation for the ASR read-back ────────────────────────────────────────────────────
_DIAC = re.compile("[%s-%s%s%s]" % (chr(0x64B), chr(0x652), chr(0x670), chr(0x640)))   # harakat, dagger alef, tatweel


def norm(s):
    s = _DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    # the SAME spoken word in Egyptian and in MSA spelling (a transcriber writes «ثالث» / «اثنين» for
    # the «تالت» / «اتنين» a Cairene says): one letter for the comparison
    s = s.replace("ث", "ت").replace("ذ", "د").replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    return s


def letters(s):
    return "".join(c for c in norm(s) if c.isalpha())


def cer(a, b):
    a, b = letters(a), letters(b)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / float(len(a))


def critical_ok(text, heard):
    """Every direction word / ordinal the segment says must be heard, and no OTHER direction word may
    be heard instead (يمين said as شمال is the defect this exists for)."""
    # spaces out: "يو تيرن" / "يوتيرن", "على طول" / "علطول" are the same words to a listener
    t, h = norm(text).replace(" ", ""), norm(heard).replace(" ", "")
    problems = []
    for w in DIRECTION + ORDINAL:
        nw = norm(w).replace(" ", "")
        if nw in t and nw not in h:
            problems.append("lost " + w)
    for a, b in (("يمين", "شمال"), ("شمال", "يمين")):
        if norm(a) in t and norm(b) not in t and norm(b) in h:
            problems.append("swapped %s→%s" % (a, b))
    return problems


ASR_PROMPT = ("Transcribe exactly what is said, in Arabic script, as Egyptian colloquial Arabic is "
              "written (write numbers as the words spoken, never digits). Output only the transcript.")


# Gemini-TTS Pro has a small per-minute quota on this project (a burst of three drew a 429): one
# shared pacer keeps every worker under it instead of every worker sleeping out its own 429.
TTS_MIN_GAP_S = float(os.environ.get("MASRI_TTS_GAP_S", "6.5"))
_tts_lock = threading.Lock()
_tts_state = {"last": 0.0}


def paced_tts(text):
    with _tts_lock:
        wait = _tts_state["last"] + TTS_MIN_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _tts_state["last"] = time.time()
    return tts.gemini_tts(text, voice=VOICE["name"], model=VOICE["model"], prompt=tts.EGY_STYLE_PROMPT,
                          encoding="LINEAR16", lang=VOICE["lang"], gain_db=VOICE["gain_db"])


# Vertex quotas are per region per base model per minute: the read-back rotates regions and is
# paced by one shared limiter (the first build hit the global per-minute quota at four threads).
ASR_REGIONS = ["global", "us-central1", "europe-west4", "us-east4", "europe-west1", "asia-northeast1"]
_asr_lock = threading.Lock()
_asr_state = {"i": 0, "last": 0.0}
ASR_MIN_GAP_S = 0.7


def asr(wav_bytes, model=QA_MODEL, mime="audio/wav"):
    last_err = None
    for attempt in range(len(ASR_REGIONS) * 2):
        with _asr_lock:
            wait = _asr_state["last"] + ASR_MIN_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            _asr_state["last"] = time.time()
            loc = ASR_REGIONS[_asr_state["i"] % len(ASR_REGIONS)]
            _asr_state["i"] += 1
        try:
            raw = tts.vertex_generate(model, [tts.audio_part(wav_bytes, mime), {"text": ASR_PROMPT}],
                                      json_out=False, location=loc)
            return raw.strip()
        except Exception as e:
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(str(last_err))


# ── audio ───────────────────────────────────────────────────────────────────────────────────────
def process(x, rate):
    a, b = acoustics.speech_bounds(x, rate, thresh_db=-45.0)
    a = max(0, a - int(LEAD_S * rate))
    b = min(len(x), b + int(TAIL_S * rate))
    y = x[a:b].astype(np.float64)
    fi, fo = int(FADE_IN_S * rate), int(FADE_OUT_S * rate)
    if len(y) > fi + fo:
        y[:fi] *= np.linspace(0.0, 1.0, fi)
        y[-fo:] *= np.linspace(1.0, 0.0, fo)
    # loudness on the voiced part (frames above -35 dB of the peak frame)
    f = acoustics.frames(y.astype(np.float32), rate, 0.02, 0.01)
    if len(f):
        r = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
        voiced = r[r > r.max() * 10 ** (-35 / 20.0)]
        rms = float(np.sqrt((voiced ** 2).mean())) if len(voiced) else float(np.sqrt((y ** 2).mean()))
    else:
        rms = float(np.sqrt((y ** 2).mean() + 1e-12))
    g = 10 ** (TARGET_RMS_DB / 20.0) / max(rms, 1e-9)
    peak = float(np.abs(y).max()) * g
    ceil = 10 ** (PEAK_CEIL_DB / 20.0)
    if peak > ceil:
        g *= ceil / peak
    return np.clip(y * g, -1.0, 1.0).astype(np.float32)


def wav_bytes(y, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((y * 32767.0).astype(np.int16).tobytes())
    return buf.getvalue()


_phones_lock = threading.Lock()


def pron_of(text, wb, heard_texts):
    """The pronunciation verdict of a take: the BLIND phone listener (phones.py — ق as q, ج as dʒ,
    ث/ذ as θ/ð, the listed Egyptian vowels) and the word listener (every word of the segment in what
    the transcriber heard; a respelled word must still be heard AS that word)."""
    from tools.masri_voice import phones
    x, rate = acoustics.pcm(wb)
    with _phones_lock:
        ipa = phones.ipa(x, rate)
    hard, soft = pronounce.check(text, ipa)
    miss = None
    for h in heard_texts:
        if h and not h.startswith("<"):
            m = pronounce.missing_words(text, h)
            if miss is None or len(m) < len(miss):
                miss = m
    miss = miss or []
    # the transcriber repairs words into the ones it expects, so a missing word only RANKS takes
    soft += ["«%s» not in the transcript" % w for w in miss]
    return {"ipa": ipa, "hard": hard, "soft": soft}


def qa_of(text, wb, take):
    rate = 24000
    with wave.open(io.BytesIO(wb)) as w:
        rate = w.getframerate()
        dur = w.getnframes() / float(rate)
    n_letters = max(1, len(letters(text)))
    s_per_letter = dur / n_letters
    def listen(model):
        try:
            h = asr(wb, model)
        except Exception as e:  # the check is a gate, not a dependency: record and keep going
            h = "<asr failed: %s>" % str(e)[:80]
        failed = h.startswith("<asr")
        return h, (cer(text, h) if not failed else 1.0), (critical_ok(text, h) if not failed else ["asr unavailable"])
    heard, c, crit = listen(QA_MODEL)
    by = QA_MODEL
    if crit or c > CER_RECHECK:
        h2, c2, crit2 = listen(QA_MODEL_STRICT)
        if (len(crit2), c2) < (len(crit), c):
            heard, c, crit, by = h2, c2, crit2, QA_MODEL_STRICT
    pace_ok = 0.035 <= s_per_letter <= 0.20
    return {"take": take, "dur_s": round(dur, 3), "s_per_letter": round(s_per_letter, 3),
            "cer": round(c, 3), "heard": heard, "critical": crit, "pace_ok": pace_ok, "by": by}


def clean(qa):
    """A clip that carries a direction word / ordinal is judged by what is heard INSIDE a line (a
    1-second clip alone fools both transcribers); every other clip by its own read-back."""
    if (qa.get("pron") or {}).get("hard"):
        return False
    cx = qa.get("context")
    if cx is not None:
        return qa["pace_ok"] and not inline_problems(qa) and qa["cer"] <= CER_MAX_IN_LINE
    return not qa["critical"] and qa["pace_ok"] and qa["cer"] <= 0.35


def inline_problems(qa):
    """The in-line (lead + clip) verdict, with the transcriber's blind spots overruled by the phones:
    a "lost ملف" / "lost تامن" is not a problem when the phone listener heard that very word."""
    cx = qa.get("context") or {}
    ipa = (qa.get("pron") or {}).get("ipa")
    return [p for p in (cx.get("critical") or [])
            if not (p.startswith("lost ") and pronounce.heard_by_phones(p[5:], ipa))]


def has_critical(text):
    t = norm(text).replace(" ", "")
    return any(norm(w).replace(" ", "") in t for w in DIRECTION + ORDINAL)


def score(qa):
    cx = qa.get("context")
    crit = (20 if inline_problems(qa) else 0) if cx is not None else (10 if qa["critical"] else 0)
    pr = qa.get("pron") or {}
    return (crit + (0 if qa["pace_ok"] else 5) + qa["cer"]
            + 15 * len(pr.get("hard") or []) + 2 * len(pr.get("soft") or []))


def render_one(text, first=None, first_say=None):
    """→ (processed WAV bytes, qa dict). Up to MAX_TAKES takes; the first clean one wins, else the best.
    `first` = a take already on disk (a resumed build) — judged before any new synthesis."""
    best = None
    take = 0

    def judge(wb, n, said):
        qa = qa_of(text, wb, n)
        qa["say"] = said
        if has_critical(text):
            qa["context"] = context_wav(text, wb)
        qa["pron"] = pron_of(text, wb, [qa.get("heard"), (qa.get("context") or {}).get("heard")])
        return qa
    if first is not None:
        take = 1
        qa = judge(first, 1, first_say or text)
        best = (score(qa), first, qa)
        if clean(qa):
            return first, qa
    limit = MAX_TAKES + (CONTEXT_EXTRA_TAKES if (has_critical(text) or pronounce.targeted(text)) else 0)
    nv = pronounce.variants(text)
    while take < limit:
        take += 1
        said = pronounce.say(text, (take - 1) % nv)
        raw = paced_tts(said)
        x, rate = acoustics.pcm(raw)
        wb = wav_bytes(process(x, rate), rate)
        qa = judge(wb, take, said)
        if best is None or score(qa) < best[0]:
            best = (score(qa), wb, qa)
        if clean(qa):
            break
    return best[1], best[2]


# ── the build ───────────────────────────────────────────────────────────────────────────────────
def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(io.open(MANIFEST, encoding="utf-8"))
    return {"voice": {}, "clips": {}}


def save_manifest(m):
    os.makedirs(BANK, exist_ok=True)
    m["voice"] = dict(VOICE, id=voice_id(), prompt=tts.EGY_STYLE_PROMPT)
    m["_readme"] = ("The Egyptian navigation clip bank — generated by tools/masri_voice/build_bank.py; "
                    "never edit by hand. Key = fnv64(segment text) = the clip's file name.")
    io.open(MANIFEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def build(redo_qa=False, workers=4, limit=None):
    segs = sorted(inventory.all_segments())
    man = load_manifest()
    vid = voice_id()
    clips = man.setdefault("clips", {})
    todo = []
    for s in segs:
        k = mv.fnv64(s)
        c = clips.get(k)
        stale = (c is None or c.get("text") != s or c.get("voice") != vid or
                 c.get("say", s) not in pronounce.all_says(s) or
                 not os.path.exists(os.path.join(BANK, k + ".mp3")))
        flagged = c is not None and not clean(c.get("qa") or {"critical": ["?"], "pace_ok": False, "cer": 1.0})
        if stale or (redo_qa and flagged):
            todo.append(s)
    force = {mv.fnv64(s) for s in todo}
    if limit:
        todo = todo[:limit]
    print("segments %d, to render %d (voice %s)" % (len(segs), len(todo), vid), flush=True)
    wav_dir = os.path.join(WORK, "bank_wav")
    mp3_dir = os.path.join(WORK, "bank_mp3")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mp3_dir, exist_ok=True)
    lock = threading.Lock()
    done = [0]
    results = {}

    def work(s):
        k = mv.fnv64(s)
        wav_p = os.path.join(wav_dir, k + ".wav")
        side_p = os.path.join(wav_dir, k + ".json")
        # RESUME: a take judged by an earlier run of THIS voice is reused as it is; a take rendered
        # but never judged (the run died) is judged before anything is synthesised again.
        if os.path.exists(side_p) and os.path.exists(wav_p):
            try:
                side = json.load(io.open(side_p, encoding="utf-8"))
                said_ok = side.get("qa", {}).get("say", s) in pronounce.all_says(s)
                if (side.get("text") == s and side.get("voice") == vid and said_ok
                        and not (redo_qa and (k in force or not clean(side["qa"])))):
                    with lock:
                        results[k] = (s, side["qa"])
                        done[0] += 1
                    return
            except ValueError:
                pass
        first = open(wav_p, "rb").read() if os.path.exists(wav_p) else None
        first_say = None
        if first is not None and os.path.exists(side_p):
            try:
                first_say = json.load(io.open(side_p, encoding="utf-8")).get("qa", {}).get("say", s)
            except ValueError:
                pass
        if first is not None and (first_say or s) not in pronounce.all_says(s):
            first = None                                   # spoken from a spelling the lexicon replaced
        wb, qa = render_one(s, first, first_say)
        open(wav_p, "wb").write(wb)
        io.open(side_p, "w", encoding="utf-8").write(json.dumps({"text": s, "voice": vid, "qa": qa}, ensure_ascii=False))
        with lock:
            results[k] = (s, qa)
            done[0] += 1
            if done[0] % 25 == 0 or qa["critical"]:
                print("  %d/%d  %s  cer=%.2f %s" % (done[0], len(todo), s, qa["cer"],
                                                   ("CRITICAL " + ",".join(qa["critical"])) if qa["critical"] else ""),
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for f in concurrent.futures.as_completed([ex.submit(work, s) for s in todo]):
            try:
                f.result()
            except Exception as e:
                print("  FAILED:", str(e)[:200], flush=True)
    if not results:
        return man
    # encode in one PowerShell session
    lst = os.path.join(WORK, "bank_list.txt")
    with io.open(lst, "w", encoding="utf-8") as fh:
        for k in results:
            fh.write("%s|%s\n" % (os.path.normpath(os.path.join(wav_dir, k + ".wav")),
                                   os.path.normpath(os.path.join(mp3_dir, k + ".mp3"))))
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS,
                          "-List", os.path.normpath(lst), "-Rate", str(VOICE["rate_hz"]), "-Kbps", str(VOICE["kbps"])],
                         capture_output=True, text=True)
    print(out.stdout.strip()[-400:], out.stderr.strip()[-400:], flush=True)
    os.makedirs(BANK, exist_ok=True)
    for k, (s, qa) in results.items():
        src = os.path.join(mp3_dir, k + ".mp3")
        if not os.path.exists(src):
            print("  no mp3 for", s)
            continue
        bare = mp3.strip(open(src, "rb").read())
        open(os.path.join(BANK, k + ".mp3"), "wb").write(bare)
        clips[k] = {"text": s, "voice": vid, "say": qa.get("say") or s,
                    "ms": int(round(mp3.duration_s(bare) * 1000)), "bytes": len(bare),
                    "qa": {x: qa.get(x) for x in ("take", "cer", "heard", "critical", "pace_ok", "s_per_letter", "by",
                                                  "context", "say", "pron")}}
    # orphans: clips no segment reaches any more
    live = {mv.fnv64(s) for s in segs}
    for k in list(clips):
        if k not in live:
            del clips[k]
            try:
                os.remove(os.path.join(BANK, k + ".mp3"))
            except OSError:
                pass
    save_manifest(man)
    return man


# ── the context check ───────────────────────────────────────────────────────────────────────────
CONTEXT_EXTRA_TAKES = 3   # a clip that carries a direction word gets this many more takes to pass in a line
CER_MAX_IN_LINE = 0.6     # its isolated read-back may be rough (1-second clips); the line decides


def context_wav(text, wb):
    """The take joined behind the lead's recorded take (the pause VoiceBank puts there) and transcribed:
    Flash, then Pro when Flash misses a word. → {lead, heard, critical, by}."""
    lead_p = os.path.join(WORK, "bank_wav", mv.fnv64(CONTEXT_LEAD) + ".wav")
    x_lead, r1 = acoustics.pcm(open(lead_p, "rb").read())
    x_take, r2 = acoustics.pcm(wb)
    if r1 != r2:
        return {"lead": CONTEXT_LEAD, "heard": "<rate mismatch>", "critical": [], "by": ""}
    gap = np.zeros(int(r1 * mv.pause_after(CONTEXT_LEAD) / 1000.0), dtype=np.float32)
    audio = wav_bytes(np.concatenate([np.asarray(x_lead, dtype=np.float32), gap,
                                      np.asarray(x_take, dtype=np.float32)]), r1)
    best = None
    for model in (QA_MODEL, QA_MODEL_STRICT):
        try:
            heard = asr(audio, model)
        except Exception as e:
            heard = "<asr failed: %s>" % str(e)[:60]
        crit = critical_ok(text, heard) if not heard.startswith("<asr") else ["asr unavailable"]
        if best is None or len(crit) < len(best[1]):
            best = (heard, crit, model)
        if not crit:
            break
    return {"lead": CONTEXT_LEAD, "heard": best[0], "critical": best[1], "by": best[2]}


# A driver never hears a 1-second clip alone: «ميّلي يمين،» by itself came back as «ماي ليمون» from
# both transcribers. A clip flagged in isolation is judged again INSIDE a line — joined behind a real
# lead exactly as VoiceBank joins clips — and its direction word / ordinal must survive THERE.
CONTEXT_LEAD = "بعد تلتمية متر،"


def context_check(all_=False):
    man = load_manifest()
    clips = man.get("clips", {})
    lead = open(os.path.join(BANK, mv.fnv64(CONTEXT_LEAD) + ".mp3"), "rb").read()
    n = ok = 0
    for k, c in sorted(clips.items(), key=lambda kv: kv[1]["text"]):
        qa = c.get("qa") or {}
        text = c["text"]
        if not (qa.get("critical") or (all_ and has_critical(text))):
            continue
        body = open(os.path.join(BANK, k + ".mp3"), "rb").read()
        audio = mp3.concat([lead, body], gaps=[mv.pause_after(CONTEXT_LEAD)])
        best = None
        for model in (QA_MODEL, QA_MODEL_STRICT):
            heard = asr(audio, model, "audio/mpeg")
            crit = critical_ok(text, heard)
            if best is None or len(crit) < len(best[1]):
                best = (heard, crit, model)
            if not crit:
                break
        qa["context"] = {"lead": CONTEXT_LEAD, "heard": best[0], "critical": best[1], "by": best[2]}
        left = inline_problems(qa)
        n += 1
        ok += not left
        note = " (the phones heard %s)" % ", ".join(p[5:] for p in best[1]) if best[1] and not left else ""
        print("  %s  %s | %s %s%s" % ("OK  " if not left else "FAIL", text, best[0], left or "", note), flush=True)
    save_manifest(man)
    print("context check: %d clips judged inside a line (as shipped), %d pass, %d fail" % (n, ok, n - ok))
    return n - ok


def pron_audit(only=None):
    """The blind listeners over the bank as it is (the recorded take of every clip): writes qa.pron
    into the manifest, so --redo-qa re-records exactly the clips that fail. → number failing."""
    man = load_manifest()
    clips = man.get("clips", {})
    n = bad = 0
    for k, c in sorted(clips.items(), key=lambda kv: kv[1]["text"]):
        text = c["text"]
        if only and text not in only:
            continue
        wav_p = os.path.join(WORK, "bank_wav", k + ".wav")
        if not os.path.exists(wav_p):
            continue
        wb = open(wav_p, "rb").read()
        qa = c.setdefault("qa", {})
        heard = [qa.get("heard"), (qa.get("context") or {}).get("heard")]
        if pronounce.targeted(text) and not (qa.get("context") or {}).get("heard"):
            cx = context_wav(text, wb)
            if not has_critical(text):
                cx["critical"] = []
            qa["context"] = cx
            heard.append(cx.get("heard"))
        qa["pron"] = pron_of(text, wb, heard)
        n += 1
        if qa["pron"]["hard"]:
            bad += 1
            print("  FAIL  %s | %s | %s" % (text, " ; ".join(qa["pron"]["hard"]), qa["pron"]["ipa"]), flush=True)
    save_manifest(man)
    print("pronunciation: %d clips judged, %d fail a hard target" % (n, bad))
    return bad


def check():
    man = load_manifest()
    vid = voice_id()
    bad = []
    for s in inventory.all_segments():
        k = mv.fnv64(s)
        c = man.get("clips", {}).get(k)
        if c is None or c.get("text") != s or c.get("voice") != vid or not os.path.exists(os.path.join(BANK, k + ".mp3")):
            bad.append(s)
    print("%d segments without a current clip" % len(bad))
    for s in bad[:20]:
        print("  ", s)
    return not bad


def report():
    man = load_manifest()
    clips = man.get("clips", {})
    crit = [(c["text"], c["qa"]) for c in clips.values() if c.get("qa", {}).get("critical")]
    ctx_fail = [c["text"] for c in clips.values() if (c.get("qa") or {}).get("context") is not None
                and inline_problems(c["qa"])]
    ctx_n = sum(1 for c in clips.values() if (c.get("qa") or {}).get("context") is not None)
    pace = [(c["text"], c["qa"]) for c in clips.values() if not c.get("qa", {}).get("pace_ok", True)]
    cers = sorted(c["qa"]["cer"] for c in clips.values() if "qa" in c)
    total_b = sum(c.get("bytes", 0) for c in clips.values())
    total_ms = sum(c.get("ms", 0) for c in clips.values())
    print("clips %d  audio %.1f min  %.2f MB" % (len(clips), total_ms / 60000.0, total_b / 1e6))
    if cers:
        print("ASR CER median %.3f  p90 %.3f  max %.3f" % (cers[len(cers) // 2], cers[int(len(cers) * 0.9)], cers[-1]))
    print("direction words / ordinals: %d clips judged inside a line, %d fail there: %s"
          % (ctx_n, len(ctx_fail), ", ".join("«%s»" % t for t in ctx_fail) or "none"))
    print("isolation read-back flags %d (1-second clips; the line decides), pace flags %d" % (len(crit), len(pace)))
    for t, q in crit[:30]:
        cx = q.get("context")
        print("  CRIT", t, "|", q["heard"], q["critical"],
              "| in a line: " + ("unchecked" if not cx else ("OK «%s»" % cx["heard"] if not cx["critical"] else "FAIL «%s»" % cx["heard"])))
    for t, q in pace[:10]:
        print("  PACE", t, q["s_per_letter"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if "--check" in a:
        sys.exit(0 if check() else 1)
    if "--report" in a:
        report()
        sys.exit(0)
    if "--pron" in a:
        sys.exit(1 if pron_audit() else 0)
    if "--context" in a:
        sys.exit(1 if context_check(all_="--all" in a) else 0)
    lim = None
    if "--limit" in a:
        lim = int(a[a.index("--limit") + 1])
    build(redo_qa="--redo-qa" in a, limit=lim)
    report()
