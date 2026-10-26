# -*- coding: utf-8 -*-
"""Verify EVERY clip of a published bank — not a sample (the owner's order, 2026-10-25).

    python -m tools.voice_bank.verify_bank --lang en [--dir <unpacked clips>] [--no-listen] [--out <report.json>]

For every clip of the bank's segment inventory (tools/voice_audition/bank_plan.english_segments — the same list the
bank was rendered from, mapped to files by VoiceBank's own fnv64 naming):

  1. FILE      present, not empty, every byte bare MPEG frames of the bank's format (MPEG-2 Layer III, 24 kHz, mono,
               32 kbps CBR — what VoiceBank.swift / VoiceBank.kt join frame by frame): no ID3, no Xing, no stray bytes.
  2. LENGTH    duration vs the words it must say: SECONDS_PER_LETTER band on the spoken letters (numbers as their
               words), and never shorter than MIN_S.
  3. CLIPPING  decoded samples at full scale (|s| >= CLIP_LEVEL) in runs: a clipped clip sounds broken.
  4. TRUNCATED the last TAIL_MS (and the first HEAD_MS) of the decoded audio must be quiet relative to the clip's
               peak — speech still loud at the very end was cut off.
  5. LOUDNESS  integrated loudness (EBU R128, ffmpeg ebur128) — every clip within LOUD_TOL_LU of the bank's median, so
               lines joined from several clips do not jump in level.
  6. WORDS     a BLIND read-back (Azure speech-to-text, tools/voice_bank/stt.py — cached, the F0 ledger): the heard
               words must equal the segment's spoken words EXACTLY after number-as-words and the recogniser's
               homophones (no near-miss tolerance — "turn right" heard "turn left" is a FAIL, so is a dropped word).
               A road name (category "road") is compared by its frame words exactly and its transliterated name by
               consonant skeleton (EN_NAME_SKELETON_CER) — a recogniser cannot spell «Gamal Abdel Nasser».
               A clip that fails the first listen is heard ONCE MORE by a different listener (Gemini, the owner's free
               key, read at runtime) before it fails: the better listen decides. `--listener local` = Whisper base.en
               on the desk; `--recheck <report>` = Whisper small.en (+ Gemini) on what it rejected; `--final <report>
               [--extra <answers.json>]` = ONE verdict per clip from every stored answer (any listener hearing the exact
               words passes it) — the clips no listener hears right go to en_say.REDO to be re-recorded.
Writes the report (every clip's measurements and verdict) and prints the failures. Exit 1 when any clip fails.
"""
import argparse
import base64
import collections
import hashlib
import io
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.request

from tools.masri_voice import masri_voice as mv
from tools.voice_audition import audition as A
from tools.voice_audition import bank_plan as bp
from tools.voice_bank import build as B
from tools.voice_bank import stt

OUT_DIR = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
SECONDS_PER_LETTER = (0.035, 0.20)   # the bank build's pace gate (build.py §3)
MIN_S = 0.30
CLIP_LEVEL = 32700                   # |sample| at or above this is full scale (16-bit)
CLIP_RUN = 3                         # this many full-scale samples in a row = audible clipping
TAIL_MS, HEAD_MS = 60, 30
EDGE_REL_DB = -24.0                  # the edge windows' RMS must sit this far below the clip's peak RMS window
LOUD_TOL_LU = 4.0
# gemini-2.5-flash was retired for this key (2026-10-26: "no longer available to new users"); 3.8-flash answered 503
# then 429 on the free key the same morning, 3.5-flash answered (~20 clips before its own 429).
GEM_MODEL = os.environ.get("VERIFY_GEMINI_MODEL", "gemini-3.5-flash")
# The desk's own listener (2026-10-25): Azure's free speech-to-text quota can run out mid-run ("Quota exceeded"), so the
# words check also runs on a local Whisper (transformers, CPU, the model cached under HF_HOME like the Egyptian listener's
# wav2vec2) — the same exact-words rule, the Gemini second listen for what it rejects.
LOCAL_MODEL = "openai/whisper-base.en"
# --recheck: the clips the first listen rejected are heard again by a stronger local model (a base model mishears
# short clips: "bear left" -> "they're left") and by Gemini, with patience for a busy model (503).
LOCAL_MODEL_STRONG = "openai/whisper-small.en"
_ASR = {}


def local_listen(path, model=LOCAL_MODEL):
    if model not in _ASR:
        os.environ.setdefault("HF_HOME", "G:/grad_all/.hf-cache")
        from transformers import pipeline
        _ASR[model] = pipeline("automatic-speech-recognition", model=model, device="cpu")
    out = _ASR[model](path)
    return (out.get("text") or "").strip()


# ── 1. frames ────────────────────────────────────────────────────────────────────────────────────────────────────
BR2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]


def frames(b):
    """(n_frames, problems) — every byte must be a frame of the bank's format (VoiceBank.frameInfo's arithmetic)."""
    probs, n, i = [], 0, 0
    if len(b) >= 3 and b[:3] == b"ID3":
        probs.append("ID3 tag")
    while i + 4 <= len(b):
        b0, b1, b2, b3 = b[i], b[i + 1], b[i + 2], b[i + 3]
        if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
            probs.append("stray bytes at %d" % i)
            break
        ver, layer = (b1 >> 3) & 3, (b1 >> 1) & 3
        bri, sri, pad = (b2 >> 4) & 15, (b2 >> 2) & 3, (b2 >> 1) & 1
        if ver != 2 or layer != 1 or sri != 1:
            probs.append("not MPEG-2 L3 24 kHz at frame %d" % n)
            break
        if BR2[bri] != 32:
            probs.append("bitrate %d kbps at frame %d" % (BR2[bri], n))
            break
        if (b3 >> 6) != 3:
            probs.append("not mono at frame %d" % n)
            break
        ln = 72 * BR2[bri] * 1000 // 24000 + pad
        if i + ln > len(b):
            probs.append("last frame cut (%d of %d bytes)" % (len(b) - i, ln))
            break
        body = b[i + 4:i + min(ln, 64)]
        if b"Xing" in body or b"Info" in body or b"VBRI" in body:
            probs.append("Xing/Info header frame")
        n += 1
        i += ln
    if i < len(b) and not probs:
        probs.append("%d trailing bytes" % (len(b) - i))
    return n, probs


# ── 2-5. audio ───────────────────────────────────────────────────────────────────────────────────────────────────
def pcm(path):
    r = subprocess.run([stt._ffmpeg(), "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
                       capture_output=True)
    if r.returncode != 0:
        return None, r.stderr.decode("utf-8", "replace")[-200:]
    n = len(r.stdout) // 2
    return struct.unpack("<%dh" % n, r.stdout[:2 * n]), None


def rms_db(xs):
    if not xs:
        return -120.0
    m = math.sqrt(sum(x * x for x in xs) / len(xs))
    return 20.0 * math.log10(max(m, 1e-9) / 32768.0)


def loudness(path):
    r = subprocess.run([stt._ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null",
                        "-"], capture_output=True)
    t = r.stderr.decode("utf-8", "replace")
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", t)
    p = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", t)
    return (float(m[-1]) if m else None), (float(p[-1]) if p else None)


def audio_checks(path, seg):
    s, err = pcm(path)
    if s is None:
        return {"decode": err}, ["decode failed: %s" % err]
    probs = []
    dur = len(s) / 24000.0
    letters = B.spoken_letters(seg, "en")
    per = dur / letters
    if dur < MIN_S:
        probs.append("too short %.2f s" % dur)
    if per < SECONDS_PER_LETTER[0]:
        probs.append("too fast/short %.3f s per letter" % per)
    if per > SECONDS_PER_LETTER[1]:
        probs.append("too slow/long %.3f s per letter" % per)
    run = best = 0
    for x in s:
        if abs(x) >= CLIP_LEVEL:
            run += 1
            best = max(best, run)
        else:
            run = 0
    if best >= CLIP_RUN:
        probs.append("clipped (%d full-scale samples in a row)" % best)
    win = int(0.05 * 24000)
    peak_rms = max((rms_db(s[i:i + win]) for i in range(0, max(1, len(s) - win), win // 2)), default=-120.0)
    tail = rms_db(s[-int(TAIL_MS * 24):])
    head = rms_db(s[:int(HEAD_MS * 24)])
    if tail - peak_rms > EDGE_REL_DB:
        probs.append("cut off at the end (last %d ms at %.1f dB of the peak)" % (TAIL_MS, tail - peak_rms))
    if head - peak_rms > EDGE_REL_DB:
        probs.append("cut off at the start (first %d ms at %.1f dB of the peak)" % (HEAD_MS, head - peak_rms))
    lufs, tp = loudness(path)
    return {"dur_s": round(dur, 3), "s_per_letter": round(per, 4), "clip_run": best, "peak_rms_db": round(peak_rms, 1),
            "tail_db": round(tail - peak_rms, 1), "head_db": round(head - peak_rms, 1), "lufs": lufs,
            "true_peak_db": tp}, probs


# ── 6. the words ─────────────────────────────────────────────────────────────────────────────────────────────────
# How a recogniser WRITES words the clip SAYS — folded before the comparison, never a tolerance on the words:
# "1,100" (thousands comma), "0.5" for "point five" after a spoken number ("four point five" → "for 0.5"), and "100"
# for the bare "hundred" after a number word ("four hundred" → "for 100").
_NUM_WORD = r"(one|two|three|four|five|six|seven|eight|nine|won|to|too|for|fore|ate|[1-9])"


def spoken_form(s):
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s or "")
    s = re.sub(r"\b%s\s+0\.(\d)\b" % _NUM_WORD, r"\1 point \2", s, flags=re.I)
    s = re.sub(r"\b%s\s+100\b" % _NUM_WORD, r"\1 hundred", s, flags=re.I)
    return s


def words(s):
    return [w for w in B._en_norm(spoken_form(s))]


def same_word(a, b):
    return a == b or (a in B._HOMO and b in B._HOMO[a])


def exact_match(src, heard, road_name=None):
    """Exact equality of the spoken words (homophones only). A road segment: «onto <name>[.,]» — the frame word
    exact and the name by consonant skeleton."""
    s, h = words(src), words(heard)
    if road_name:
        if h and h[0] != "onto" and h[0].startswith("onto") and len(h[0]) > 4:     # "Ontolotfi, Alsayed Street"
            h = ["onto", h[0][4:]] + h[1:]
        if not h or h[0] != "onto":
            return False, "frame word «onto» not heard"
        sk_s, sk_h = B._skeleton(words(road_name)), B._skeleton(h[1:])
        cer = B._lev(sk_s, sk_h) / max(1, len(sk_s))
        return cer <= B.EN_NAME_SKELETON_CER, "name skeleton %s vs %s (%.2f)" % (sk_s, sk_h, cer)
    if len(s) != len(h) or not all(same_word(a, b) for a, b in zip(s, h)):
        return False, "heard «%s» for «%s»" % (" ".join(h), " ".join(s))
    return True, ""


def gemini_listen(path):
    key = A.api_key()
    audio = base64.b64encode(open(path, "rb").read()).decode("ascii")
    body = {"contents": [{"parts": [
        {"text": "Transcribe this English audio clip exactly, word for word, as spoken. Write numbers as digits. "
                 "Return only the transcript."},
        {"inline_data": {"mime_type": "audio/mpeg", "data": audio}}]}],
        "generationConfig": {"temperature": 0}}
    req = urllib.request.Request(A.API % GEM_MODEL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(6):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
        except Exception as e:                                   # noqa: BLE001
            time.sleep(6 * (attempt + 1))
            last = A.scrub(str(e)) if hasattr(A, "scrub") else type(e).__name__
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en"])
    ap.add_argument("--dir", default="G:/grad_all/.p5work/t1/en")
    ap.add_argument("--no-listen", action="store_true")
    ap.add_argument("--recheck", default=None,
                    help="a report: hear only its clips that failed on WORDS again (stronger local model + Gemini) "
                         "and write <report>.recheck.json")
    ap.add_argument("--final", default=None,
                    help="a report: one verdict per clip from every listener's stored answer (no new listening)")
    ap.add_argument("--extra", action="append", default=[],
                    help="with --final: a {segment: transcript} file from another listener's run")
    ap.add_argument("--listener", default="azure", choices=["azure", "local"],
                    help="local = Whisper on the desk (an Azure answer already cached still counts)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "verify_en.json"))
    a = ap.parse_args()
    if a.recheck:
        return recheck(a.recheck, a.dir)
    if a.final:
        return final(a.final, a.dir, a.extra)
    roads_json = os.path.join(bp.OUT, "major_roads.json")
    roads = [r for r in json.load(io.open(roads_json, encoding="utf-8"))["roads"] if r["km"] >= bp.ROAD_MIN_KM]
    road_of = {}
    for r in roads:
        if r.get("en"):
            road_of["onto %s." % r["en"]] = r["en"]
            road_of["onto %s," % r["en"]] = r["en"]
    segs = bp.english_segments(roads)
    rows, fails = [], []
    for k, (seg, cat) in enumerate(sorted(segs.items(), key=lambda kv: (kv[1], kv[0]))):
        name = mv.fnv64(seg) + ".mp3"
        path = os.path.join(a.dir, name)
        row = {"seg": seg, "cat": cat, "file": name, "problems": []}
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            row["problems"].append("missing or empty")
            rows.append(row)
            fails.append(row)
            continue
        b = open(path, "rb").read()
        n, fp = frames(b)
        row["frames"] = n
        row["problems"] += fp
        m, ap_ = audio_checks(path, seg)
        row.update(m)
        row["problems"] += ap_
        if not a.no_listen and a.listener == "local":
            heard = local_listen(path)
            row["heard_local"] = heard
            ok, why = exact_match(seg, heard, road_of.get(seg))
            if not ok:
                g = gemini_listen(path)
                row["heard_gemini"] = g
                ok2, why2 = exact_match(seg, g or "", road_of.get(seg))
                if not ok2:
                    row["problems"].append("words: local %s | gemini %s" % (why, why2))
        elif not a.no_listen:
            try:
                r = stt.verdict(b, "en")
                row["heard_azure"] = r["transcript"]
                ok, why = exact_match(seg, r["transcript"], road_of.get(seg))
                if not ok:
                    ok2 = False
                    if r.get("itn"):
                        ok2, _ = exact_match(seg, r["itn"], road_of.get(seg))
                    if not ok2:
                        g = gemini_listen(path)
                        row["heard_gemini"] = g
                        ok2, why2 = exact_match(seg, g or "", road_of.get(seg))
                        if not ok2:
                            row["problems"].append("words: azure %s | gemini %s" % (why, why2))
            except stt.Exhausted as e:
                row["problems"].append("not heard: %s" % e)
        rows.append(row)
        if row["problems"]:
            fails.append(row)
        if k % 100 == 0:
            print("%4d/%d  failures so far %d" % (k, len(segs), len(fails)), flush=True)
    lufs = [r["lufs"] for r in rows if r.get("lufs") is not None]
    med = statistics.median(lufs) if lufs else None
    for r in rows:
        if med is not None and r.get("lufs") is not None and abs(r["lufs"] - med) > LOUD_TOL_LU:
            r["problems"].append("loudness %.1f LUFS vs the bank's median %.1f" % (r["lufs"], med))
            if r not in fails:
                fails.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    io.open(a.out, "w", encoding="utf-8").write(json.dumps({"median_lufs": med, "clips": rows}, ensure_ascii=False,
                                                           indent=0))
    print("\n%d clips, %d failing; median loudness %s LUFS (spread %s..%s)" % (
        len(rows), len(fails), med, min(lufs) if lufs else None, max(lufs) if lufs else None))
    by = collections.Counter(p.split(" (")[0].split(":")[0] for r in fails for p in r["problems"])
    for p, c in by.most_common():
        print("  %4d  %s" % (c, p))
    for r in fails[:40]:
        print("FAIL %-50s %s" % (r["seg"][:50], "; ".join(r["problems"])[:160]))
    print("report:", a.out)
    return 1 if fails else 0


def azure_cached(audio):
    """Azure's read-back of these exact bytes when the bank build or an earlier run already paid for it — never a new
    request (the F0 month can be spent)."""
    wav = stt.wav16(audio)
    locale = stt.LOCALES.get("en", "en")
    k = hashlib.sha256(locale.encode() + b"\n" + wav).hexdigest()
    p = os.path.join(stt.CACHE, k[:2], k + ".json")
    if not os.path.exists(p):
        return None
    r = json.load(io.open(p, encoding="utf-8"))
    return [t for t in (r.get("lexical"), r.get("itn"), r.get("display")) if t] if r.get("status") == "Success" else []


def final(report, clip_dir, extra):
    """ONE verdict per clip from EVERY listener that heard it: the report's own listen, the --recheck answers, any
    extra {segment: transcript} files (another listener's run) and Azure's cached read-back. A clip PASSES when any
    listener heard its exact words (same rule, same folding) — the listeners mishear differently, a clip that says
    the words is heard right by one of them; it FAILS only when every listener heard something else, and those are
    the clips to regenerate."""
    roads_json = os.path.join(bp.OUT, "major_roads.json")
    roads = [r for r in json.load(io.open(roads_json, encoding="utf-8"))["roads"] if r["km"] >= bp.ROAD_MIN_KM]
    road_of = {}
    for r in roads:
        if r.get("en"):
            road_of["onto %s." % r["en"]] = r["en"]
            road_of["onto %s," % r["en"]] = r["en"]
    d = json.load(io.open(report, encoding="utf-8"))
    rc = report.replace(".json", ".recheck.json")
    again = {}
    if os.path.exists(rc):
        for c in json.load(io.open(rc, encoding="utf-8"))["rechecked"]:
            again[c["seg"]] = c.get("recheck") or {}
    more = {}
    for f in extra:
        for seg, t in json.load(io.open(f, encoding="utf-8")).items():
            if t:
                more.setdefault(seg, []).append(t)
    rows, fails, mech = 0, [], 0
    for c in d["clips"]:
        rows += 1
        other = [p for p in c["problems"] if not p.startswith("words")]
        if other:
            mech += 1
            fails.append((c, "; ".join(other), []))
            continue
        if not any(p.startswith("words") for p in c["problems"]):
            continue
        heard = [c.get("heard_local"), c.get("heard_azure"), c.get("heard_gemini"),
                 again.get(c["seg"], {}).get("local_strong"), again.get(c["seg"], {}).get("gemini")]
        heard += more.get(c["seg"], [])
        az = azure_cached(open(os.path.join(clip_dir, c["file"]), "rb").read())
        heard += az or []
        heard = [h for h in heard if h]
        if any(exact_match(c["seg"], h, road_of.get(c["seg"]))[0] for h in heard):
            continue
        fails.append((c, "no listener heard it", heard))
    print("%d clips; %d fail a mechanical check; %d fail the words after every listener" % (
        rows, mech, len(fails) - mech))
    for c, why, heard in fails:
        print("FAIL %-40s %s  %s" % (c["seg"][:40], why, " / ".join("«%s»" % h[:40] for h in dict.fromkeys(heard))))
    return 1 if fails else 0


def recheck(report, clip_dir):
    """Hear the clips a report failed on WORDS again: the stronger local model, then Gemini. A clip passes when either
    hears its words exactly (the same rule); the rest are printed for regeneration."""
    roads_json = os.path.join(bp.OUT, "major_roads.json")
    roads = [r for r in json.load(io.open(roads_json, encoding="utf-8"))["roads"] if r["km"] >= bp.ROAD_MIN_KM]
    road_of = {}
    for r in roads:
        if r.get("en"):
            road_of["onto %s." % r["en"]] = r["en"]
            road_of["onto %s," % r["en"]] = r["en"]
    d = json.load(io.open(report, encoding="utf-8"))
    todo = [c for c in d["clips"] if any(p.startswith("words") for p in c["problems"])]
    still = []
    for c in todo:
        path = os.path.join(clip_dir, c["file"])
        heard = local_listen(path, LOCAL_MODEL_STRONG)
        ok, why = exact_match(c["seg"], heard, road_of.get(c["seg"]))
        g = None
        if not ok and not os.environ.get("VERIFY_NO_GEMINI"):   # a busy model costs ~2 min a clip in retries
            g = gemini_listen(path)
            ok2, why2 = exact_match(c["seg"], g or "", road_of.get(c["seg"]))
            ok = ok2
            why = "%s | gemini %s" % (why, why2)
        c["recheck"] = {"local_strong": heard, "gemini": g, "ok": ok}
        if not ok:
            still.append(c)
        print("%-5s %-48s small.en «%s»%s" % ("ok" if ok else "FAIL", c["seg"][:48], heard[:50],
                                             "" if g is None else "  gemini «%s»" % (g or "")[:50]), flush=True)
    out = report.replace(".json", ".recheck.json")
    io.open(out, "w", encoding="utf-8").write(json.dumps({"rechecked": todo, "still_failing": still},
                                                         ensure_ascii=False, indent=0))
    print("\n%d re-heard, %d pass now, %d still failing -> %s" % (len(todo), len(todo) - len(still), len(still), out))
    return 1 if still else 0


if __name__ == "__main__":
    sys.exit(main())
