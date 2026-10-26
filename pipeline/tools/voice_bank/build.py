# -*- coding: utf-8 -*-
"""The navigation voice bank — the generator. The owner's voices (2026-10-17): ARABIC = gemini-3.8-flash-tts / Despina
with the audition's settings, ENGLISH = Azure en-US-PhoebeMultilingualNeural. The daily run that drives this file and
publishes the result is tools/voice_bank/daily.py (docs/voice-bank-publish-2026-10.md); this CLI is the manual tool.

    python -m tools.voice_bank.build run --provider azure --first 1200   # render the N most spoken segments
    python -m tools.voice_bank.build judge                               # the gates (Gemini judge + phone listener)
    python -m tools.voice_bank.build retake                              # failed clips: the next SAY spelling
    python -m tools.voice_bank.build export                              # passed clips → <build>/release/
    python -m tools.voice_bank.build status
    python -m tools.voice_bank.build plan --provider azure [--first N]   # characters / requests, no network

  options: --voice-ar / --ar-rate / --ar-pitch / --ar-spell pausal|say / --voice-en / --en-rate / --en-pitch /
  --en-style (defaults = docs/azure-voice-2026-10.md's recommendation: audition.AZ_BANK_AR + AZ_AR_FINAL, AZ_EN_PICK[0]),
  --lang ar|en|all, --first N (the N most spoken segments, both languages), --batch-ar K / --batch-en K / --batch K,
  --sep nl|s|break, --max-chars N (this run), --no-phones; --provider gemini --voice-ar Despina --voice-en Leda
  (25 per request). judge / retake / status / export take --build ID when more than one build exists with other
  settings (the id is the settings' hash; the same options find the same build). The daily run's builds are named
  (`ar-gemini-despina`, `en-azure-phoebe`: tools/voice_bank/banks.py).

WHAT IT RENDERS: the ranked inventory (tools/voice_bank/inventory.py — Arabic 1,070 composer segments + 680 road
segments, English 1,254 incl. 672 road segments), MOST SPOKEN FIRST and every composer segment before any road name
(`prio`), so a first batch already speaks most lines. Every clip is the bank's format (MPEG-2 L3, 24 kHz mono 32 kbps,
trimmed and levelled by build_bank.process) named fnv64(segment).mp3 — VoiceBank's own naming — so the downloaded bank
and the bundled one are read the same way. The bundled Arabic bank (assets/voice/ar) stays the fallback; nothing here
touches it. No TTS at navigation time.

PER PROVIDER (measured 2026-10-17, docs/azure-voice-2026-10.md §5):
  • azure — 20 requests a minute, no daily request quota, a MONTHLY CHARACTER budget (F0 500,000, shared with Sekka's
    voice). Characters are the cost, so the request size follows the markup: one Arabic segment per request pays the
    42-character prosody wrapper per segment (120,151 characters for the Arabic bank); AZ_BANK_BATCH segments in one
    request separated by AZ_BANK_SEP pay it once per request (49,591) and are cut by the aligned cut (the separator
    is billable too). English carries no markup, so it goes one segment per request (no cut at all).
    Retakes need a DIFFERENT input (Azure is deterministic): the next SAY spelling, else the clip is flagged for a
    human listen. Every request is counted in the desk ledger (azure.py) and refused past --max-chars or the month's
    BANK_MONTH_CEILING (the rest of the month is Sekka's).
  • gemini — 10 TTS requests a day per model (free tier), so 25 segments a request with the audition's director's-notes
    prompt and the aligned cut; the run spends the day's quota and stops; the next day resumes (nothing is asked
    twice). RETAKES ARE BATCHED too (the failed segments of the queue, 25 a request, each reading its next SAY
    spelling) and share the queue with first takes in priority order — a failed top-ranked segment is retaken before a
    low-ranked one gets its first take. A request whose cut fails is re-asked in two halves, never one per segment.

THE GATES (docs/voice-bank-plan-2026-10.md §f): §1 the cut (one piece per segment; a Gemini request whose audio is
outside LEN_RATIO × its text's expected length fails every piece unheard); §3 pace 0.035-0.20 s per SPOKEN letter
(English numbers counted as the words said); §4-7 the blind judge's read-back (Arabic heard INSIDE a line behind the
bank's lead clip, as build_bank judges a direction word; CER, direction words / ordinals, the addressee's gender,
Cairene with ج = g and ق = glottal) — the bank's judges are the audition's chain WITHOUT the flash-lite family
(BANK_JUDGES), a batch that comes back with one transcript for two clips or shifted by a clip is thrown away unheard,
and a read-back / dialect failure is heard ONCE MORE by another judge model (twice for Azure, which cannot retake)
before it stands — the clip is judged by the better listen, as build_bank's Flash → Pro second listen; §8 the blind
phone listener (Arabic: no hard flag; when it cannot run, Arabic verdicts wait for the next run); §9 consistency: a
Gemini piece more than F0_DRIFT_CENTS from the voice's median pitch fails; §10 up to MAX_TAKES takes, then the clip
is kept for a human listen (status `listen`) and never published. `export` / publish.py ship only passing clips.

OUTPUT (never in the repo): $VOICE_BANK_OUT (default G:/grad_all/.p5work/voice_bank)/<build id>/.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
from xml.sax.saxutils import escape

from tools.masri_voice import acoustics
from tools.masri_voice import masri_voice as mv
from tools.masri_voice import pronounce
from tools.voice_audition import audition as A
from tools.voice_bank import align as AL
from tools.voice_bank import en_say as ES
from tools.voice_bank import inventory as INV

OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
BANK_MONTH_CEILING = 250_000      # the desk's Azure characters in one month (bank + audition); Sekka keeps the rest
# Clips per judge request. The plan assumed 12; the smoke build (2026-10-17) heard 12 / 10 / 5 clips per request with
# gemini-3.5-flash-lite (the last judge left that day): it returned ONE transcript of every clip for each English
# clip and shifted the Arabic ones by a clip. 6 is the audition's measured size (JUDGE_BATCH); `unreliable` below
# throws away a batch that comes back like that instead of failing its clips.
JUDGE_CLIPS = 6
# The judges the bank trusts: the audition's chain without the flash-lite family (the smoke build's failures above).
BANK_JUDGES = [m for m in A.JUDGE_CHAIN if "lite" not in m]
# Extra listens by ANOTHER judge model before a read-back / dialect failure stands (Azure cannot retake: two).
RELISTENS = {"gemini": 1, "azure": 2}
# THE JUDGE (2026-10-17, the owner): "stt" = the quota-free listeners — the Azure speech-to-text read-back
# (tools/voice_bank/stt.py; deterministic, so no second listen: a failed English clip is RE-RENDERED instead) + for
# Arabic the blind phone listener; the free Gemini TTS requests are then the bank's ONLY Gemini spend. "gemini" = the
# Gemini judge chain below (kept for a manual re-hear: `build judge --judge gemini`).
DEFAULT_JUDGE = "stt"
# Azure English retakes: Azure is deterministic, so a retake is a different rendition — the same words at another
# rate (take 2, take 3). The first take's settings (the bank's prompt_id) are unchanged.
EN_RETAKE_RATES = ["-8%", "-16%"]
# The STT read-back of a ROAD segment ("onto Al Azhar Bridge,"): the frame words must be heard exactly; the
# transliterated Arabic name is judged on its consonant skeleton (a recogniser spells "Kornish Al Nil" its own way —
# what matters is that the name was SAID, whole) — at most this edit distance per skeleton letter.
EN_NAME_SKELETON_CER = 0.40
EN_CRITICAL = ["right", "left", "straight", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
               "eighth", "u turn", "exit"]
EN_CRITICAL_WORDS = {w for c in EN_CRITICAL for w in c.split()} - {"turn"}
EN_ROAD_TYPES = {"street", "road", "bridge", "axis", "tunnel", "corridor", "square", "highway", "corniche"}
EN_FRAME = {"onto", "street", "road", "bridge", "axis", "tunnel", "corridor", "square", "highway", "desert", "ring",
            "corniche", "of", "the", "to", "from"}
MAX_DISCARDS = 2                  # a clip whose batch was thrown away twice keeps the third verdict
# Azure Arabic: segments per request and their separator — the audition's batching measurement
# (docs/azure-voice-2026-10.md §5): one <prosody> wrapper per request instead of per segment.
AZ_BANK_BATCH = 25
AZ_BANK_SEP = "nl"
GEMINI_BATCH = 25                 # the plan's lower bound (§d: 25-40 segments a request) — the measured base
# The first Gemini day MEASURES larger requests before one is adopted (Build.calibrate): the same CAL_SHARED segments
# alone and spread through a request of each size; the largest size within CAL_TOL of the base wins. 150 segments ≈
# 385 s of speech ≈ 9,600 output tokens (the model's limit: 16,384 ≈ 650 s). The proxy on gemini-3.8-flash-lite-tts
# (Despina, 2026-10-17, the STT judge): shared segments 25 → 0.84, 50 → 0.96, 100 → 0.88 (all 100: 0.93) — the
# 100-segment take slipped by a line in one stretch and was re-cut by its transcript (realign).
CAL_SIZES = [25, 50, 100, 150]
CAL_SHARED = 25
CAL_TOL = 0.08                    # 2 of 25
REALIGN_MIN = 2                   # pieces reading as a neighbour's line before a request is cut again by transcript
MAX_TAKES = 3
MAX_ERRORS = 3                    # requests that gave no audio for a segment before it waits for a human
PACE_MIN, PACE_MAX = 0.035, 0.20  # seconds per spoken letter
LEN_RATIO = (0.5, 2.0)            # §f.1: a batched Gemini request's audio against its text's expected length
EXPECT_LPS = 11.4                 # letters a second of the Gemini voices (the audition's measurement, plan §g)
LINE_PAUSE_S = 1.0                # the batch prompt's "clear one-second pause after every line"
F0_DRIFT_CENTS = 600              # §f.9: half an octave off the voice's median = another voice / register
F0_MIN_REF = 20                   # clips before the voice's median is trusted
EN_LEAD = "In 300 meters,"        # English segments are judged alone; the lead is for the page's context only


def phrasebook_hash(full=False):
    p = os.path.join(mv.ROOT, "nav-spec", "masri_voice.json")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return h if full else h[:16]


def prio(x):
    """The build order: every composer segment before any road name, then the most spoken first."""
    return (x["cat"] == "road", x["rank"])


def spoken_letters(seg, lang):
    """The letters a voice SAYS for a segment — English digits as their words ("700 meters" = seven hundred meters)."""
    if lang == "en":
        return max(1, sum(len(w.replace("'", "")) for w in A.en_words(seg)))
    return max(1, sum(1 for ch in seg if ch.isalpha()))


def az_doc(c, texts, lang):
    """The SSML of one Azure request: one segment, or K segments separated by c["sep"] (billable markup)."""
    from tools.voice_audition import azure as AZR
    voice = c["voice_ar"] if lang == "ar" else c["voice_en"]
    rate = c["ar_rate"] if lang == "ar" else c["en_rate"]
    pitch = c["ar_pitch"] if lang == "ar" else c["en_pitch"]
    style = None if lang == "ar" else (c.get("en_style") or None)
    if len(texts) == 1:
        return AZR.ssml(texts[0], voice, lang, rate=rate or None, pitch=pitch or None, style=style)
    body = A.AZ_SEPS[c["sep"]].join(escape(t) for t in texts)
    if c["sep"] == "s":
        body = "<s>" + body + "</s>"
    return AZR.ssml(None, voice, lang, rate=rate or None, pitch=pitch or None, style=style, body_xml=body)


def atomic_write(path, text):
    """Write `path` whole or not at all. Windows refuses the rename while another process (a reader, the virus
    scanner) has the old file open — the first daily run lost a step to that (WinError 5): retry for ~20 s."""
    import time
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    for i in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.5)


def _exhausted():
    """Every 'out of quota / budget' exception of both providers (a clean stop, resumable)."""
    from tools.voice_audition import azure as AZR
    return (A.Exhausted, AZR.Exhausted)


class Build:
    def __init__(self, a):
        self.a = a
        cfg = {"provider": a.provider, "voice_ar": a.voice_ar, "voice_en": a.voice_en, "ar_rate": a.ar_rate,
               "ar_pitch": a.ar_pitch, "en_rate": a.en_rate, "en_pitch": a.en_pitch, "en_style": a.en_style,
               "batch_ar": a.batch_ar, "batch_en": a.batch_en, "sep": a.sep, "ar_spell": a.ar_spell}
        if a.provider == "gemini":
            cfg["model"] = getattr(a, "model", None) or A.MAIN_MODEL
        self.id = a.build or "%s-%s" % (a.provider, hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
                                        .hexdigest()[:10])
        self.dir = os.path.join(OUT, self.id)
        os.makedirs(os.path.join(self.dir, "clips"), exist_ok=True)
        self.state_p = os.path.join(self.dir, "state.json")
        if os.path.exists(self.state_p):
            self.s = json.load(io.open(self.state_p, encoding="utf-8"))
            if not a.build:
                self.s["cfg"] = cfg
        else:
            self.s = {"id": self.id, "cfg": cfg, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                      "phrasebook": phrasebook_hash(), "segs": {}, "requests": []}
        self.cfg = self.s["cfg"]
        self.need_phones = not getattr(a, "no_phones", False)
        self.judge_mode = getattr(a, "judge", None) or DEFAULT_JUDGE

    def save(self):
        atomic_write(self.state_p, A.scrub(json.dumps(self.s, ensure_ascii=False, indent=1)))

    def rec(self, x):
        return self.s["segs"].setdefault(x["lang"] + "|" + x["seg"], {
            "seg": x["seg"], "lang": x["lang"], "cat": x["cat"], "rank": x["rank"], "takes": [], "status": "pending"})

    # ── what to make ─────────────────────────────────────────────────────────────────────────────────────────
    def can_retake(self, r):
        if len(r["takes"]) >= MAX_TAKES or r.get("errors", 0) >= MAX_ERRORS:
            return False
        if self.cfg["provider"] == "azure":       # deterministic: only a different input is a new take
            if r["lang"] == "en":
                if ES.variants(r["seg"]):         # a spoken-form override: one take per variant (en_say.py)
                    return len(r["takes"]) < len(ES.variants(r["seg"]))
                return len(r["takes"]) < 1 + len(EN_RETAKE_RATES)
            return pronounce.variants(r["seg"]) > len(r["takes"])
        return True

    def queue(self, order, langs=("ar", "en"), first=None):
        """[(segment, take number)] still to render — first takes and retakes together, in build order (`prio`)."""
        firsts = {(x["lang"], x["seg"]) for x in order[:first]} if first else None
        self.apply_en_say()
        out = []
        for x in order:
            if x["lang"] not in langs or (firsts is not None and (x["lang"], x["seg"]) not in firsts):
                continue
            r = self.rec(x)
            if r.get("errors", 0) >= MAX_ERRORS:
                continue
            if r["status"] == "pending" and not r["takes"]:
                out.append((x, 1))
            elif r["status"] == "fail" and self.can_retake(r):
                out.append((x, len(r["takes"]) + 1))
        out.sort(key=lambda it: prio(it[0]))
        return out

    def todo(self, order):                          # (kept for callers of the first version)
        return [x for x, t in self.queue(order, (self.a.lang,) if self.a.lang != "all" else ("ar", "en")) if t == 1]

    def apply_en_say(self):
        """A segment whose English spoken-form override (en_say.py) is new or changed, and that has no passing take, is
        rendered afresh: its earlier takes (made from the written text or an older override) are archived under
        `old_takes` and their clip files moved aside (encode_mp3 never overwrites). → how many were reset."""
        n = 0
        for r in self.s["segs"].values():
            if r["lang"] != "en":
                continue
            oid = ES.override_id(r["seg"])
            redo = r["seg"] in ES.REDO
            if oid is None or r.get("say_id") == oid or (any(t.get("pass") for t in r["takes"]) and not redo):
                continue
            keep = self.published_take(r) if redo else None
            for t in r["takes"]:
                src = os.path.join(self.dir, t["clip"])
                if os.path.exists(src):
                    k = 0
                    while os.path.exists("%s.old%d" % (src, k)):
                        k += 1
                    os.replace(src, "%s.old%d" % (src, k))
                    t["clip"] = os.path.relpath("%s.old%d" % (src, k), self.dir).replace(os.sep, "/")
            r.setdefault("old_takes", []).extend(r["takes"])
            if keep is not None:                # REDO: the published take ships until a new take passes
                r["fallback"] = keep
            r.update(takes=[], status="pending", say_id=oid, errors=0)
            r.pop("last_error", None)
            n += 1
        if n:
            self.save()
        return n

    def text_for(self, seg, lang, take):
        if lang == "en":                    # the words the voice is asked to say (en_say.py), else the segment
            v = ES.variant(seg, take)
            return ES.plain_of(v) if v else seg
        if lang != "ar":
            return seg
        t = pronounce.say(seg, (take - 1) % pronounce.variants(seg))
        if self.cfg["provider"] == "azure" and self.cfg.get("ar_spell", "pausal") == "pausal":
            t = A.pausal(A.az_respell(t))       # the audition's Azure findings (docs/azure-voice-2026-10.md §2)
        return t

    # ── Azure ────────────────────────────────────────────────────────────────────────────────────────────────
    def az_doc(self, texts, lang):
        return az_doc(self.cfg, texts, lang)

    def az_guard(self, n):
        from tools.voice_audition import azure as AZR
        L = AZR.ledger()
        month = datetime.date.today().isoformat()[:7]
        if L.d["by_month"].get(month, 0) + n > BANK_MONTH_CEILING:
            raise AZR.Exhausted("the month's desk ceiling (%d characters; the rest is Sekka's) would be passed"
                                % BANK_MONTH_CEILING)
        spent = sum(r.get("chars", 0) for r in self.s["requests"] if r.get("run") == self.s.get("run"))
        if self.a.max_chars and spent + n > self.a.max_chars:
            raise AZR.Exhausted("--max-chars %d reached" % self.a.max_chars)

    def render_azure(self, group, takes):
        from tools.voice_audition import azure as AZR
        lang = group[0]["lang"]
        texts = [self.text_for(x["seg"], lang, t) for x, t in zip(group, takes)]
        v = ES.variant(group[0]["seg"], takes[0]) if lang == "en" and len(group) == 1 else None
        if v:                                  # a spoken-form override: its SSML body and its own rate (en_say.py)
            doc = AZR.ssml(None, self.cfg["voice_en"], "en", rate=v.get("rate") or self.cfg["en_rate"] or None,
                           pitch=self.cfg["en_pitch"] or None, style=self.cfg.get("en_style") or None,
                           body_xml=v["ssml"])
        elif lang == "en" and len(group) == 1 and takes[0] > 1:   # an English retake: the same words, another rate
            doc = az_doc(dict(self.cfg, en_rate=EN_RETAKE_RATES[min(takes[0], 1 + len(EN_RETAKE_RATES)) - 2]),
                         texts, lang)
        else:
            doc = self.az_doc(texts, lang)
        self.az_guard(AZR.billable(doc))
        wav, meta = AZR.synth_wav(doc, purpose="bank")
        return texts, wav, meta

    # ── Gemini ───────────────────────────────────────────────────────────────────────────────────────────────
    def render_gemini(self, group, takes, extra_texts=()):
        """The audition's settings exactly: the model, the voice, the director's-notes prompt (prompt_ar /
        prompt_ar_batch), no generationConfig beyond the voice (the API's default temperature), the SAY spellings —
        STREAMED (gemini_tts: the same request, its time to first audio measured). `extra_texts` ride at the end of
        the same request (Sekka's live measurement: no request of its own)."""
        from tools.voice_bank import gemini_tts as GT
        lang = group[0]["lang"]
        texts = [self.text_for(x["seg"], lang, t) for x, t in zip(group, takes)]
        voice = self.cfg["voice_ar"] if lang == "ar" else self.cfg["voice_en"]
        allt = texts + list(extra_texts)
        if len(allt) == 1:
            prompt = (A.prompt_ar if lang == "ar" else A.prompt_en)(allt[0])
        else:
            prompt = (A.prompt_ar_batch if lang == "ar" else A.prompt_en_batch)("\n".join(allt), len(allt))
        key, wav, meta = GT.synth(self.cfg.get("model") or A.MAIN_MODEL, voice, prompt, max(takes))
        return texts, wav, dict(meta, key=key)

    # ── one request → clips ──────────────────────────────────────────────────────────────────────────────────
    def make(self, group, take=1, takes=None, extras=None, calib=None):
        """ONE request for `group` (+ `extras` [(id, plain, say)] appended: pieces returned, never takes; `calib` =
        segments whose pieces are CALIBRATION pieces, recorded apart from the takes) → the request's record."""
        takes = list(takes or [take] * len(group))
        extras = list(extras or [])
        calib = set(calib or ())
        gem = self.cfg["provider"] == "gemini"
        if gem:
            texts, wav, meta = self.render_gemini(group, takes, [e[2] for e in extras])
        else:
            texts, wav, meta = self.render_azure(group, takes)
        lang = group[0]["lang"]
        allt = texts + [e[2] for e in extras]
        plain = [x["seg"] for x in group] + [e[1] for e in extras]
        k = len(allt)
        x_all, rate = acoustics.pcm(wav)
        audio_s = len(x_all) / float(rate)
        pieces = [wav] if k == 1 else A.split_on_silence(wav, k, weights=[A.letters_of(t) for t in allt])
        gate = None
        if gem and k > 1:                            # §f.1: a request that added (or lost) whole lines
            exp = sum(spoken_letters(t, lang) for t in plain) / EXPECT_LPS + LINE_PAUSE_S * (k - 1)
            ratio = audio_s / max(0.1, exp)
            if not LEN_RATIO[0] <= ratio <= LEN_RATIO[1]:
                gate = "the request came back %.2f x its expected length" % ratio
        recovered = None
        if gem and k > 2 and lang == "ar" and (pieces is None or gate or AL.suspicious(wav, k)):
            # the pauses cannot be trusted (an added / dropped line): cut by what was SAID (align.py)
            from tools.voice_bank import stt as STT
            try:
                tp, rep = AL.transcript_cut(wav, plain, lang)
                A.log("  transcript cut: %d of %d lines matched (%d islands, %d left out, %d missing)%s" % (
                    rep["matched"], k, rep["islands"], rep["added_islands"], rep["missing"],
                    " — %s" % gate if gate else ""))
                if rep["matched"] >= max(1, k // 2):
                    pieces, recovered = tp, {key: rep[key] for key in ("islands", "matched", "added_islands",
                                                                     "missing")}
                    recovered["why"] = gate or ("no cut" if pieces is None else "the pauses do not match the lines")
                    gate = None
            except (STT.Exhausted, STT.SttError) as e:
                A.log("  transcript cut unavailable: %s" % A.scrub(e)[:160])
        req = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "n": len(group), "extras": len(extras),
               "calib": len(calib), "chars": meta.get("chars", 0), "latency_s": meta.get("latency_s"),
               "first_audio_s": meta.get("first_audio_s"), "audio_s": round(audio_s, 1),
               "tokens_out": ((meta.get("usage") or {}).get("candidatesTokenCount")), "cut_ok": pieces is not None,
               "take": max(takes), "gate": gate, "recovered": recovered, "run": self.s.get("run"),
               "key": meta.get("key") or None, "lang": lang,
               "segs": plain if (gem and k > 2) else None}
        self.s["requests"].append(req)
        res = {"req": req, "extra_pieces": [], "meta": {k: meta.get(k) for k in ("first_audio_s", "latency_s",
                                                                                   "audio_s", "usage", "key")}}
        if pieces is None:
            if len(group) > 1:                       # the cut failed: re-ask (Gemini: two halves; Azure: one each)
                if gem:
                    h = len(group) // 2
                    self.make(group[:h], takes=takes[:h])
                    self.make(group[h:], takes=takes[h:])
                else:
                    for x, t in zip(group, takes):
                        self.make([x], takes=[t])
            self.save()
            return res
        cal = self.s.setdefault("calibration", {}).setdefault("pieces", {}) if calib else None
        for i, (x, t_no, t, pc) in enumerate(zip(group, takes, texts, pieces)):
            r = self.rec(x)
            if pc is None:                           # not in the take (the transcript cut found no match)
                if x["seg"] in calib:
                    continue
                r["errors"] = r.get("errors", 0) + 1
                r["last_error"] = "missing from a batched take (%s)" % (recovered or {}).get("why")
                continue
            out = A.processed(pc)
            st = acoustics.clip_stats(pc, x["seg"])
            spl = st["speech_s"] / spoken_letters(x["seg"], x["lang"])
            is_cal = x["seg"] in calib
            rel = "clips/%s/%s%d/%s.mp3" % (x["lang"], "cal%d-t" % k if is_cal else "t", t_no, mv.fnv64(x["seg"]))
            A.encode_mp3(out, os.path.join(self.dir, rel))
            rec = {"take": t_no, "text": t, "clip": rel, "s_per_letter": round(spl, 4),
                   "pace_ok": PACE_MIN <= spl <= PACE_MAX, "batched": k, "pos": i,
                   "audio_s": round(len(acoustics.pcm(pc)[0]) / float(rate), 2), "f0": st["f0_med"],
                   "gate": gate, "listens": [], "verdict": None, "phones": None, "pass": None,
                   "first_audio_s": meta.get("first_audio_s"), "recovered": bool(recovered),
                   "req": meta.get("key") if (gem and k > 2) else None}
            if x["lang"] == "en" and ES.variant(x["seg"], t_no):
                rec["spoken"] = ES.plain_of(ES.variant(x["seg"], t_no))      # the judge reads back against this
                rec["say_id"] = ES.override_id(x["seg"])
                r["say_id"] = rec["say_id"]
            if is_cal:
                cal.setdefault(x["seg"], {})[str(k)] = rec
                continue
            r["takes"].append(rec)
            r["status"] = "made"
        for e, pc in zip(extras, pieces[len(group):]):
            res["extra_pieces"].append((e[0], pc))
        self.save()
        return res

    def tts_round(self, order, langs=("ar", "en"), first=None, max_requests=None, extras=None, on_extras=None):
        """Render the queue until it is empty, the quota / budget stops it, or `max_requests`. → why it stopped.
        `extras` ride at the end of the first Arabic request (Sekka's live measurement); `on_extras(result)` gets
        their pieces."""
        self.s["run"] = datetime.datetime.now().isoformat(timespec="seconds")
        q = self.queue(order, langs, first)
        A.log("== %s: %d segments to render (%d retakes), %s / %s per request (ar / en)" % (
            self.id, len(q), sum(1 for _, t in q if t > 1), self.cfg["batch_ar"], self.cfg["batch_en"]))
        groups = []
        for lang in ("ar", "en"):
            xs = [it for it in q if it[0]["lang"] == lang]
            k = max(1, int(self.cfg["batch_" + lang]))
            groups += [xs[i:i + k] for i in range(0, len(xs), k)]
        from tools.voice_audition import azure as AZR
        n = 0
        refused_in_a_row = 0
        for g in groups:
            if max_requests is not None and n >= max_requests:
                self.save()
                return "limit (%d requests)" % n
            try:
                ex = extras if (extras and g[0][0]["lang"] == "ar" and self.cfg["provider"] == "gemini") else None
                res = self.make([x for x, _ in g], takes=[t for _, t in g], extras=ex)
                if ex:
                    extras = None
                    if on_extras:
                        on_extras(res)
            except _exhausted() as e:                 # quota / ceiling: stop, resumable
                A.log("  stopped: %s" % A.scrub(e)[:200])
                self.save()
                return "quota: %s" % A.scrub(e)[:160]
            except (RuntimeError, AZR.AzureError) as e:
                msg = A.scrub(e)[:200]
                # only an answer ABOUT these segments (no audio / a 4xx refusal) is counted against them; anything
                # else (the network, a 5xx storm, a broken tool) stops the round — the next run resumes
                if not ("no audio" in msg or "HTTP 4" in msg or "azure HTTP 4" in msg or "empty audio" in msg):
                    A.log("  stopped (the network / the service): %s" % msg)
                    self.save()
                    return "error: %s" % msg
                refused_in_a_row += 1
                A.log("  no clip from this request (%s) — its %d segments wait" % (msg, len(g)))
                for x, _ in g:
                    r = self.rec(x)
                    r["errors"] = r.get("errors", 0) + 1
                    r["last_error"] = msg
                self.save()
                if refused_in_a_row >= 3:
                    return "error: three requests in a row gave no audio (%s)" % msg
                continue
            refused_in_a_row = 0
            n += 1
        self.save()
        return "done (%d requests)" % n

    def run(self, order):
        langs = ("ar", "en") if self.a.lang == "all" else (self.a.lang,)
        return self.tts_round(order, langs, first=self.a.first)

    # ── the batch size, MEASURED before it is adopted (the owner, 2026-10-17) ────────────────────────────────────
    def calibrate(self, order, sizes=None):
        """The first Gemini day asks the SAME CAL_SHARED segments at every size of CAL_SIZES: alone as one request of
        CAL_SHARED (their bank take 1), and spread evenly through a request of N (CAL_SHARED of them + N - CAL_SHARED
        new segments, whose pieces are their take 1). Every piece is judged by the same judge (the STT read-back +
        the phone listener); the largest N whose shared segments pass within CAL_TOL of the CAL_SHARED request — and
        whose request as a whole does too — becomes the build's batch (cfg batch_ar). Resumable; decided once."""
        sizes = sorted(sizes or CAL_SIZES)
        cal = self.s.setdefault("calibration", {})
        if cal.get("decision"):
            return "decided: %d segments a request" % cal["decision"]["batch"]
        plans = cal.setdefault("plans", {})
        if not plans:
            q = [x for x, t in self.queue(order, ("ar",)) if t == 1]
            need = CAL_SHARED + sum(n - CAL_SHARED for n in sizes if n != CAL_SHARED)
            if len(q) < need:
                cal["decision"] = {"batch": int(self.cfg["batch_ar"]), "why": "too few segments left to measure"}
                self.save()
                return "decided: %d (too few segments left to measure)" % int(self.cfg["batch_ar"])
            shared, rest = q[:CAL_SHARED], q[CAL_SHARED:]
            for n in sizes:
                if n == CAL_SHARED:
                    plans[str(n)] = {"segs": [x["seg"] for x in shared], "calib": [], "made": False}
                    continue
                seq, rest = list(rest[:n - CAL_SHARED]), rest[n - CAL_SHARED:]
                step = n / float(CAL_SHARED)
                for i, x in enumerate(shared):             # spread through the request: front, middle and end
                    seq.insert(min(len(seq), int(round(i * step))), x)
                plans[str(n)] = {"segs": [x["seg"] for x in seq], "calib": [x["seg"] for x in shared], "made": False}
            self.save()
        bykey = {x["seg"]: x for x in order if x["lang"] == "ar"}
        self.s["run"] = datetime.datetime.now().isoformat(timespec="seconds")
        for n in sizes:
            p = plans[str(n)]
            if p["made"]:
                continue
            group = [bykey[g] for g in p["segs"] if g in bykey]
            A.log("== %s: calibration request of %d segments (%d shared)" % (self.id, len(group), len(p["calib"])))
            try:
                res = self.make(group, takes=[1] * len(group), calib=p["calib"])
            except _exhausted() as e:
                self.save()
                return "quota: %s" % A.scrub(e)[:160]
            p["made"] = True
            p["request"] = res["req"]
            self.save()
        for _ in range(3):
            jr = self.judge(order=order, langs=("ar",))
            if jr != "done":
                return "waiting for verdicts (%s)" % jr
            left = self.judge_calibration()
            if left:
                return "waiting for %d calibration verdicts" % left
            if not self.realign():               # a request cut wrong is cut again and heard again first
                break
        return self.decide(sizes)

    def judge_calibration(self):
        """The calibration pieces through the same gates and judge as a take. → how many still wait."""
        from tools.voice_bank import stt as STT
        left = 0
        for seg, by in sorted(self.s.get("calibration", {}).get("pieces", {}).items()):
            for size, t in sorted(by.items()):
                if t.get("verdict") is not None:
                    continue
                why = t.get("gate") or (None if t["pace_ok"] else "pace %.3f s per letter" % t["s_per_letter"])
                if why:
                    t.update(verdict={"skipped": why}, **{"pass": False})
                    continue
                if self.need_phones and not A.PHONES["on"]:
                    left += 1
                    continue
                try:
                    v = STT.verdict(A.in_context(A.mp3_to_wav(os.path.join(self.dir, t["clip"]))), "ar")
                except (STT.Exhausted, STT.SttError):
                    left += 1
                    continue
                self.verdict({"seg": seg, "lang": "ar", "cat": "calibration", "takes": [t], "status": "made"}, v)
                left += t.get("verdict") is None
        self.save()
        return left

    def decide(self, sizes):
        cal = self.s["calibration"]
        plans = cal["plans"]

        def first_pass(seg):
            r = self.s["segs"].get("ar|" + seg)
            return bool(r and r["takes"] and r["takes"][0].get("pass"))

        shared = plans[str(CAL_SHARED)]["segs"]
        base = sum(1 for g in shared if first_pass(g)) / float(len(shared))
        req0 = plans[str(CAL_SHARED)].get("request") or {}
        rows = {str(CAL_SHARED): {"shared_pass": round(base, 3), "all_pass": round(base, 3), "cut_ok": req0.get("cut_ok"),
                                  "recovered": req0.get("recovered"), "audio_s": req0.get("audio_s"),
                                  "first_audio_s": req0.get("first_audio_s"), "latency_s": req0.get("latency_s")}}
        best = CAL_SHARED
        for n in sizes:
            if n == CAL_SHARED:
                continue
            p = plans[str(n)]
            sh = [cal.get("pieces", {}).get(g, {}).get(str(len(p["segs"]))) for g in p["calib"]]
            sh_ok = sum(1 for t in sh if t and t.get("pass"))
            new = [g for g in p["segs"] if g not in set(p["calib"])]
            new_ok = sum(1 for g in new if first_pass(g))
            sh_rate = sh_ok / float(len(p["calib"]))
            all_rate = (sh_ok + new_ok) / float(len(p["segs"]))
            req = p.get("request") or {}
            ok = bool(req.get("cut_ok")) and sh_rate >= base - CAL_TOL and all_rate >= base - CAL_TOL
            rows[str(n)] = {"shared_pass": round(sh_rate, 3), "new_pass": round(new_ok / float(max(1, len(new))), 3),
                            "all_pass": round(all_rate, 3), "cut_ok": req.get("cut_ok"), "holds": ok,
                            "recovered": req.get("recovered"), "audio_s": req.get("audio_s"),
                            "first_audio_s": req.get("first_audio_s"), "latency_s": req.get("latency_s")}
            if ok:
                best = max(best, n)
        cal["decision"] = {"batch": best, "base_shared_pass": round(base, 3), "rows": rows, "tolerance": CAL_TOL,
                           "at": datetime.datetime.now().isoformat(timespec="seconds")}
        self.cfg["batch_ar"] = best
        self.save()
        A.log("== %s: batch size decided — %d segments a request; %s" % (self.id, best, json.dumps(rows)))
        return "decided: %d segments a request" % best

    # ── the gates ────────────────────────────────────────────────────────────────────────────────────────────
    def judge(self, again=False, order=None, langs=("ar", "en")):
        """Every last take without a verdict, most important first — passes until nothing is left to hear (a second
        listen is due in the pass after the first). → why it stopped ("done" / "quota: …")."""
        A.LIM = A.LIM or A.Limits()
        A.JUDGE_CHAIN = BANK_JUDGES
        if again:                                  # --again: every last take is heard anew (a bad judge day)
            for r in self.s["segs"].values():
                if r["takes"] and r["status"] in ("pass", "fail", "listen"):
                    r["takes"][-1].update(verdict=None, phones=None, listens=[], **{"pass": None})
                    r["status"] = "made"
        rank = {(x["lang"], x["seg"]): x for x in (order or [])}
        res = "done"
        for _ in range(2 + max(RELISTENS.values()) + MAX_DISCARDS):
            res, applied = self._judge_pass(rank, langs)
            if res != "done":
                break
            if self.judge_mode == "stt" and "ar" in langs and self.realign():
                continue                           # a re-cut request's pieces are heard again
            if not applied:
                break
        return res

    # ── a batched take whose pieces read as their NEIGHBOURS' lines: cut it again by what was said ────────────
    def _slots(self, key):
        """[(pos, segment record or None, the piece dict)] of every take / calibration piece cut from request `key`."""
        out = []
        for r in self.s["segs"].values():
            for t in r["takes"]:
                if t.get("req") == key:
                    out.append((t["pos"], r, t))
        for seg, by in (self.s.get("calibration") or {}).get("pieces", {}).items():
            for t in by.values():
                if t.get("req") == key:
                    out.append((t["pos"], {"seg": seg, "lang": "ar", "cat": "calibration"}, t))
        return sorted(out, key=lambda x: x[0])

    def realign(self):
        """Every judged Arabic batch request with >= REALIGN_MIN shifted pieces (a piece that does not read as its own
        line but reads as the line before or after it — the aligned cut slipped by a line) is cut again by the
        transcript (align.transcript_cut, on the request's own cached audio); its pieces are replaced and heard again.
        Once per request. → how many requests were re-cut."""
        from tools.voice_bank import stt as STT
        n = 0
        for req in self.s["requests"]:
            if req.get("lang") != "ar" or not req.get("segs") or req.get("realigned") or not req.get("key"):
                continue
            slots = self._slots(req["key"])
            if not slots or any(t.get("verdict") is None for _, _, t in slots):
                continue
            segs = req["segs"]

            def heard(t):
                ls = t.get("listens") or []
                return ls[-1]["transcript"] if ls else None
            shifted = 0
            for pos, r, t in slots:
                h = heard(t)
                if not h or t.get("pass"):
                    continue
                own = read_back_stt(r, h)["text_ok"]
                if own:
                    continue
                for j in (pos - 1, pos + 1):
                    if 0 <= j < len(segs) and _bare(segs[j]) != _bare(r["seg"]) and read_back_stt(
                            {"lang": "ar", "seg": segs[j], "cat": "?"}, h)["text_ok"]:
                        shifted += 1
                        break
            if shifted < REALIGN_MIN:
                continue
            wav_p = os.path.join(A.CACHE, req["key"] + ".wav")
            if not os.path.exists(wav_p):
                req["realigned"] = {"skipped": "the request's audio is not in the cache"}
                continue
            try:
                pieces, rep = AL.transcript_cut(open(wav_p, "rb").read(), segs, "ar")
            except (STT.Exhausted, STT.SttError) as e:
                A.log("  realign: %s" % A.scrub(e)[:160])
                return n
            A.log("  realign %s: %d pieces read as a neighbour's line — transcript cut matched %d of %d" % (
                req["key"][:12], shifted, rep["matched"], len(segs)))
            req["realigned"] = {"shifted": shifted, "matched": rep["matched"], "islands": rep["islands"],
                                "added_islands": rep["added_islands"], "missing": rep["missing"]}
            for pos, r, t in slots:
                pc = pieces[pos] if pos < len(pieces) else None
                old = os.path.join(self.dir, t["clip"])
                for f in (old, old + ".heard.mp3"):
                    if os.path.exists(f):
                        os.remove(f)
                if pc is None:
                    t.update(verdict={"skipped": "not found in the take by the transcript cut"}, listens=[],
                             phones=None, **{"pass": False})
                    if r.get("takes") and r["takes"] and r["takes"][-1] is t:
                        r["status"] = "fail"
                    continue
                st = acoustics.clip_stats(pc, r["seg"])
                spl = st["speech_s"] / spoken_letters(r["seg"], "ar")
                A.encode_mp3(A.processed(pc), old)
                x, rate = acoustics.pcm(pc)
                t.update(s_per_letter=round(spl, 4), pace_ok=PACE_MIN <= spl <= PACE_MAX, f0=st["f0_med"],
                         audio_s=round(len(x) / float(rate), 2), recovered=True, verdict=None, listens=[],
                         phones=None, gate=None, **{"pass": None})
                if r.get("takes") and r["takes"][-1] is t:
                    r["status"] = "made"
            n += 1
            self.save()
        return n

    def _judge_pass(self, rank, langs):
        applied = 0
        pend = [r for r in self.s["segs"].values()
                if r["lang"] in langs and r["takes"] and r["takes"][-1].get("verdict") is None]
        # the local gates first: no judge request for a clip that cannot pass
        heard = []
        for r in pend:
            t = r["takes"][-1]
            why = t.get("gate") or (None if t["pace_ok"] else "pace %.3f s per letter" % t["s_per_letter"])
            if why:
                t.update(verdict={"skipped": why}, **{"pass": False})
                r["status"] = "fail"
            else:
                heard.append(r)
        self.save()
        heard.sort(key=lambda r: prio(rank.get((r["lang"], r["seg"])) or r))
        if not heard:
            return "done", 0
        if self.judge_mode == "stt":
            return self._stt_pass(heard, langs)
        A.log("== %s: judging %d clips (%d per request; judges %s)" % (self.id, len(heard), JUDGE_CLIPS,
                                                                       ",".join(BANK_JUDGES)))
        for lang in ("ar", "en"):
            if lang not in langs:
                continue
            if lang == "ar" and self.need_phones and not A.PHONES["on"]:
                A.log("  the phone listener is off (%s): Arabic verdicts wait" % A.PHONES.get("err"))
                continue
            buckets = {}
            for r in heard:
                if r["lang"] == lang:
                    ex = frozenset(l.get("model") for l in r["takes"][-1].get("listens") or [])
                    buckets.setdefault(ex, []).append(r)
            for ex, xs in sorted(buckets.items(), key=lambda kv: prio(kv[1][0])):
                for i in range(0, len(xs), JUDGE_CLIPS):
                    chunk = xs[i:i + JUDGE_CLIPS]
                    try:
                        vs, model = self.hear(lang, chunk, ex)
                    except A.Exhausted as e:
                        A.log("  judge stopped: %s" % A.scrub(e)[:160])
                        self.save()
                        return "quota: %s" % A.scrub(e)[:160], applied
                    why = unreliable(chunk, vs)
                    if why and min(r["takes"][-1].get("discards", 0) for r in chunk) < MAX_DISCARDS:
                        A.log("  judge batch thrown away (%s): %s" % (why, model))
                        for r, m in zip(chunk, self.heard_mp3s(lang, chunk)):
                            r["takes"][-1]["discards"] = r["takes"][-1].get("discards", 0) + 1
                            vp = os.path.join(A.CACHE, "verdict_%s.json" % A._verdict_key(
                                lang, m, None, model if ex else self.a.judge_model))
                            if os.path.exists(vp):     # and the per-clip verdict cache forgets them
                                os.remove(vp)
                        applied += 1
                        self.save()
                        continue
                    for r, v in zip(chunk, vs):
                        n0 = len(r["takes"][-1].get("listens") or [])
                        self.verdict(r, v)
                        applied += len(r["takes"][-1].get("listens") or []) > n0
                        if lang == "ar" and self.need_phones and not A.PHONES["on"]:
                            break
                    self.save()
                    if lang == "ar" and self.need_phones and not A.PHONES["on"]:
                        A.log("  the phone listener stopped (%s): the rest of the Arabic verdicts wait"
                              % A.PHONES.get("err"))
                        break
                else:
                    continue
                break
        self.save()
        return "done", applied

    def _stt_pass(self, heard, langs):
        """The quota-free judge: one Azure STT read-back per clip (Arabic heard INSIDE a line, behind the bank's lead
        clip — the same audio the Gemini judge heard), the phone listener for Arabic in verdict()."""
        from tools.voice_bank import stt as STT
        applied = 0
        A.log("== %s: judging %d clips (Azure speech-to-text read-back%s)" % (
            self.id, len(heard), " + the blind phone listener" if self.need_phones else ""))
        for k, r in enumerate(heard):
            if r["lang"] not in langs:
                continue
            if r["lang"] == "ar" and self.need_phones and not A.PHONES["on"]:
                A.log("  the phone listener is off (%s): Arabic verdicts wait" % A.PHONES.get("err"))
                continue
            try:
                clip = os.path.join(self.dir, r["takes"][-1]["clip"])
                # English: the clip itself — and, for a clip the recogniser could not read alone, a second listen
                # INSIDE a line (behind the bank's own "In 300 meters," clip: a one-word clip alone is often heard
                # as nothing or as a homophone). Arabic: always inside the line (the lead clip + VoiceBank's pause).
                # WAV, no second MP3 generation for a recogniser.
                ctx = r["lang"] == "ar" or bool(r["takes"][-1].get("listens"))
                if r["lang"] == "en":
                    audio = self.en_in_context(clip) if ctx else open(clip, "rb").read()
                else:
                    audio = A.in_context(A.mp3_to_wav(clip))
                v = dict(STT.verdict(audio, r["lang"]), context=ctx)
            except STT.Exhausted as e:
                self.save()
                return "quota: %s" % A.scrub(e)[:160], applied
            except STT.SttError as e:
                self.save()
                return "error: %s" % A.scrub(e)[:160], applied
            n0 = len(r["takes"][-1].get("listens") or [])
            self.verdict(r, v)
            applied += len(r["takes"][-1].get("listens") or []) > n0
            if k % 50 == 49:
                self.save()
                A.log("  %d / %d heard" % (k + 1, len(heard)))
        self.save()
        return "done", applied

    def en_in_context(self, clip):
        """An English clip behind this bank's own lead clip EN_LEAD, with VoiceBank's comma pause (as a line)."""
        import numpy as np
        if "_en_lead" not in self.__dict__:
            lr = self.s["segs"].get("en|" + EN_LEAD)
            t = (self.published_take(lr) or lr["takes"][-1]) if lr and lr["takes"] else None
            self._en_lead = acoustics.pcm(A.mp3_to_wav(os.path.join(self.dir, t["clip"]))) if t else None
        x, rate = acoustics.pcm(A.mp3_to_wav(clip))
        if not self._en_lead:
            return build_bank_wav(x, rate)
        gap = np.zeros(int(rate * mv.pause_after(EN_LEAD) / 1000.0), dtype=np.float32)
        return build_bank_wav(np.concatenate([np.asarray(self._en_lead[0], dtype=np.float32), gap,
                                              np.asarray(x, dtype=np.float32)]), rate)

    def heard_mp3s(self, lang, chunk):
        out = []
        for r in chunk:
            t = r["takes"][-1]
            p = os.path.join(self.dir, t["clip"] + ".heard.mp3")
            if not os.path.exists(p):
                wav = A.mp3_to_wav(os.path.join(self.dir, t["clip"]))
                if lang == "ar":
                    wav = A.in_context(wav)
                A.encode_mp3(wav, p)
            out.append(open(p, "rb").read())
        return out

    def hear(self, lang, chunk, exclude):
        """One judge request for `chunk` → (verdicts, model). A first listen goes through the bank's chain; a second
        listen goes to a judge that has not heard these clips yet."""
        mp3s = self.heard_mp3s(lang, chunk)
        if not exclude:
            vs = A.judge_batch(lang, mp3s, force=self.a.judge_model)
            return vs, (vs[0] or {}).get("_judge_model") if vs else None
        last = None
        for m in BANK_JUDGES:
            if m in exclude:
                continue
            try:
                return A.judge_batch(lang, mp3s, force=m), m
            except A.Exhausted as e:
                last = e
        raise A.Exhausted("no second judge left (%s)" % A.scrub(last))

    def f0_ok(self, r, t):
        """§f.9: the piece's median pitch against the voice's (the passed clips', else every take's)."""
        if self.cfg["provider"] != "gemini" or not t.get("f0"):
            return True
        same = [r2["takes"][-1] for r2 in self.s["segs"].values() if r2["lang"] == r["lang"] and r2["takes"]]
        ref = [x["f0"] for x in same if x.get("pass") and x.get("f0")]
        if len(ref) < F0_MIN_REF:
            ref = [x["f0"] for x in same if x.get("f0")]
        if len(ref) < F0_MIN_REF:
            return True
        cents = 1200.0 * math.log2(t["f0"] / statistics.median(ref))
        t["f0_cents"] = round(cents)
        return abs(cents) <= F0_DRIFT_CENTS

    def verdict(self, r, v):
        t = r["takes"][-1]
        ph = t.get("phones")
        if r["lang"] == "ar" and self.need_phones and ph is None:
            wav = A.in_context(A.mp3_to_wav(os.path.join(self.dir, t["clip"])))
            ph = A.phones_verdict(read_line(r)["plain"], wav)
            if ph is None:                         # the listener could not run: the judge's verdict is cached,
                return                             # this clip is heard again by the next run
            t["phones"] = ph
        stt = str(v.get("_judge_model") or "").startswith("azure-stt")
        rr = dict(r, spoken=t["spoken"]) if t.get("spoken") else r      # an overridden segment: its SPOKEN form
        chk = (read_back_stt(rr, v.get("transcript", ""), v.get("itn", ""), lead=EN_LEAD if v.get("context") else None)
               if stt else read_back(r, v.get("transcript", "")))
        # the STT read-back hears words, not the accent: for Arabic the dialect is the phone listener's (no hard flag)
        dialect = (None if stt else A.dialect_ok(v)) if r["lang"] == "ar" else None
        t.setdefault("listens", []).append({"model": v.get("_judge_model"), "transcript": v.get("transcript", ""),
                                            "check": chk, "dialect_ok": dialect, "judge": v})
        best = min(t["listens"], key=lambda l: (not l["check"]["text_ok"], l["dialect_ok"] is False,
                                                len(l["check"].get("critical") or []),
                                                l["check"].get("cer", l["check"].get("wer", 1.0))))
        judge_ok = bool(best["check"]["text_ok"] and (r["lang"] != "ar" or best["dialect_ok"] is not False)
                        and (r["lang"] != "ar" or best["dialect_ok"] is not None or
                             str(best.get("model") or "").startswith("azure-stt")))
        local_ok = bool(t["pace_ok"] and not (ph or {}).get("hard") and self.f0_ok(r, t))
        # the STT read-back is deterministic: English gets ONE second listen, inside a line (en_in_context)
        relistens = (1 if r["lang"] == "en" else 0) if stt else RELISTENS.get(self.cfg["provider"], 0)
        if local_ok and not judge_ok and len(t["listens"]) <= relistens:
            r["status"] = "made"                   # heard once more, by another judge, before it stands
            return
        t["verdict"] = {"judge": best["judge"], "check": best["check"], "dialect_ok": best["dialect_ok"],
                        "listens": len(t["listens"]), "f0_cents": t.get("f0_cents")}
        t["pass"] = judge_ok and local_ok
        r["status"] = "pass" if t["pass"] else "fail"

    def settle(self):
        """Fails that cannot be retaken (MAX_TAKES, Azure's determinism, MAX_ERRORS) wait for a human listen."""
        n = 0
        for r in self.s["segs"].values():
            if (r["status"] == "fail" and not self.can_retake(r)) or (
                    r["status"] == "pending" and r.get("errors", 0) >= MAX_ERRORS):
                r["status"] = "listen"
                n += 1
        self.save()
        return n

    def retake(self, order):
        """The failed clips again — Gemini batched with the rest of the queue's retakes (the next SAY spelling)."""
        self.settle()
        langs = ("ar", "en") if self.a.lang == "all" else (self.a.lang,)
        retakes = [(x, t) for x, t in self.queue(order, langs) if t > 1]
        if not retakes:
            A.log("== 0 retakes")
            return "done"
        keep = {(x["lang"], x["seg"]) for x, _ in retakes}
        sub = [x for x in order if (x["lang"], x["seg"]) in keep]
        return self.tts_round(sub, langs)

    def status(self):
        segs = list(self.s["segs"].values())
        st = {}
        for r in segs:
            st.setdefault(r["lang"], {}).setdefault(r["status"], 0)
            st[r["lang"]][r["status"]] += 1
        chars = sum(q.get("chars", 0) for q in self.s["requests"])
        return {"build": self.id, "cfg": self.cfg, "segments": st, "requests": len(self.s["requests"]),
                "chars": chars, "cut_failures": sum(1 for q in self.s["requests"] if not q["cut_ok"])}

    def published_take(self, r):
        """The take a release ships for a segment: its LATEST passing take (a later retake replaces an earlier one); a
        REDO segment (en_say.REDO) with no new passing take yet keeps shipping the take it had."""
        ok = [t for t in r["takes"] if t.get("pass")]
        return ok[-1] if ok else r.get("fallback")

    def export(self):
        from tools.voice_bank.publish import bank_frames      # the published form: bare frames, the bank's header
        rel = os.path.join(self.dir, "release")
        man = {"build": self.id, "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "phrasebook": self.s["phrasebook"], "provider": self.cfg["provider"],
               "voices": {"ar": self.cfg["voice_ar"], "en": self.cfg["voice_en"]}, "settings": self.cfg,
               "format": "MPEG-2 L3 24 kHz mono 32 kbps, bare frames", "files": {}}
        for r in self.s["segs"].values():
            t = self.published_take(r)
            if not t:
                continue
            name = "%s/%s.mp3" % (r["lang"], mv.fnv64(r["seg"]))
            dst = os.path.join(rel, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(bank_frames(open(os.path.join(self.dir, t["clip"]), "rb").read()))
            man["files"][name] = {"seg": r["seg"], "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                                  "take": t["take"], "cat": r["cat"], "rank": r["rank"]}
        io.open(os.path.join(rel, "manifest.json"), "w", encoding="utf-8").write(
            json.dumps(man, ensure_ascii=False, indent=1))
        return len(man["files"]), rel


def read_line(r):
    lead = A.CONTEXT_LEAD + " " if r["lang"] == "ar" else ""
    return {"lang": r["lang"], "plain": lead + r["seg"], "fem_plain": lead + A.fem_of(r["seg"])}


def read_back(r, heard):
    """The judge's transcript against the segment (Arabic: behind the bank's lead clip, or the segment alone when the
    judge wrote it without the lead)."""
    chk = A.text_check(read_line(r), heard or "")
    if not chk["text_ok"] and r["lang"] == "ar":
        alone = A.text_check({"lang": "ar", "plain": r["seg"], "fem_plain": A.fem_of(r["seg"])}, heard or "")
        if alone["text_ok"]:
            chk = dict(alone, read="segment alone")
    return chk


_EN_FOLD = [("wanna", "want to"), ("gonna", "going to"), ("on to", "onto"), ("st", "street"), ("rd", "road"), ("you turn", "u turn"), ("u-turn", "u turn"), ("okay", "ok"), ("o k", "ok"), ("km", "kilometers"),
            ("kilometre", "kilometer"), ("i will", "i'll"), ("you have", "you've"), ("do not", "don't")]


def _en_norm(s):
    w = " ".join(A.en_words(s))
    for a, b in _EN_FOLD:
        w = re.sub(r"\b%s\b" % re.escape(a), b, w)
    return w.split()


def _skeleton(words):
    """A transliterated name's consonant skeleton — how it SOUNDS, not how a recogniser spells it ("Kornish Al Nil"
    / "cornish el neil" → krnshlnl): letters only, digraphs folded, vowels / h / y / w and doubled letters gone."""
    s = "".join(words).lower()
    for a, b in (("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("th", "t"),
                 ("dh", "d"), ("z", "s"), ("x", "ks"), ("j", "g"), ("v", "f"), ("'", "")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"[aeiouyhw]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# words that SOUND the same — a recogniser cannot tell them apart, so neither can the gate ("continue write")
EN_HOMOPHONES = [{"right", "write", "rite", "wright"}, {"four", "for", "fore"}, {"two", "to", "too"},
                 {"bear", "bare"}, {"one", "won"}, {"eight", "ate"}, {"meters", "meter's"}, {"route", "root"},
                 {"you're", "your"}, {"there", "their"}, {"weigh", "way"}]
_HOMO = {w: frozenset(g) for g in EN_HOMOPHONES for w in g}
EN_TAIL_FUNCTION = {"in", "about", "at", "to", "on", "the", "a", "of", "for", "by"}


def _tok_eq(a, b):
    """Two words a recogniser may write for the same spoken word: equal, homophones (EN_HOMOPHONES), a plural -s
    ("limit" / "limits"), or one edit apart on a word of ≥ 4 letters ("ahead" / "head"). Other than homophones, the
    critical words (en_critical) are matched exactly."""
    if a == b or (a in _HOMO and b in _HOMO[a]):
        return True
    if a in EN_CRITICAL_WORDS or b in EN_CRITICAL_WORDS:
        return False
    if a.rstrip("s") == b.rstrip("s"):
        return True
    return min(len(a), len(b)) >= 4 and abs(len(a) - len(b)) <= 1 and _lev(a, b) <= 1


def fuzzy_wer(a, b):
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (not _tok_eq(ca, cb))))
        prev = cur
    return prev[-1] / float(len(a))


def en_critical(src, heard):
    """English direction words / ordinals / U-turn / exit: each the segment says must be heard; right ⇄ left never
    swapped."""
    heard = [next((w for w in _HOMO.get(x, ()) if w in EN_CRITICAL_WORDS), x) for x in heard]
    t, h = " " + " ".join(src) + " ", " " + " ".join(heard) + " "
    probs = ["lost " + w for w in EN_CRITICAL if (" %s " % w) in t and (" %s " % w) not in h]
    for a, b in (("right", "left"), ("left", "right")):
        if " %s " % a in t and " %s " % b not in t and " %s " % b in h:
            probs.append("swapped %s→%s" % (a, b))
    return probs


def build_bank_wav(x, rate):
    from tools.masri_voice import build_bank
    return build_bank.wav_bytes(x, rate)


def _read_back_en(r, heard):
    src, got = _en_norm(r.get("spoken") or r["seg"]), _en_norm(heard)
    crit = en_critical(src, got)
    if r.get("cat") == "road":
        # "onto <name> <type>": the TYPE word heard; everything before it (the recogniser glues "onto" to the name:
        # "aunt taha hussein", "entourage garab") by its consonant skeleton
        typ = src[-1] if src and src[-1] in EN_ROAD_TYPES else None
        rest_s = src[:-1] if typ else src
        rest_h = list(got)
        if typ and typ not in rest_h:
            # a recogniser glues the type word to the name ("talat harbaxis", "al ganzoriakis"): split it off
            for i in range(len(rest_h) - 1, -1, -1):
                w = rest_h[i]
                if len(w) > len(typ) + 1 and _lev(w[-len(typ):], typ) <= (0 if len(typ) <= 4 else 1):
                    rest_h[i:i + 1] = [w[:-len(typ)], typ]
                    break
            got = rest_h
        if typ and typ in rest_h:
            rest_h = list(rest_h)
            del rest_h[len(rest_h) - 1 - rest_h[::-1].index(typ)]
        ks, kh = _skeleton(rest_s), _skeleton(rest_h)
        ncer = _lev(ks, kh) / float(max(1, len(ks)))
        typ_ok = typ is None or typ in got
        ok = not crit and typ_ok and ncer <= EN_NAME_SKELETON_CER
        return {"wer": round(fuzzy_wer(src, got), 3), "frame_wer": 0.0 if typ_ok else 1.0, "name_cer": round(ncer, 3),
                "skeleton": [ks, kh], "critical": crit, "text_ok": ok,
                "problems": crit + ([] if typ_ok else ["lost %s" % typ])
                + ([] if ncer <= EN_NAME_SKELETON_CER else ["name skeleton %s != %s (%.2f)" % (ks, kh, ncer)])}
    if len(src) > 1 and src[-1] in EN_TAIL_FUNCTION and (not got or not _tok_eq(got[-1], src[-1])):
        src = src[:-1]          # a fragment's last function word ("Road incident in") is swallowed by a recogniser
    w = fuzzy_wer(src, got)
    errors = round(w * len(src))
    # one ordinary word a recogniser got wrong is allowed in a segment of 4 words or more (the critical words never)
    ok = not crit and (w <= A.WER_PASS or (len(src) >= 4 and errors <= 1))
    return {"wer": round(w, 3), "critical": crit, "text_ok": ok,
            "problems": crit + ([] if ok or crit else ["WER %.2f" % w])}


def _ar_resplit(heard, seg):
    """A recogniser's word boundary is not the voice's: «خليك يمين» (khallīk yamīn) is the same sound as «خليكي مين»
    — which would read as the FEMININE «خليكي». When a heard word ending in ي and the next word are both the
    segment's words once the ي moves across the space, it moves."""
    exp = set(A.ar_words(seg))
    ws = (heard or "").split()
    for i in range(len(ws) - 1):
        a, b = ws[i], ws[i + 1]
        if a.endswith("ي") and len(a) > 2 and A.ar_words(a[:-1]) and A.ar_words(a[:-1])[0] in exp and A.ar_words(
                "ي" + b) and A.ar_words("ي" + b)[0] in exp and A.ar_words(a)[0] not in exp:
            ws[i], ws[i + 1] = a[:-1], "ي" + b
    return " ".join(ws)


def read_back_stt(r, heard, itn="", lead=None):
    """The Azure STT read-back against the segment. English: the lexical words AND the recogniser's ITN form (digits),
    both through the bank's number-as-words fold, the better of the two — "200" = "two hundred", "1100" = "eleven
    hundred" = "one thousand one hundred"; case and punctuation never count; a recogniser's near-miss on an ordinary
    word is no error (`_tok_eq`); the critical words are exact (en_critical); WER <= WER_PASS. A ROAD segment: its frame
    words exact, its transliterated name by consonant skeleton (EN_NAME_SKELETON_CER). Arabic: the bank's text_check
    (folds, CER, direction words / ordinals, the feminine forms) — on the whole line (lead + segment) or on the segment
    with the lead's words taken off the front of the transcript (a recogniser writes «بعد تلتمية متر» as one, two or
    four words), whichever reads best."""
    heard = heard or ""
    if r["lang"] == "en":
        cands = []
        for h in sorted({heard, itn or heard}):
            ws = h.split()
            # heard inside a line: the lead's words come first (a recogniser writes "In 300 meters," as 3-5 words)
            for k in (range(0, min(6, len(ws) + 1)) if lead else (0,)):
                c = _read_back_en(r, " ".join(ws[k:]))
                c["read"] = "segment after %d heard words" % k if lead else "alone"
                cands.append(c)
        return min(cands, key=lambda c: (not c["text_ok"], c.get("frame_wer", 0.0) + c.get("name_cer", 0.0),
                                         c["wer"]))
    heard = _ar_resplit(heard, r["seg"])
    best = A.text_check(read_line(r), heard)
    best["read"] = "line"
    words = heard.split()
    seg_line = {"lang": "ar", "plain": r["seg"], "fem_plain": A.fem_of(r["seg"])}
    for k in range(0, min(6, len(words))):
        c = A.text_check(seg_line, " ".join(words[k:]))
        if (c["text_ok"], -c["cer"]) > (best["text_ok"], -best["cer"]):
            best = dict(c, read="segment after %d heard words" % k)
    return best


def _bare(seg):
    """A segment's words without the punctuation that only sets the pause after it."""
    return " ".join("".join(" " if ch in "،,.!?؟؛:…" else ch for ch in seg).split()).lower()


def unreliable(chunk, vs):
    """Why a judge batch cannot be trusted (None = it can): fewer verdicts than clips, the same transcript given to
    two different segments, or transcripts SHIFTED by a clip (two clips that read as another clip's segment and not
    as their own) — the multi-clip protocol broke (2026-10-17's flash-lite answers)."""
    if len(vs) < len(chunk):
        return "%d verdicts for %d clips" % (len(vs), len(chunk))
    seen = {}
    for r, v in zip(chunk, vs):
        t = " ".join(((v or {}).get("transcript") or "").split())
        if t and t in seen and seen[t] != _bare(r["seg"]):     # «يمين» and «يمين،» are the same words
            return "one transcript for two clips"
        seen[t] = _bare(r["seg"])
    if len(chunk) > 2:
        shifted = 0
        for i, (r, v) in enumerate(zip(chunk, vs)):
            h = (v or {}).get("transcript") or ""
            if not h or read_back(r, h)["text_ok"]:
                continue
            if any(read_back(o, h)["text_ok"] for j, o in enumerate(chunk)
                   if j != i and _bare(o["seg"]) != _bare(r["seg"])):
                shifted += 1
        if shifted >= 2:
            return "%d transcripts read as another clip's segment" % shifted
    return None


def plan(a, order):
    """Characters (Azure) or requests (Gemini) and days for this configuration — no network."""
    from tools.voice_audition import azure as AZR
    rows = {}
    for lang in ("ar", "en"):
        if a.lang != "all" and lang != a.lang:
            continue
        xs = sorted([x for x in order if x["lang"] == lang], key=prio)
        if a.first:
            firsts = {x["seg"] + x["lang"] for x in order[:a.first]}
            xs = [x for x in xs if x["seg"] + x["lang"] in firsts]
        k = max(1, a.batch_ar if lang == "ar" else a.batch_en)
        reqs = [xs[i:i + k] for i in range(0, len(xs), k)]
        cfg = {"voice_ar": a.voice_ar, "voice_en": a.voice_en, "ar_rate": a.ar_rate, "ar_pitch": a.ar_pitch,
               "en_rate": a.en_rate, "en_pitch": a.en_pitch, "en_style": a.en_style, "sep": a.sep}
        if lang != "ar":
            say = (lambda seg: seg)
        elif a.ar_spell == "pausal" and a.provider == "azure":
            say = (lambda seg: A.pausal(A.az_respell(pronounce.say(seg, 0))))
        else:
            say = (lambda seg: pronounce.say(seg, 0))
        chars = 0
        if a.provider == "azure":
            for g in reqs:
                chars += AZR.billable(az_doc(cfg, [say(x["seg"]) for x in g], lang))
        text = sum(len(say(x["seg"])) for x in xs)
        rows[lang] = {"segments": len(xs), "requests": len(reqs), "text_chars": text, "billable_chars": chars,
                      "markup_chars": max(0, chars - text), "minutes_at_20_per_min": round(len(reqs) / 20.0, 1),
                      "days_at_10_per_day": round(len(reqs) / 10.0, 1)}
    tot = sum(r["billable_chars"] for r in rows.values())
    retake = 1.3
    return {"provider": a.provider, "batch_ar": a.batch_ar, "batch_en": a.batch_en, "sep": a.sep, "by_lang": rows,
            "one_take_chars": tot,
            "with_retakes_chars": int(tot * retake), "pct_of_f0_month": round(100.0 * tot * retake / 500_000, 1),
            "fits_desk_ceiling": tot * retake <= BANK_MONTH_CEILING}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "judge", "retake", "status", "export"])
    ap.add_argument("--provider", default="azure", choices=["azure", "gemini"])
    ap.add_argument("--build", default=None, help="an existing build id (judge / retake / status / export)")
    # the defaults = the audition's recommendation (docs/azure-voice-2026-10.md §5); the owner's picks are banks.py
    ap.add_argument("--voice-ar", default=A.AZ_BANK_AR)
    ap.add_argument("--voice-en", default=A.AZ_EN_PICK[0][0])
    ap.add_argument("--ar-rate", default=A.AZ_AR_FINAL[A.AZ_BANK_AR].get("rate", ""))
    ap.add_argument("--ar-pitch", default=A.AZ_AR_FINAL[A.AZ_BANK_AR].get("pitch", ""))
    ap.add_argument("--ar-spell", default="pausal", choices=["pausal", "say"],
                    help="azure Arabic: pronounce.say + AZ_SAY + the pausal sukun (default), or say only")
    ap.add_argument("--en-rate", default="")
    ap.add_argument("--en-pitch", default="")
    ap.add_argument("--en-style", default="")
    ap.add_argument("--lang", default="all", choices=["ar", "en", "all"])
    ap.add_argument("--first", type=int, default=None, help="only the N most spoken segments (both languages)")
    ap.add_argument("--batch", type=int, default=None, help="segments per request, both languages")
    ap.add_argument("--batch-ar", type=int, default=None,
                    help="Arabic segments per request (azure default AZ_BANK_BATCH: one prosody wrapper per request)")
    ap.add_argument("--batch-en", type=int, default=None,
                    help="English segments per request (azure default 1: English carries no markup to share)")
    ap.add_argument("--sep", default=AZ_BANK_SEP, choices=sorted(A.AZ_SEPS))
    ap.add_argument("--max-chars", type=int, default=None, help="Azure: this run's characters")
    ap.add_argument("--no-phones", action="store_true")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, choices=["stt", "gemini"],
                    help="stt = Azure speech-to-text read-back (+ the phone listener for Arabic); gemini = the judge chain")
    ap.add_argument("--judge-model", default=None, help="judge: one Gemini judge model (default: BANK_JUDGES)")
    ap.add_argument("--again", action="store_true", help="judge: hear every last take again (a bad judge day)")
    a = ap.parse_args()
    dflt = {"azure": (AZ_BANK_BATCH, 1), "gemini": (GEMINI_BATCH, GEMINI_BATCH)}[a.provider]
    a.batch_ar = a.batch_ar or a.batch or dflt[0]
    a.batch_en = a.batch_en or a.batch or dflt[1]
    order, _, _ = INV.ranked()
    if a.cmd == "plan":
        print(json.dumps(plan(a, order), ensure_ascii=False, indent=1))
        return
    if a.no_phones:
        A.PHONES["on"] = False
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    A.LIM = A.Limits()
    b = Build(a)
    langs = ("ar", "en") if a.lang == "all" else (a.lang,)
    try:
        if a.cmd == "run":
            print(b.run(order))
        elif a.cmd == "judge":
            print(b.judge(again=a.again, order=order, langs=langs))
            b.settle()
        elif a.cmd == "retake":
            print(b.retake(order))
        elif a.cmd == "export":
            n, where = b.export()
            print("exported %d clips → %s" % (n, where))
    except Exception:
        import traceback
        A.log("stopped: " + A.scrub(traceback.format_exc()))
        b.save()
        raise SystemExit(1)
    print(json.dumps(b.status(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
