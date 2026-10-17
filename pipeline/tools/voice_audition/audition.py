# -*- coding: utf-8 -*-
"""Voice audition — RouteMind's navigation voice on the FREE Gemini API (desk only; the phone never calls a TTS).

    python -m tools.voice_audition.audition lines            # the lines and their TTS text (no network)
    python -m tools.voice_audition.audition run [--no-phones] [--stages prescreen,judge,audition,english,compare,
                 raw,batch,retake] [--voices-ar A,B] [--voices-en A,B] [--en-screen-model M] [--test-model M]
    python -m tools.voice_audition.audition page             # rebuild the page from the cache (no network)
    python -m tools.voice_audition.audition phones           # the blind phone listener over every Arabic take (CPU)
    python -m tools.voice_audition.audition recut            # cut every one-request take again (no network)
    python -m tools.voice_audition.audition rejudge --group G --judge-model M   # one judge for a whole group
    python -m tools.voice_audition.audition limits           # what the API's quota answers said
    python -m tools.voice_audition.audition azure [--azure-stages tune,final,screen,english,batch,judge]
                 [--azure-groups G,..] [--judge-batch N] [--judge-model M] [--no-phones]   # the Azure voices
    python -m tools.voice_audition.audition phones-azure     # the blind phone listener over the Azure takes (CPU)

AZURE (2026-10-17, docs/azure-voice-2026-10.md): the free F0 Speech resource (tools/voice_audition/azure.py — key,
character ledger, cap). ar-EG-SalmaNeural and ar-EG-ShakirNeural on every Arabic line with the tuned SSML
(AZ_AR_FINAL: the SAY spellings + AZ_SAY + the pausal sukun `pausal`, rate / pitch), the English voices screened
(AZ_EN_SCREEN) and the five picks (AZ_EN_PICK) on every English
line; one line per request (Azure bills characters, not requests); the same processing, judge and phone listener as
the Gemini takes; the page shows "Gemini voices" and "Azure voices" beside today's bundled clip on every line.

WHAT IT DOES. Pre-screens candidate prebuilt voices on ONE probe line (Arabic and English), ranks them by a judge,
then auditions the picks on the app's REAL lines (Arabic composed by tools/masri_voice/masri_voice.py, masculine
addressee; English in the app's own words), compares models for the best Arabic voice, tests the ق respelling and
N-segments-per-request batching, and writes a listening page with the CURRENT bundled clip (assets/voice/ar, joined
exactly as VoiceBank joins them) beside every new take.

THE FREE TIER IS THE BUDGET (measured 2026-10-16): 10 TTS requests a day per model, 3 a minute on the lite model,
20 judge requests a day per text model. So the pre-screen is one request per voice, but every audition is ALL its
lines in ONE request, cut at the pauses by an aligned cut (each piece matched to its line's letters), and the judge
hears up to JUDGE_BATCH clips per request. Generation and judging are separate passes: a take whose judge is out of
quota keeps its audio and is judged by the next run.

THE TEXT THE VOICE READS is the bank build's: each segment through tools/masri_voice/pronounce.py `say()` (ق written
as the Cairene glottal stop, Egyptian vowels diacritized), the retake rotating to the next spelling exactly as
build_bank does. The steering is a director's-notes prompt — the only form the free models speak cleanly
(`systemInstruction` is refused, a "Say …:" prefix is read aloud): profile, scene, notes, then the transcript. The
accent notes carry the bank's own sound list (tools/masri_voice/tts.py EGY_STYLE_PROMPT); the scene says the driver
is a MAN and every word is spoken as written (a "she sits beside the driver" scene turned «لف» into «لفّي»).

THE JUDGE (every take): gemini-3.8-flash (then JUDGE_CHAIN when it is overloaded or out of quota; the verdict
records which model heard it) hears the take as the bank ships it — the MP3 — BLIND (never told the text), and
returns a transcript + accent / ج / ق / addressee / warmth notes; this file compares the transcript with the source
text (diacritics, alef / ya / ta-marbuta, ق-as-hamza and digits folded; build_bank's direction-word / ordinal check;
the feminine forms of the same line) — a mismatch is regenerated ONCE. Bank segments are judged inside a line, behind
the bank's lead clip, as build_bank judges a direction word. The judge reads back well and ranks voices poorly (it
rated every voice 10/10) — the page is for LISTENING. The blind phone listener (tools/masri_voice/phones.py) runs on
every Arabic take when it loads (--no-phones skips it; `phones` back-fills).

THE KEY lives OUTSIDE the repo ($ROUTEMIND_GEMINI_KEY_FILE, default C:\\Users\\ibrah\\routemind-secrets\\gemini_key.txt).
It is read at runtime, sent only as the `x-goog-api-key` header, and scrubbed from every error printed or stored.

OUTPUT (never in the repo; $VOICE_AUDITION_OUT, default G:/grad_all/.p5work/voice_audition): cache/ — every take
keyed by sha256(model, voice, full prompt text) (+ "take=n" for a retake), every verdict keyed by its own inputs, so a
re-run never re-asks and a quota stop resumes where it stopped; clips/ — the bank-format MP3s the page plays
(24 kHz mono 32 kbps, the takes trimmed and levelled by build_bank.process exactly like a bank clip); limits.json —
every quota answer the API gave; index.html — the page (works from file://).
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

import numpy as np

from tools.masri_voice import acoustics
from tools.masri_voice import build_bank
from tools.masri_voice import masri_voice as mv
from tools.masri_voice import mp3 as mp3mod
from tools.masri_voice import pronounce
from tools.masri_voice import tts as bank_tts

ROOT = mv.ROOT
OUT = os.environ.get("VOICE_AUDITION_OUT", "G:/grad_all/.p5work/voice_audition")
CACHE = os.path.join(OUT, "cache")
CLIPS = os.path.join(OUT, "clips")
KEY_FILE = os.environ.get("ROUTEMIND_GEMINI_KEY_FILE", r"C:\Users\ibrah\routemind-secrets\gemini_key.txt")
BANK_DIR = os.path.join(ROOT, "assets", "voice", "ar")
API = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"

MAIN_MODEL = "gemini-3.8-flash-tts"
COMPARE_MODELS = ["gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview"]
JUDGE_MODEL = "gemini-3.8-flash"
# when the judge is overloaded (503 "high demand", seen 2026-10-16) the next text model hears the take; the verdict
# records which one did (`_judge_model`) and the page shows it
JUDGE_CHAIN = [JUDGE_MODEL, "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3-flash-preview",
               "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]

# Female voices only for Arabic: the phrasebook's first-person lines are a woman's («أنا صاحية معاك»,
# «وأنا حاسباها في الوقت» — nav-spec/masri_voice.json qm.long / qm.jam), so a man's voice would misgender himself.
AR_SCREEN = ["Zephyr", "Sulafat", "Leda", "Kore", "Aoede", "Despina", "Vindemiatrix", "Achernar", "Callirrhoe",
             "Erinome"]
INCUMBENT_AR = "Zephyr"      # the bundled bank's voice (Gemini 2.5 Pro TTS) — always auditioned
# English: the app spoke en-US-Journey-F (warm, natural, female, clear, mid-pace) — the Gemini voices nearest it
EN_SCREEN = ["Aoede", "Kore", "Leda", "Sulafat", "Despina", "Achernar"]
EN_REFS = [("Journey-F (shipped English voice)", "docs/voice-samples/google-journey__en-US-Journey-F/07_en.mp3"),
           ("Chirp3-HD en-US-Aoede", "docs/voice-samples/google-chirp3__en-US-Aoede/07_en.mp3")]
EN_REF_TEXT = "Heavy traffic in about 2 kilometers on Salah Salem — it'll cost you around 6 minutes."

CER_PASS = 0.20     # a whole line's read-back (isolated 1-second clips get 0.35 in the bank; a line is easier)
WER_PASS = 0.15


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE KEY — read at runtime, never written, scrubbed from everything printed or stored
# ════════════════════════════════════════════════════════════════════════════════════════════════
_KEY = {"v": None}
_KEYLIKE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def api_key():
    if _KEY["v"] is None:
        with io.open(KEY_FILE, encoding="utf-8") as fh:
            _KEY["v"] = fh.read().strip()
    return _KEY["v"]


def scrub(s):
    s = str(s)
    if _KEY["v"]:
        s = s.replace(_KEY["v"], "<key>")
    s = _KEYLIKE.sub("<key>", s)
    az = sys.modules.get("tools.voice_audition.azure")      # the Azure key and region, when this run loaded them
    return az.scrub(s) if az else s


def log(*a):
    print(scrub(" ".join(str(x) for x in a)), flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE LINES — the app's own composers, masculine addressee
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _street(lead_m, act_key, street, variant=0):
    """A turn ONTO a named road. The composer has no road-name segment today (the plan proposes «على <name>» as
    one segment per major road) — so this joins the composer's own lead + action with the proposed segment."""
    def build(fem):
        return mv.SEP + mv.SEP.join([mv.seg_lead(mv.dist_band(lead_m), variant), mv.pick(act_key, variant, fem), street])
    return build


def arabic_lines():
    M = mv.Maneuver
    spec = [
        ("ar01_far", "Far first announcement — the whole picture (800 m)",
         lambda f: mv.compose(mv.APPROACH, M("turn", "right"), 800.0, variant=0, fem=f)),
        ("ar02_remind", "Short reminder — staged «remind» (200 m, already told)",
         lambda f: mv.compose(mv.APPROACH, M("turn", "right"), 200.0, variant=1, fem=f, told=True)),
        ("ar03_now", "At the turn — staged «now»",
         lambda f: mv.compose(mv.FINAL, M("turn", "right"), 30.0, variant=0, fem=f, told=True)),
        ("ar04_roundabout", "Roundabout exit (2nd exit, straight on)",
         lambda f: mv.compose(mv.APPROACH, M("roundabout", "straight", 2, 180.0), 300.0, variant=0, fem=f)),
        ("ar05_offramp", "Highway exit (off ramp, 1 km)",
         lambda f: mv.compose(mv.APPROACH, M("off ramp", "slight right"), 1000.0, variant=0, fem=f)),
        ("ar06_bridge_take", "Take the bridge on the right (ev_bridge_take)",
         lambda f: mv.ev_bridge_take(False, "right", 400.0, "", 0, f)),
        ("ar07_bridge_dont", "Stay under the bridge ahead (ev_bridge_dont, ahead)",
         lambda f: mv.ev_bridge_dont(False, "", 300.0, 0, f)),
        ("ar08_uturn", "U-turn (250 m)",
         lambda f: mv.compose(mv.APPROACH, M("turn", "uturn"), 250.0, variant=0, fem=f)),
        ("ar09_fork", "Keep left at the fork (ForkSense early call, 400 m)",
         lambda f: mv.ev_fork("left", 400.0, 0, f)),
        ("ar10_camera", "Speed camera with the limit (ev_camera, 80)",
         lambda f: mv.ev_camera("speed", 500.0, 80, 0, f)),
        ("ar11_jam_offer", "Jam ahead + faster-route offer (ev_traffic offer.jam, 7 min)",
         lambda f: mv.ev_traffic("offer.jam", None, 7.0, 0, f)),
        ("ar12_missed", "Reroute after a missed turn (ev.offroute.missed)",
         lambda f: mv.ev_simple("offroute.missed", 0, f)),
        ("ar13_arrival", "Arrival, destination on the right (ev_arrival + qm.arrive)",
         lambda f: mv.ev_arrival("right", True, 0, f, "qm.arrive")),
        ("ar14_street_tes3een", "Turn onto a MAJOR street by name — «شارع التسعين» (300 m)",
         _street(300.0, "act.turn.right.approach", "على شارع التسعين")),
        ("ar15_street_ring", "Highway exit onto the Ring Road by name (1.5 km)",
         _street(1500.0, "act.offramp.right.approach", "على الطريق الدائري")),
        ("ar16_quiet", "Quiet moment — a long open road (quiet_line long)",
         lambda f: mv.quiet_line("long", 0, f)),
    ]
    out = []
    for lid, label, build in spec:
        text = build(False)
        out.append({"id": lid, "lang": "ar", "label": label, "line": text, "plain": mv.plain(text),
                    "segments": mv.segments(text), "fem_plain": mv.plain(build(True))})
    return out


# The Arabic probe: two real lines in one transcript — ج (جامدة), ق (قدّامك, الطريق — spelled ء for the voice),
# the Egyptian vowel of «تلت», numbers.
def probe_ar():
    a = mv.ev_traffic("jam", 2000.0, 20.0, 0, False)
    b = mv.ev_bridge_dont(False, "", None, 0, False)
    segs = mv.segments(a) + mv.segments(b)
    fem = mv.segments(mv.ev_traffic("jam", 2000.0, 20.0, 0, True)) + mv.segments(mv.ev_bridge_dont(False, "", None, 0, True))
    return {"id": "ar00_probe", "lang": "ar", "label": "PROBE — jam ahead + stay under the bridge (ج, ق, تلت, numbers)",
            "line": mv.SEP + mv.SEP.join(segs), "plain": " ".join(segs), "segments": segs, "fem_plain": " ".join(fem),
            "parts": [a, b]}


def english_lines():
    from types import SimpleNamespace as S
    from tools.traffic_advisor import traffic_advisor as ta
    M = mv.Maneuver
    spec = [
        ("en01_far", "Far first announcement — the provider's own line (road name)",
         "In 800 meters, turn right onto Salah Salem Street.", "Mapbox-style voice instruction (first line about a maneuver)"),
        ("en02_near", "Near — staged reminder (brief_en)",
         mv.brief_en(mv.APPROACH, M("turn", "right"), 200.0), "masri_voice.brief_en (remind)"),
        ("en03_now", "At the turn (brief_en now)", mv.brief_en(mv.FINAL, M("turn", "right"), 30.0),
         "masri_voice.brief_en (now)"),
        ("en04_roundabout", "Roundabout exit — the provider's line",
         "In 300 meters, enter the roundabout and take the 2nd exit onto Gameat El Dowal El Arabeya Street.",
         "Mapbox-style voice instruction"),
        ("en05_offramp", "Highway exit — the provider's line",
         "In 1 kilometer, take the exit on the right toward Ring Road.", "Mapbox-style voice instruction"),
        ("en06_camera", "Speed camera with the limit (the app's own words)",
         "Speed camera in 500 meters, limit 80 kilometers per hour", "MapboxNavigationActivity camera voice"),
        ("en07_traffic", "Jam ahead + faster-route offer (TrafficAdvisor.english)",
         ta.english(S(kind="OFFER", reason="jam", minutes=7.0, dist_m=None)), "traffic_advisor.english"),
        ("en08_arrival", "Arrival — the provider's line",
         "You have arrived at your destination, on the right.", "Mapbox-style arrival instruction"),
    ]
    return [{"id": i, "lang": "en", "label": l, "line": t, "plain": t, "segments": [t], "source": src}
            for i, l, t, src in spec]


def probe_en():
    return {"id": "en00_probe", "lang": "en", "label": "PROBE — the line the shipped Journey-F reference clip says",
            "line": EN_REF_TEXT, "plain": EN_REF_TEXT, "segments": [EN_REF_TEXT], "source": "docs/voice-samples 07_en"}


# ── the batching question (for the bank plan): bank SEGMENTS, one request each vs N in ONE request cut at the
# pauses. Both are judged the way build_bank judges a direction-word clip: INSIDE a line, behind the bank's own
# lead clip «بعد تلتمية متر،» (a 1-second clip alone fools every transcriber). Seven segments the bank has (its
# 2.5-Pro single-request clip is the comparison) and five it lacks.
BATCH_SEGS = ["لف يمين", "خليك شمال عند التفريعة", "اطلع الكوبري اللي على يمينك", "فيه رادار،", "خُد تاني مخرج،",
              "خُد المخرج اللي على يمينك", "حمد الله على السلامة!",           # in the bank: the CURRENT clip beside it
              "هتخش يمين", "دلوقتي يمين", "لف وارجع", "تحب نمشي منه؟", "على الطريق الدائري"]   # not in the bank
CONTEXT_LEAD = build_bank.CONTEXT_LEAD


def fem_of(seg):
    for vs in mv.PH.values():
        for v in vs:
            if isinstance(v, dict) and seg in (v["m"], mv.comma(v["m"])):
                return v["f"] if seg == v["m"] else mv.comma(v["f"])
    return seg


def batch_lines():
    return [{"id": "bt%02d" % (i + 1), "lang": "ar", "label": "Bank segment inside a line (batch test): «%s»" % sg,
             "line": mv.SEP + mv.SEP.join([CONTEXT_LEAD, sg]), "plain": CONTEXT_LEAD + " " + sg,
             "segments": [CONTEXT_LEAD, sg], "seg": sg, "fem_plain": CONTEXT_LEAD + " " + fem_of(sg)}
            for i, sg in enumerate(BATCH_SEGS)]


def all_lines():
    return [probe_ar()] + arabic_lines() + [probe_en()] + english_lines() + batch_lines() + sekka_lines()


# ── Sekka (the assistant): its LIVE voice (2026-10-17, docs/azure-voice-2026-10.md §8) ───────────────
# Six typical replies, the same meaning in both languages: a greeting, a traffic answer, an add-stop confirmation,
# a "searching…" line, a reroute explanation, a sign-off (the last is the engine's own ack_bye). The Arabic is
# Cairene as the turn engine writes it (masculine addressee, numbers in words); `fem` is the feminine-addressee form
# (the read-back's feminized check). `diac` is the reply FULLY DIACRITIZED the Cairene way — ق written as the
# glottal stop Cairo says (ء / أ), ث as the s / t Cairo says, Egyptian vowels («تِحِبّ», «عَنْد», «الجِديد»), no case
# endings (the pausal sukun is added after, as the runtime does) — written on the desk as the BEST-CASE input of a
# diacritizing model: what Azure's ar-EG voices make of perfectly marked Egyptian text.
SEKKA_REPLIES = [
    ("greet", "Greeting",
     "أهلاً، أنا سِكّة. تحب أساعدك في إيه؟", "أهلاً، أنا سِكّة. تحبي أساعدك في إيه؟",
     "أَهْلاً، أَنا سِكّة. تِحِبّ أَساعْدَك في إيه؟",
     "Hey, I'm Sekka. What can I do for you?"),
    ("traffic", "Traffic answer",
     "فيه زحمة جامدة على الدائري قدام كوبري المنيب، هتأخرك حوالي ربع ساعة.",
     "فيه زحمة جامدة على الدائري قدام كوبري المنيب، هتأخرك حوالي ربع ساعة.",
     "فيه زَحْمَة جامْدَة عَلى الدّائِري أُدّام كوبْري المُنيب، هَتْأَخَّرَك حَوالي رُبْع ساعَة.",
     "There's heavy traffic on the Ring Road near the Moneeb bridge — it'll cost you about fifteen minutes."),
    ("stop", "Add-stop confirmation",
     "تمام، ضفتلك وقفة عند بنزينة وطنية على طريقك، هتزوّد دقيقتين بس.",
     "تمام، ضفتلك وقفة عند بنزينة وطنية على طريقك، هتزوّد دقيقتين بس.",
     "تَمام، ضِفْتِلَك وَءْفَة عَنْد بَنْزينَة وَطَنِيَّة عَلى طَريئَك، هَتْزَوِّد دِئيئْتين بَسّ.",
     "Done — I added a stop at the Wataniya station on your way. It only adds two minutes."),
    ("search", "Searching…",
     "ثواني، بدوّرلك على أقرب صيدلية مفتوحة.", "ثواني، بدوّرلك على أقرب صيدلية مفتوحة.",
     "سَواني، بَدَوَّرْلَك عَلى أَءْرَب صَيْدَلِيَّة مَفْتوحَة.",
     "One sec, I'm looking for the nearest open pharmacy."),
    ("reroute", "Reroute explanation",
     "غيّرتلك الطريق عشان فيه حادثة على المحور، والطريق الجديد أسرع بعشر دقايق.",
     "غيّرتلك الطريق عشان فيه حادثة على المحور، والطريق الجديد أسرع بعشر دقايق.",
     "غَيَّرْتِلَك الطَّريء عَشان فيه حادْسَة عَلى المِحْوَر، وِالطَّريء الجِديد أَسْرَع بِعَشَر دَآيِء.",
     "I switched your route because there's an accident on the Mehwar — the new one is ten minutes faster."),
    ("bye", "Sign-off (ack_bye)",
     "تمام، أنا معاك لو احتجت حاجة.", "تمام، أنا معاكي لو احتجتي حاجة.",
     "تَمام، أَنا مَعاك لَوْ احْتَجْت حاجَة.",
     "Okay — I'm here if you need me."),
]


def sekka_lines():
    out = []
    for key, label, ar, fem, diac, en in SEKKA_REPLIES:
        out.append({"id": "sk_ar_" + key, "lang": "ar", "label": "Sekka — " + label, "line": ar, "plain": ar,
                    "segments": [ar], "fem_plain": fem, "diac": diac, "sekka": True, "key": key})
        out.append({"id": "sk_en_" + key, "lang": "en", "label": "Sekka — " + label, "line": en, "plain": en,
                    "segments": [en], "source": "Sekka reply", "sekka": True, "key": key})
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE PROMPTS — director's notes; only the transcript is spoken
# ════════════════════════════════════════════════════════════════════════════════════════════════
# the bank's own sound list (tts.EGY_STYLE_PROMPT), stated as notes
ACCENT_AR = ("natural Egyptian Arabic with a Cairo accent. ج is always a hard g (as in «جامدة», gamda); ق is a glottal "
             "stop (as in «قدّام», 'uddam) — the transcript already spells it ء / أ where it matters; ث as t, ذ as d or z; "
             "Egyptian vowels and Egyptian intonation. Never formal Fusha, never a Gulf or Levantine accent.")
assert "glottal stop" in bank_tts.EGY_STYLE_PROMPT and "hard g" in bank_tts.EGY_STYLE_PROMPT


def say_text(line, take=1, raw=False):
    """The text the voice reads: every segment through pronounce.say — spelling variant (take - 1), as build_bank."""
    if line["lang"] != "ar" or raw:
        return line["plain"]
    return " ".join(pronounce.say(s, (take - 1) % pronounce.variants(s)) for s in line["segments"])


def prompt_ar(text):
    return ("# AUDIO PROFILE: A Cairene woman in her thirties, born and raised in Cairo — the driver's calm friend in "
            "the passenger seat, who knows every street of the city.\n"
            "## THE SCENE: Inside a car in Cairo traffic. The driver is a man. She tells him what is coming next on "
            "the road, speaking to him directly in the masculine — exactly as the transcript is written.\n"
            "### DIRECTOR'S NOTES\n"
            "Accent: " + ACCENT_AR + "\n"
            "Style: warm, calm, clear and confident — a friend helping, not a radio announcer, not theatrical, "
            "no added emotion.\n"
            "Pace: a natural conversational pace, unhurried; a short pause at each comma.\n"
            "Rule: speak every word of the transcript exactly as written, once — add nothing, drop nothing, and never "
            "change a word's grammatical gender: the listener is a man.\n"
            "#### TRANSCRIPT\n" + text)


def prompt_ar_batch(text, n):
    return prompt_ar(text).replace("#### TRANSCRIPT\n", (
        "Lines: the transcript is %d separate short lines. Say each line once, on its own, as a complete short phrase "
        "in the same calm voice, with a clear one-second pause after every line. Never read the line breaks aloud.\n"
        "#### TRANSCRIPT\n") % n)


def prompt_en_batch(text, n):
    return prompt_en(text).replace("#### TRANSCRIPT\n", (
        "Lines: the transcript is %d separate short lines. Say each line once, on its own, as a complete sentence in "
        "the same calm voice, with a clear one-second pause after every line. Never read the line breaks aloud.\n"
        "#### TRANSCRIPT\n") % n)


def prompt_en(text):
    return ("# AUDIO PROFILE: A warm, natural American woman in her thirties — a navigation voice that sounds like a "
            "calm, friendly person, not a machine.\n"
            "## THE SCENE: Inside a car in Cairo. She tells the driver what is coming next on the road.\n"
            "### DIRECTOR'S NOTES\n"
            "Accent: General American English. Egyptian street names said simply and clearly, the way an English "
            "speaker who lives in Cairo says them.\n"
            "Style: warm, natural, clear and friendly — a calm friend, not a radio announcer, not theatrical.\n"
            "Pace: a natural mid-pace, unhurried; a short pause at each comma.\n"
            "Rule: speak every word of the transcript exactly as written, once — add nothing, drop nothing.\n"
            "#### TRANSCRIPT\n" + text)


def build_prompt(line, text):
    return prompt_ar(text) if line["lang"] == "ar" else prompt_en(text)


# THE JUDGE HEARS SEVERAL TAKES PER REQUEST: the free tier gives a text model 20 requests a day
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20 on gemini-3.8-flash, measured 2026-10-16), so one request
# carries up to JUDGE_BATCH clips, each announced by a text label, and returns one verdict per clip.
JUDGE_VERSION = "b1"
JUDGE_BATCH = 6
JUDGE_HEAD = ("You will hear {n} separate clips of a car navigation voice, each announced by a text label \"CLIP k\" "
              "just before it. Judge EACH clip on its own — never let one clip's words or voice leak into another "
              "clip's verdict. Report only what you HEAR: do not guess what a clip was meant to say and do not correct "
              "anything.\n")
JUDGE_REF = ("Before the clips there is a REFERENCE clip (the navigation voice the app used to have); use it ONLY for "
             "the sim_* keys.\n")
JUDGE_TAIL = "Return a JSON array of exactly {n} objects, in clip order, each with \"clip\" (its number) and:\n"
JUDGE_AR_ROLE = "You are a native Cairene Egyptian Arabic listener and a phonetician. "
JUDGE_AR_KEYS = (
    "transcript: the exact words in Arabic script as Egyptian Arabic is written; numbers as the words spoken, never "
    "digits; keep the grammatical gender you hear;\n"
    "accent: one of \"Cairene\", \"Egyptian (non-Cairene)\", \"MSA\", \"Levantine\", \"Gulf\", \"Maghrebi\", \"mixed\";\n"
    "accent_score: 1-10 (10 = a native Cairene hears nothing foreign);\n"
    "jim: how every ج sounds: \"g\", \"dʒ\", \"ʒ\", \"mixed\", or \"none\" if the clip has no ج;\n"
    "qaf: how the words that have ق in standard spelling sound: \"glottal\", \"q\", \"g\", \"mixed\", or \"none\";\n"
    "addressee: grammatical gender of the person spoken to: \"masculine\", \"feminine\", \"mixed\", \"unclear\";\n"
    "warmth: 1-10 (10 = a warm calm friend in the passenger seat, 1 = cold, robotic or a broadcast announcer);\n"
    "naturalness: 1-10;\n"
    "pace: \"slow\", \"natural\" or \"fast\";\n"
    "problems: a list of short strings — mispronounced words (quote them), odd stress, a robotic or metallic sound, "
    "an added or dropped word, a breath, noise, a cut-off; empty if none;\n"
    "notes: one or two sentences in English.")
JUDGE_EN_ROLE = "You are a careful listener judging a car navigation voice in English. "
JUDGE_EN_KEYS = (
    "transcript: the exact words spoken, numbers as words;\n"
    "accent: e.g. \"General American\", \"British\", \"Australian\", \"other\";\n"
    "gender: \"female\" or \"male\";\n"
    "warmth: 1-10 (10 = warm and friendly, 1 = cold or robotic);\n"
    "naturalness: 1-10;\n"
    "clarity: 1-10;\n"
    "pace: \"slow\", \"natural\" or \"fast\";\n"
    "problems: a list of short strings (mispronounced words — quote them, odd stress, artifacts, added or dropped "
    "words); empty if none;\n"
    "notes: one or two sentences.")
JUDGE_SIM_KEYS = (
    ";\nsim_timbre, sim_pitch, sim_warmth, sim_pace, sim_overall: each 1-10, how close this clip's VOICE (not its "
    "words) is to the REFERENCE voice (10 = indistinguishable);\n"
    "sim_notes: one sentence on how this voice differs from the reference.")


# Sekka's replies (lang "ar-sekka" / "en-sekka"): the same blind protocol, framed as an in-car voice ASSISTANT, with
# what the owner asked for — English ENERGY (warm and upbeat, not over the top, not sleepy), Arabic case endings.
JUDGE_SEKKA_EN_ROLE = ("You are a careful listener judging the spoken replies of an in-car voice ASSISTANT (it answers "
                       "the driver's questions, adds stops, explains route changes) in English. ")
JUDGE_SEKKA_EN_KEYS = JUDGE_EN_KEYS.replace("notes: one or two sentences.", (
    "energy: 1-10 (1 = sleepy, flat or bored; 4 = calm but low; 6-8 = warm and upbeat, a friendly person who is glad "
    "to help; 9 = very excited; 10 = over the top, salesy or cartoonish);\n"
    "notes: one or two sentences."))
JUDGE_SEKKA_AR_ROLE = JUDGE_AR_ROLE + ("You judge the spoken replies of an in-car voice ASSISTANT (it answers the "
                                       "driver, adds stops, explains route changes) that must sound like a Cairene. ")
JUDGE_SEKKA_AR_KEYS = JUDGE_AR_KEYS.replace("notes: one or two sentences in English.", (
    "case_endings: Modern-Standard case endings or nunation heard at the ends of words (e.g. «طريقٍ», «يميناً»): "
    "\"none\", \"some\" or \"many\";\n"
    "notes: one or two sentences in English."))
assert "energy" in JUDGE_SEKKA_EN_KEYS and "case_endings" in JUDGE_SEKKA_AR_KEYS


def judge_prompt(lang, n, with_ref):
    if lang in ("ar-sekka", "en-sekka"):
        role, keys = ((JUDGE_SEKKA_AR_ROLE, JUDGE_SEKKA_AR_KEYS) if lang == "ar-sekka"
                      else (JUDGE_SEKKA_EN_ROLE, JUDGE_SEKKA_EN_KEYS))
        head = JUDGE_HEAD.replace("a car navigation voice", "an in-car voice assistant")
        return role + head.format(n=n) + JUDGE_TAIL.format(n=n) + keys
    role, keys = (JUDGE_AR_ROLE, JUDGE_AR_KEYS) if lang == "ar" else (JUDGE_EN_ROLE, JUDGE_EN_KEYS)
    return (role + JUDGE_HEAD.format(n=n) + (JUDGE_REF if with_ref else "") + JUDGE_TAIL.format(n=n) + keys
            + (JUDGE_SIM_KEYS if with_ref else ""))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE API — paced per model, every quota answer recorded
# ════════════════════════════════════════════════════════════════════════════════════════════════
class Exhausted(Exception):
    pass


class Overloaded(Exception):
    pass


def _pt_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        return datetime.datetime.utcnow() - datetime.timedelta(hours=7)


def _next_pt_midnight_epoch():
    now = _pt_now()
    nxt = (now + datetime.timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return time.time() + (nxt - now).total_seconds()


class Limits:
    def __init__(self):
        self.path = os.path.join(OUT, "limits.json")
        self.d = json.load(io.open(self.path, encoding="utf-8")) if os.path.exists(self.path) else {"models": {}}
        self.last = {}

    def m(self, model):
        return self.d["models"].setdefault(model, {
            "ok": 0, "e429": 0, "e5xx": 0, "e_other": 0, "no_audio": 0, "ok_by_ptday": {}, "violations": [],
            "gap_s": 0.0, "blocked_until": 0.0, "lat_sum": 0.0, "lat_n": 0, "lat_max": 0.0,
            "tokens_in": 0, "tokens_out": 0, "audio_s": 0.0})

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        io.open(self.path, "w", encoding="utf-8").write(scrub(json.dumps(self.d, ensure_ascii=False, indent=1)))

    def wait(self, model):
        m = self.m(model)
        if m["blocked_until"] > time.time():
            raise Exhausted("%s: daily quota spent until %s" % (model, time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(m["blocked_until"]))))
        w = self.last.get(model, 0.0) + m["gap_s"] - time.time()
        if w > 0:
            time.sleep(w)
        self.last[model] = time.time()

    def ok(self, model, lat, usage):
        m = self.m(model)
        m["ok"] += 1
        day = _pt_now().strftime("%Y-%m-%d")
        m["ok_by_ptday"][day] = m["ok_by_ptday"].get(day, 0) + 1
        m["lat_sum"] += lat
        m["lat_n"] += 1
        m["lat_max"] = max(m["lat_max"], lat)
        m["tokens_in"] += int((usage or {}).get("promptTokenCount") or 0)
        m["tokens_out"] += int((usage or {}).get("candidatesTokenCount") or 0)
        self.save()

    def note_429(self, model, err):
        m = self.m(model)
        m["e429"] += 1
        viol, retry = [], None
        for det in (err.get("details") or []):
            t = det.get("@type", "")
            if t.endswith("QuotaFailure"):
                for v in det.get("violations") or []:
                    viol.append({k: v.get(k) for k in ("quotaMetric", "quotaId", "quotaDimensions", "quotaValue")})
            elif t.endswith("RetryInfo"):
                rd = str(det.get("retryDelay") or "")
                mm = re.match(r"([\d.]+)s", rd)
                retry = float(mm.group(1)) if mm else None
        rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"),
               "pt_day": _pt_now().strftime("%Y-%m-%d"),
               "ok_today_before": m["ok_by_ptday"].get(_pt_now().strftime("%Y-%m-%d"), 0),
               "violations": viol, "retryDelay_s": retry, "message": scrub(err.get("message", ""))[:300]}
        m["violations"].append(rec)
        m["violations"] = m["violations"][-40:]
        per_day = any("PerDay" in (v.get("quotaId") or "") for v in viol)
        for v in viol:
            qid, val = v.get("quotaId") or "", v.get("quotaValue")
            if "PerMinute" in qid and "Request" in qid and val:
                try:
                    m["gap_s"] = max(m["gap_s"], 60.0 / float(val) * 1.08)
                except ValueError:
                    pass
        if per_day:
            m["blocked_until"] = _next_pt_midnight_epoch()
        self.save()
        return per_day, retry, viol


LIM = None
STATS = {"calls": 0, "max_calls": None}


def _post(model, body, timeout=240):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(API % model, data=data, headers={
        "x-goog-api-key": api_key(), "Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call(model, body, kind, max5xx=None):
    """One generateContent, paced; 429 per-minute → wait it out, per-day → Exhausted; 5xx → backoff
    (→ Overloaded after `max5xx` of them, so a caller with another model can move on)."""
    soft429 = 0
    n5xx = 0
    for attempt in range(10):
        if STATS["max_calls"] is not None and STATS["calls"] >= STATS["max_calls"]:
            raise Exhausted("--max-calls reached")
        LIM.wait(model)
        STATS["calls"] += 1
        t0 = time.time()
        try:
            r = _post(model, body)
            LIM.ok(model, time.time() - t0, r.get("usageMetadata"))
            return r
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(raw).get("error", {})
            except ValueError:
                err = {"message": raw[:300]}
            if e.code == 429:
                per_day, retry, viol = LIM.note_429(model, err)
                ids = ",".join("%s=%s" % (v.get("quotaId"), v.get("quotaValue")) for v in viol) or "no details"
                log("  429 %s (%s) retry=%s" % (model, ids, retry))
                if per_day:
                    raise Exhausted("%s: per-day quota (%s)" % (model, ids))
                soft429 += 1
                if not viol and soft429 >= 5:
                    LIM.m(model)["blocked_until"] = time.time() + 3600
                    LIM.save()
                    raise Exhausted("%s: repeated 429 without details" % model)
                time.sleep(max(retry or 0.0, LIM.m(model)["gap_s"], 15.0) + 1.0)
                continue
            if e.code in (500, 502, 503, 504):
                LIM.m(model)["e5xx"] += 1
                LIM.save()
                n5xx += 1
                if max5xx is not None and n5xx >= max5xx:
                    log("  %d %s — overloaded, moving on" % (e.code, model))
                    raise Overloaded(model)
                wait = min(120.0, 6.0 * (2 ** min(attempt, 4)))
                log("  %d %s — %s; retry in %.0fs" % (e.code, model, scrub(err.get("message", ""))[:120], wait))
                time.sleep(wait)
                continue
            LIM.m(model)["e_other"] += 1
            LIM.save()
            raise RuntimeError(scrub("HTTP %d %s: %s" % (e.code, model, err.get("message", raw[:300]))))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            log("  network %s: %s" % (model, scrub(e)))
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError("%s: gave up after retries" % model)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# AUDIO
# ════════════════════════════════════════════════════════════════════════════════════════════════
def to_wav(mime, raw):
    if raw[:4] == b"RIFF":
        return raw
    mm = re.search(r"rate=(\d+)", mime or "")
    rate = int(mm.group(1)) if mm else 24000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:      # Gemini's "L16" PCM is little-endian (its own docs feed it as s16le)
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(raw)
    return buf.getvalue()


def processed(wav_b):
    """The take as a bank clip would be: lead / tail trimmed, faded, speech RMS to -20 dBFS (build_bank.process)."""
    x, rate = acoustics.pcm(wav_b)
    return build_bank.wav_bytes(build_bank.process(x, rate), rate)


_LEAD = {}


def in_context(wav_b):
    """The take behind the bank's own lead clip «بعد تلتمية متر،», with the pause VoiceBank puts there."""
    if "x" not in _LEAD:
        lp = os.path.join(BANK_DIR, mv.fnv64(CONTEXT_LEAD) + ".mp3")
        _LEAD["x"], _LEAD["rate"] = acoustics.pcm(mp3_to_wav(lp))
    x, rate = acoustics.pcm(wav_b)
    gap = np.zeros(int(rate * mv.pause_after(CONTEXT_LEAD) / 1000.0), dtype=np.float32)
    return build_bank.wav_bytes(np.concatenate([np.asarray(_LEAD["x"], dtype=np.float32), gap,
                                                np.asarray(x, dtype=np.float32)]), rate)


def split_on_silence(wav_b, k, min_gap_s=0.15, weights=None):
    """Cut one take of k lines at its k - 1 longest inner pauses → k WAVs, or None when it has fewer pauses."""
    x, rate = acoustics.pcm(wav_b)
    hop = int(rate * 0.01)
    n = len(x) // hop
    if n < 10:
        return None
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    silent = db < db.max() - 40.0
    voiced = np.where(~silent)[0]
    if len(voiced) == 0:
        return None
    first, last = voiced[0], voiced[-1]
    gaps, i = [], first
    while i <= last:
        if silent[i]:
            j = i
            while j <= last and silent[j]:
                j += 1
            if (j - i) * 0.01 >= min_gap_s:
                gaps.append((j - i, i, j))
            i = j
        else:
            i += 1
    if len(gaps) < k - 1:
        return None
    if weights is None or k <= 1:
        cuts = sorted(sorted(gaps, reverse=True)[:k - 1], key=lambda g: g[1])
    else:
        cuts = _aligned_cuts(gaps, first, last + 1, weights)
        if cuts is None:
            return None
    bounds = [0] + [((a + b) // 2) * hop for _, a, b in cuts] + [len(x)]
    return [build_bank.wav_bytes(x[bounds[q]:bounds[q + 1]], rate) for q in range(k)]


def letters_of(text):
    """A line's expected length for the aligned cut: its letters (harakat are not letters) + a short floor."""
    return sum(1 for c in text if c.isalpha()) + 4


def _aligned_cuts(gaps, start, end, weights):
    """The k - 1 pauses whose pieces best match the lines' expected lengths (∝ `weights`, e.g. letters): a
    sentence-internal pause inside a long line can be longer than the pause BETWEEN two lines, so the longest
    pauses alone misalign (2026-10-16: the 17-line take cut a two-sentence line in half). Dynamic programming over
    the candidate pauses: cost = Σ log²(piece / expected) − 0.6 · Σ log(pause length)."""
    k = len(weights)
    m = len(gaps)
    speech = (end - start) - 0.5 * sum(g[0] for g in gaps)          # frames, the pauses partly belong to no line
    tot_w = float(sum(weights))
    exp_len = [max(1.0, speech * w / tot_w) for w in weights]
    mids = [(a + b) / 2.0 for _, a, b in gaps]
    INF = float("inf")

    def piece_cost(i, a_pos, b_pos):
        d = max(1.0, b_pos - a_pos)
        return math.log(d / exp_len[i]) ** 2

    # f[c][j]: best cost of the first c+1 pieces with cut c at gap j
    f = [[INF] * m for _ in range(k - 1)]
    back = [[-1] * m for _ in range(k - 1)]
    for j in range(m):
        f[0][j] = piece_cost(0, start, mids[j]) - 0.6 * math.log(gaps[j][0])
    for c in range(1, k - 1):
        for j in range(c, m):
            best, arg = INF, -1
            for p in range(c - 1, j):
                if f[c - 1][p] == INF:
                    continue
                v = f[c - 1][p] + piece_cost(c, mids[p], mids[j])
                if v < best:
                    best, arg = v, p
            if arg >= 0:
                f[c][j] = best - 0.6 * math.log(gaps[j][0])
                back[c][j] = arg
    best, arg = INF, -1
    for j in range(k - 2, m):
        if f[k - 2][j] == INF:
            continue
        v = f[k - 2][j] + piece_cost(k - 1, mids[j], end)
        if v < best:
            best, arg = v, j
    if arg < 0:
        return None
    idx = [arg]
    for c in range(k - 2, 0, -1):
        idx.append(back[c][idx[-1]])
    return [gaps[j] for j in reversed(idx)]


def encode_mp3(wav_b, out_path):
    """The bank's own format: MPEG-2 Layer III, 24 kHz mono, 32 kbps CBR."""
    if os.path.exists(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.wav"
    open(tmp, "wb").write(wav_b)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "24000", "-codec:a",
                    "libmp3lame", "-b:a", "32k", out_path], check=True)
    os.remove(tmp)


def mp3_to_wav(path):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "24000", "-f", "wav", "-"],
                       capture_output=True, check=True)
    return r.stdout


def sha(*parts):
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE CHECKS — the judge's transcript against the source text
# ════════════════════════════════════════════════════════════════════════════════════════════════
_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}


def ar_fold(s):
    s = (s or "").translate(_AR_DIGITS)
    s = re.sub(r"\d+", lambda m: " " + mv.num(int(m.group(0))) + " ", s)
    return build_bank.norm(pronounce.fold(s))


def ar_letters(s):
    return "".join(c for c in ar_fold(s) if c.isalpha())


def ar_words(s):
    return [w for w in re.split(r"[^\w]+", ar_fold(s)) if w]


def cer_ar(src, heard):
    a, b = ar_letters(src), ar_letters(heard)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / float(len(a))


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth", "6th": "sixth",
        "7th": "seventh", "8th": "eighth", "26th": "twenty sixth"}


def en_num(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + en_num(n % 100))
    return en_num(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + en_num(n % 1000))


def en_words(s):
    s = (s or "").lower().replace("—", " ").replace("-", " ").replace("’", "'")
    for k, v in _ORD.items():
        s = re.sub(r"\b%s\b" % k, v, s)
    s = re.sub(r"(\d+)\.5", lambda m: en_num(int(m.group(1))) + " point five", s)
    s = re.sub(r"\d+", lambda m: en_num(int(m.group(0))), s)
    s = s.replace("kilometres", "kilometers").replace("kilometre", "kilometer").replace("metres", "meters")
    s = s.replace("it'll", "it will").replace("there's", "there is").replace("you're", "you are")
    return re.findall(r"[a-z']+", s)


def wer(a, b):
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / float(len(a))


def text_check(line, heard):
    if line["lang"] == "en":
        w = wer(en_words(line["plain"]), en_words(heard))
        return {"wer": round(w, 3), "text_ok": w <= WER_PASS, "problems": [] if w <= WER_PASS else ["WER %.2f" % w]}
    src = line["plain"]
    c = cer_ar(src, heard)
    crit = build_bank.critical_ok(src, heard)
    masc = set(ar_words(src))
    fem_markers = sorted(set(ar_words(line.get("fem_plain") or "")) - masc)
    heard_w = set(ar_words(heard))
    feminized = [w for w in fem_markers if w in heard_w]
    missing = pronounce.missing_words(src, heard)
    probs = list(crit) + ["feminized «%s»" % w for w in feminized]
    if c > CER_PASS:
        probs.append("CER %.2f" % c)
    return {"cer": round(c, 3), "critical": crit, "feminized": feminized, "missing": missing,
            "text_ok": not crit and not feminized and c <= CER_PASS, "problems": probs}


def dialect_ok(j):
    if not j:
        return False
    acc = (j.get("accent") or "").lower()
    jim = (j.get("jim") or "none").lower()
    qaf = (j.get("qaf") or "none").lower()
    return acc.startswith("cairene") and jim in ("g", "none", "") and qaf in ("glottal", "none", "")


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE BLIND PHONE LISTENER (optional — it needs torch + the wav2vec2 model)
# ════════════════════════════════════════════════════════════════════════════════════════════════
PHONES = {"on": True, "mod": None, "err": None}


def phones_verdict(text, wav_b):
    if not PHONES["on"]:
        return None
    p = os.path.join(CACHE, "phones_%s.json" % hashlib.sha256(wav_b).hexdigest()[:24])
    if os.path.exists(p):
        return json.load(io.open(p, encoding="utf-8"))
    try:
        if PHONES["mod"] is None:
            from tools.masri_voice import phones as ph
            PHONES["mod"] = ph
        x, rate = acoustics.pcm(wav_b)
        ipa = PHONES["mod"].ipa(x, rate)
        hard, soft = pronounce.check(text, ipa)
        v = {"ipa": ipa, "hard": hard, "soft": soft}
    except Exception as e:
        PHONES["on"] = False
        PHONES["err"] = scrub(e)[:200]
        log("  phones listener unavailable: %s" % PHONES["err"])
        return None
    io.open(p, "w", encoding="utf-8").write(json.dumps(v, ensure_ascii=False))
    return v


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A TAKE — synthesise (cached), process, judge (cached), check
# ════════════════════════════════════════════════════════════════════════════════════════════════
REFS = {}


def ref_wav(path):
    if path not in REFS:
        REFS[path] = processed(mp3_to_wav(os.path.join(ROOT, path)))
    return REFS[path]


JUDGE_FORCE = {"model": None}     # --judge-model: one judge for every verdict of this run (consistent ranking)
_DEAD_JUDGES = set()              # models that answered 404 / 400 this run


def _verdict_key(lang, wav_b, ref_b, force):
    return sha("judgeB", JUDGE_VERSION, force or "", lang, hashlib.sha256(wav_b).hexdigest(),
               hashlib.sha256(ref_b).hexdigest() if ref_b else "")


def ref_mp3(path):
    return open(os.path.join(ROOT, path), "rb").read()


def cached_verdict(lang, wav_b, ref=None, force=None):
    ref_b = ref_mp3(ref) if ref else None
    p = os.path.join(CACHE, "verdict_%s.json" % _verdict_key(lang, wav_b, ref_b, force or JUDGE_FORCE["model"]))
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def judge_batch(lang, wavs, ref=None, force=None, rounds=6):
    """One request, len(wavs) clips → one verdict per clip (cached per clip). Exhausted when no judge can answer."""
    force = force or JUDGE_FORCE["model"]
    ref_b = ref_mp3(ref) if ref else None
    keys = [_verdict_key(lang, w, ref_b, force) for w in wavs]
    paths = [os.path.join(CACHE, "verdict_%s.json" % k) for k in keys]
    if all(os.path.exists(p) for p in paths):
        return [json.load(io.open(p, encoding="utf-8")) for p in paths]
    n = len(wavs)
    parts = []
    if ref_b:
        parts += [{"text": "REFERENCE"}, bank_tts.audio_part(ref_b, "audio/mpeg")]
    for i, w in enumerate(wavs):
        parts += [{"text": "CLIP %d" % (i + 1)}, bank_tts.audio_part(w, "audio/mpeg")]
    parts.append({"text": judge_prompt(lang, n, bool(ref_b))})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}}
    chain = [force] if force else [m for m in JUDGE_CHAIN if m not in _DEAD_JUDGES]
    last = None
    for rnd in range(rounds):
        for jm in list(chain):
            try:
                r = call(jm, body, "judge", max5xx=2)
            except Overloaded as e:
                last = e
                continue
            except Exhausted as e:           # out of quota today: LIM remembers, the next call skips it at once
                last = e
                chain.remove(jm)
                continue
            except RuntimeError as e:        # 404 (retired) / 400 (no audio input): never this run again
                last = e
                _DEAD_JUDGES.add(jm)
                chain.remove(jm)
                log("  judge %s unusable: %s" % (jm, scrub(e)[:160]))
                continue
            txt = "".join(pp.get("text", "") for pp in ((r.get("candidates") or [{}])[0].get("content") or {})
                          .get("parts", []) if not pp.get("thought"))
            try:
                arr = json.loads(txt)
            except ValueError:
                mm = re.search(r"\[.*\]", txt, re.S)
                try:
                    arr = json.loads(mm.group(0)) if mm else None
                except ValueError:
                    arr = None
            if isinstance(arr, dict):
                arr = arr.get("clips") or arr.get("results") or [arr]
            if not isinstance(arr, list) or len(arr) != n:
                last = RuntimeError("%s returned %s verdicts for %d clips" % (jm, len(arr) if isinstance(arr, list)
                                                                               else "no", n))
                log("  judge %s: %s" % (jm, last))
                if n > 1:                    # split and ask again (each half is its own request)
                    h = n // 2
                    return (judge_batch(lang, wavs[:h], ref, force, rounds)
                            + judge_batch(lang, wavs[h:], ref, force, rounds))
                continue
            byclip = {}
            for k, v in enumerate(arr):
                c = v.get("clip") if isinstance(v, dict) else None
                byclip[int(c) if isinstance(c, (int, float)) or (isinstance(c, str) and c.isdigit()) else k + 1] = v
            out = []
            for i in range(n):
                v = byclip.get(i + 1) or (arr[i] if isinstance(arr[i], dict) else {})
                v = dict(v)
                v["_judge_model"] = jm
                v["_batch"] = n
                io.open(paths[i], "w", encoding="utf-8").write(json.dumps(v, ensure_ascii=False))
                out.append(v)
            return out
        if not chain:
            break
        log("  every judge overloaded — waiting 90 s (round %d)" % (rnd + 1))
        time.sleep(90)
    raise Exhausted("no judge answered (%s)" % scrub(last))


def apply_verdict(rec, line, j):
    rec["judge"] = j
    rec["check"] = text_check(line, j.get("transcript", ""))
    rec["dialect_ok"] = dialect_ok(j) if line["lang"] == "ar" else None
    rec["pass"] = rec["check"]["text_ok"] and (rec["dialect_ok"] if line["lang"] == "ar" else True)


def heard_mp3(rec):
    """What the judge hears: the clip the page plays — the bank's own format, exactly what a driver would hear
    (and 1/12 of the WAV's bytes in a request that carries up to JUDGE_BATCH of them)."""
    return open(os.path.join(OUT, rec["clip"]), "rb").read()


def heard_wav(rec):
    hp = os.path.join(CACHE, rec.get("heard_wav") or "-")
    if os.path.exists(hp):
        return open(hp, "rb").read()
    return processed(open(os.path.join(CACHE, rec["key"] + ".wav"), "rb").read())


def judge_pending(R, groups=None, force=None):
    """Every take without a verdict (of `groups`), JUDGE_BATCH clips per request, grouped by language + reference
    (a group's takes in ONE request where it fits, so a ranking is made by one judge in one sitting)."""
    lines = {ln["id"]: ln for ln in all_lines()}
    todo = [r for r in R.takes.values() if r.get("judge") is None and (groups is None or r["group"] in groups)]
    buckets = {}
    for r in todo:
        buckets.setdefault((r["group"], r["lang"], r.get("ref") or ""), []).append(r)
    left = 0
    for (grp, lang, ref), recs in sorted(buckets.items()):
        # one request = DIFFERENT lines of one voice / model: a request of the same line by three models came back
        # with one transcript copied three times (2026-10-16) — the read-back must be independent per clip
        recs.sort(key=lambda r: (r["voice"], r["model"], r["take"], r["line"]))
        size = len(recs) if (grp == "prescreen" and len(recs) <= 10) else JUDGE_BATCH
        for i in range(0, len(recs), size):
            chunk = recs[i:i + size]
            try:
                vs = judge_batch(lang, [heard_mp3(r) for r in chunk], ref or None, force)
            except Exhausted as e:
                left = sum(1 for r in R.takes.values() if r.get("judge") is None
                           and (groups is None or r["group"] in groups))
                log("  judge stopped: %s (%d takes wait for a verdict)" % (scrub(e)[:160], left))
                return left
            for r, v in zip(chunk, vs):
                apply_verdict(r, lines[r["line"]], v)
                log("  %-4s %s %s %s  %s  %s" % ("PASS" if r["pass"] else "fail", r["line"], r["voice"], r["model"],
                                                (v.get("transcript") or "")[:60],
                                                ";".join(r["check"]["problems"])[:70]))
            R.save()
    return left


def synth(model, voice, prompt, take):
    key = sha(model, voice, prompt, *(["take=%d" % take] if take > 1 else []))
    wav_p = os.path.join(CACHE, key + ".wav")
    meta_p = os.path.join(CACHE, key + ".json")
    if os.path.exists(wav_p) and os.path.exists(meta_p):
        return key, open(wav_p, "rb").read(), json.load(io.open(meta_p, encoding="utf-8"))
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for attempt in range(2):
        t0 = time.time()
        r = call(model, body, "tts")
        lat = time.time() - t0
        parts = ((r.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        aud = [p["inlineData"] for p in parts if "inlineData" in p]
        if aud:
            mime = aud[0].get("mimeType", "")
            raw = b"".join(base64.b64decode(a["data"]) for a in aud)
            wav_b = to_wav(mime, raw)
            x, rate = acoustics.pcm(wav_b)
            meta = {"model": model, "voice": voice, "take": take, "mime": mime, "latency_s": round(lat, 2),
                    "usage": r.get("usageMetadata"), "audio_s": round(len(x) / float(rate), 2),
                    "at": datetime.datetime.now().isoformat(timespec="seconds")}
            LIM.m(model)["audio_s"] += meta["audio_s"]
            LIM.save()
            os.makedirs(CACHE, exist_ok=True)
            open(wav_p, "wb").write(wav_b)
            io.open(meta_p, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False))
            return key, wav_b, meta
        LIM.m(model)["no_audio"] += 1
        LIM.save()
        fr = ((r.get("candidates") or [{}])[0]).get("finishReason")
        log("  no audio from %s/%s (finishReason=%s)" % (model, voice, fr))
    raise RuntimeError("no audio from %s/%s" % (model, voice))


class Run:
    """The takes made so far (takes.json) and the run state (state.json). A take is GENERATED first (cached audio,
    no verdict) and judged later in batches (judge_pending) — the free tier's per-day request quotas, not time, are
    the budget, so a take whose judge is out of quota waits for tomorrow's quota instead of being lost."""

    def __init__(self):
        self.idx_p = os.path.join(OUT, "takes.json")
        self.takes = json.load(io.open(self.idx_p, encoding="utf-8")) if os.path.exists(self.idx_p) else {}
        self.state_p = os.path.join(OUT, "state.json")
        self.state = json.load(io.open(self.state_p, encoding="utf-8")) if os.path.exists(self.state_p) else {}
        self.missing = []
        for r in self.takes.values():          # verdicts of an older judge format are asked again (batched)
            if r.get("judge") is not None and r["judge"].get("_batch") is None:
                r["judge"] = None
                r["pass"] = None
                r["check"] = None

    def save(self):
        io.open(self.idx_p, "w", encoding="utf-8").write(json.dumps(self.takes, ensure_ascii=False, indent=1))
        io.open(self.state_p, "w", encoding="utf-8").write(json.dumps(self.state, ensure_ascii=False, indent=1))

    @staticmethod
    def tid(group, line_id, voice, model, take, raw=False):
        return "%s|%s|%s|%s|%d|%s" % (group, line_id, voice, model, take, "raw" if raw else "say")

    def one(self, line, voice, model, group, take=1, raw=False, ref=None, context=False):
        tid = self.tid(group, line["id"], voice, model, take, raw)
        if tid in self.takes:
            return self.takes[tid]
        if context:
            text = pronounce.say(line["seg"], (take - 1) % pronounce.variants(line["seg"]))
        else:
            text = say_text(line, take, raw)
        prompt = build_prompt(line, text)
        key, wav_raw, meta = synth(model, voice, prompt, take)
        return self.record(line, voice, model, group, take, text, key, wav_raw, meta, raw, ref, context)

    def record(self, line, voice, model, group, take, text, key, wav_raw, meta, raw=False, ref=None, context=False,
               sub=""):
        wav_b = processed(wav_raw)
        if context:
            wav_b = in_context(wav_b)
        clip = "clips/%s%s.mp3" % (key[:16], sub)
        encode_mp3(wav_b, os.path.join(OUT, clip))
        heard_p = os.path.join(CACHE, "%s%s.heard.wav" % (key, sub))     # exactly what the judge hears
        if not os.path.exists(heard_p):
            open(heard_p, "wb").write(wav_b)
        if sub:                                                          # a piece of a batched take
            open(os.path.join(CACHE, "%s%s.wav" % (key, sub)), "wb").write(wav_raw)
        ph = phones_verdict(line["plain"], wav_b) if line["lang"] == "ar" else None
        st = acoustics.clip_stats(wav_raw, line.get("seg") or line["plain"])
        rec = {"key": key, "sub": sub, "line": line["id"], "lang": line["lang"], "voice": voice, "model": model,
               "group": group, "take": take, "raw_spelling": raw, "tts_text": text, "clip": clip, "meta": meta,
               "judge": None, "check": None, "pass": None, "dialect_ok": None, "phones": ph, "stats": st, "ref": ref,
               "context": context, "heard_wav": os.path.basename(heard_p)}
        cv = cached_verdict(line["lang"], open(os.path.join(OUT, clip), "rb").read(), ref)
        if cv is not None:
            apply_verdict(rec, line, cv)
        self.takes[self.tid(group, line["id"], voice, model, take, raw)] = rec
        self.save()
        return rec

    def take(self, line, voice, model, group, raw=False, ref=None, context=False, take=1):
        """Generate one take (no verdict yet). A miss is recorded with its reason, never raised."""
        try:
            rec = self.one(line, voice, model, group, take, raw, ref, context)
            log("  made %s %s %s take %d (%.1f s)" % (line["id"], voice, model, take, rec["meta"].get("audio_s") or 0))
            return rec
        except (Exhausted, RuntimeError) as e:
            self.missing.append({"line": line["id"], "voice": voice, "model": model, "group": group,
                                 "why": scrub(e)[:200]})
            log("  MISSING %s %s %s — %s" % (line["id"], voice, model, scrub(e)[:160]))
            return None

    def retakes(self, groups=("audition", "compare", "batch-piece")):
        """ONE regeneration for every judged first take whose words did not match (the retake reads the next SAY
        spelling, as build_bank does). Takes that were made N-per-request are retaken the same way: the failed lines
        of one (group, voice, model) in ONE request."""
        lines = {ln["id"]: ln for ln in all_lines()}
        n = 0
        bundles = {}
        for r in list(self.takes.values()):
            if r["group"] not in groups or r["take"] != 1 or r.get("raw_spelling") or not r.get("check"):
                continue
            if r["check"]["text_ok"]:
                continue
            if self.tid(r["group"], r["line"], r["voice"], r["model"], 2) in self.takes:
                continue
            if (r.get("meta") or {}).get("batched"):
                bundles.setdefault((r["group"], r["voice"], r["model"]), []).append(lines[r["line"]])
            elif self.take(lines[r["line"]], r["voice"], r["model"], r["group"], ref=r.get("ref"),
                           context=r.get("context", False), take=2):
                n += 1
        for (g, v, m), lns in sorted(bundles.items()):
            n += self.batched(lns, v, m, g, take=2)
        return n

    def batched(self, lines, voice, model, group, take=1):
        """ALL `lines` in ONE request — the free tier gives a TTS model 10 requests a day — cut at the pauses into
        one take per line (sub "_p<i>"). Lines already made for (group, voice, model, take) are left out. Bank
        segments (lines with "seg") are spoken alone and judged inside a line, behind the bank's lead."""
        todo = [ln for ln in lines if self.tid(group, ln["id"], voice, model, take) not in self.takes]
        if not todo:
            return 0
        lang = todo[0]["lang"]
        ctx = [bool(ln.get("seg")) for ln in todo]
        texts = [pronounce.say(ln["seg"], (take - 1) % pronounce.variants(ln["seg"])) if ln.get("seg")
                 else say_text(ln, take) for ln in todo]
        prompt = (prompt_ar_batch if lang == "ar" else prompt_en_batch)("\n".join(texts), len(todo))
        try:
            key, wav_raw, meta = synth(model, voice, prompt, take)
        except (Exhausted, RuntimeError) as e:
            for ln in todo:
                self.missing.append({"line": ln["id"], "voice": voice, "model": model, "group": group,
                                     "why": scrub(e)[:200]})
            log("  MISSING %d lines %s %s %s — %s" % (len(todo), group, voice, model, scrub(e)[:140]))
            return 0
        rel = "clips/batch_%s.mp3" % key[:16]
        encode_mp3(processed(wav_raw), os.path.join(OUT, rel))
        pieces = split_on_silence(wav_raw, len(todo), weights=[letters_of(t) for t in texts])
        info = {"group": group, "voice": voice, "model": model, "take": take, "n": len(todo), "key": key, "clip": rel,
                "audio_s": meta.get("audio_s"), "latency_s": meta.get("latency_s"), "usage": meta.get("usage"),
                "cut_ok": pieces is not None, "lines": [ln["id"] for ln in todo]}
        self.state.setdefault("batches", {})["%s|%s|%s|%d" % (group, voice, model, take)] = info
        if pieces is None:
            for ln in todo:
                self.missing.append({"line": ln["id"], "voice": voice, "model": model, "group": group,
                                     "why": "the one-request take of %d lines could not be cut at its pauses (%s)"
                                            % (len(todo), rel)})
            log("  batch %s %s %s: could not cut %d lines" % (group, voice, model, len(todo)))
            self.save()
            return 0
        rate = acoustics.pcm(wav_raw)[1]
        for i, (ln, pc) in enumerate(zip(todo, pieces)):
            m2 = dict(meta, batched=len(todo), batch_clip=rel, piece=i,
                      audio_s=round(len(acoustics.pcm(pc)[0]) / float(rate), 2))
            self.record(ln, voice, model, group, take, texts[i], key, pc, m2, context=ctx[i], sub="_p%d" % i)
        log("  made %d lines in ONE request: %s %s %s (%.0f s of audio)" % (len(todo), group, voice, model,
                                                                         meta.get("audio_s") or 0))
        self.save()
        return len(todo)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# AZURE — the free F0 resource (tools/voice_audition/azure.py: key, ledger, characters)
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Azure has no daily request quota (20 requests / 60 s) but a MONTHLY CHARACTER budget the bank shares, so every take
# is ONE line per request (no cut, no break markup — a <break> costs ~22 characters) and the SSML carries only the
# attributes a setting needs. Takes go through the same pipeline as Gemini's (process → bank-format MP3 → the same
# blind judge → the phone listener), grouped "azure" (the final settings, shown beside the Gemini takes),
# "azure-tune" (the Arabic settings tried), "azure-screen" (the English voices tried), "azure-batch" (the batching
# measurement for the bank).
AZ_AR = ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"]
# The Arabic voice the bank build and the batching measurement use: the one the blind phone listener heard say ج as
# a hard g (Salma says dʒ on every ج word, and no SSML fixed it — docs/azure-voice-2026-10.md §2).
AZ_BANK_AR = "ar-EG-ShakirNeural"
# Azure-only respellings, applied AFTER pronounce.say (word → spelling): what the ar-EG voices still said wrong.
# A sukun on the last letter keeps the MSA case ending off a bare direction: Salma read «دلوقتي يمين» as «… يميناً»
# (the judge's read-back, 4 of 6 settings, 2026-10-17); «دلوقتي يمينْ» came back «yamiːn» (the phone listener).
AZ_SAY = {"يمين": "يمينْ", "شمال": "شمالْ",
          # «إنت فوّت» came back «إن تفوت» (Shakir, 3 settings): the Egyptian «inta» written out
          "إنت": "إِنْتَ", "انت": "إِنْتَ"}
# The final Arabic settings per voice (docs/azure-voice-2026-10.md §2 has how each was chosen).
AZ_AR_FINAL = {
    "ar-EG-SalmaNeural": {"spell": "pausal", "rate": "+15%"},
    "ar-EG-ShakirNeural": {"spell": "pausal", "rate": "+20%", "pitch": "+2%"},
}
# The Arabic settings tried on AZ_TUNE_LINES (each one request per line per voice).
AZ_AR_TUNE = [
    {"spell": "plain"},
    {"spell": "say"},
    {"spell": "say", "rate": "+6%"},
    {"spell": "say", "rate": "+6%", "pitch": "+3%"},
    {"spell": "say", "rate": "+12%", "pitch": "+5%"},
    {"spell": "say", "rate": "-6%"},
    {"spell": "say", "rate": "+15%"},      # round 2 (2026-10-17): the default pace is 7.9 letters/s (Gemini's 11.4)
    {"spell": "say", "rate": "+25%"},
    {"spell": "azsay", "rate": "+15%"},    # round 3: + AZ_SAY (the sukun on a bare direction)
    {"spell": "azsay", "rate": "+20%", "pitch": "+2%"},
    {"spell": "pausal", "rate": "+15%"},   # round 4: + the pausal sukun on every word (no MSA case endings)
    {"spell": "pausal", "rate": "+20%", "pitch": "+2%"},
]
AZ_TUNE_LINES = ["ar00_probe", "ar03_now", "ar11_jam_offer", "ar12_missed", "ar16_quiet"]
# English: the voices screened on the Journey-F reference line (voice, settings). Female first (Sekka and the guidance
# voice are a woman's in Arabic), a few men, the HD / MAI families that answered on F0.
AZ_EN_SCREEN = [
    ("en-US-AvaMultilingualNeural", {}),
    ("en-US-AvaNeural", {}),
    ("en-US-EmmaMultilingualNeural", {}),
    ("en-US-JennyNeural", {"style": "friendly"}),
    ("en-US-JennyNeural", {"style": "assistant"}),
    ("en-US-AriaNeural", {"style": "friendly"}),
    ("en-US-AriaNeural", {"style": "customerservice"}),
    ("en-US-LunaNeural", {"style": "conversation"}),
    ("en-US-SaraNeural", {"style": "friendly"}),
    ("en-US-MichelleNeural", {}),
    ("en-US-PhoebeMultilingualNeural", {}),
    ("en-US-NancyMultilingualNeural", {"style": "friendly"}),
    ("en-US-SerenaMultilingualNeural", {"style": "friendly"}),
    ("en-US-LolaMultilingualNeural", {}),
    ("en-US-CoraMultilingualNeural", {}),
    ("en-US-NovaTurboMultilingualNeural", {}),
    ("en-US-ShimmerTurboMultilingualNeural", {}),
    ("en-GB-SoniaNeural", {"style": "cheerful"}),
    ("en-GB-AdaMultilingualNeural", {}),
    ("en-AU-NatashaNeural", {}),
    ("en-US-Ava:DragonHDLatestNeural", {}),
    ("en-US-Emma:DragonHDLatestNeural", {}),
    ("en-US-Aria:DragonHDLatestNeural", {}),
    ("en-US-Jenny:DragonHDLatestNeural", {}),
    ("en-GB-Ada:DragonHDLatestNeural", {}),
    ("en-US-Tiana:DragonHDFlashLatestNeural", {}),
    ("en-US-Harper:MAI-Voice-2-Flash", {}),
    ("en-US-Iris:MAI-Voice-2-Flash", {}),
    ("en-US-AndrewMultilingualNeural", {}),
    ("en-US-BrianMultilingualNeural", {}),
    ("en-US-KaiNeural", {"style": "conversation"}),
    ("en-GB-OllieMultilingualNeural", {}),
]
# The five English voices every English line is rendered with (chosen from the screen — docs/azure-voice-2026-10.md §3).
AZ_EN_PICK = [
    ("en-US-AvaMultilingualNeural", {}),
    ("en-US-EmmaMultilingualNeural", {}),
    ("en-US-PhoebeMultilingualNeural", {}),
    ("en-US-Ava:DragonHDLatestNeural", {}),
    ("en-US-Harper:MAI-Voice-2-Flash", {}),
]


def az_model(cfg):
    """The take's `model` field: 'azure' + the settings (the page groups and labels by it)."""
    parts = ["azure"]
    for k in ("spell", "style", "degree", "rate", "pitch"):
        if cfg.get(k):
            parts.append("%s=%s" % (k, cfg[k]))
    return " ".join(parts)


def az_text(line, cfg, context=False):
    if line["lang"] != "ar":
        return line["plain"]
    if cfg.get("spell") == "sekka-runtime":     # Sekka's committed runtime text (azure_tts.ts arabicForAzure)
        return sekka_runtime_ar(line["plain"])
    if cfg.get("spell") == "sekka-diac":        # fully diacritized (SEKKA_REPLIES `diac`) + the pausal sukun
        return pausal(line["diac"])
    if cfg.get("spell") in SEKKA_LEXES:         # + a pronunciation lexicon for the words that still broke
        return sekka_lex(pausal(line["diac"]), cfg["spell"])
    if context:
        segs = [line["seg"]]
    else:
        segs = line["segments"]
    if cfg.get("spell") == "plain":
        text = " ".join(segs)
    else:
        text = " ".join(pronounce.say(s, 0) for s in segs)
    if cfg.get("spell") in ("azsay", "pausal"):  # + the Azure-only respellings the tuning found
        text = az_respell(text)
    if cfg.get("spell") == "pausal":             # + the pausal sukun on every word (no MSA case endings)
        text = pausal(text)
    return text


# The pausal form (round 4, 2026-10-17): the ar-EG voices read Egyptian text with MSA case endings and nunation where
# the judge heard them — «طريقٍ أسرعَ», «يميناً», «إن تفوت» for «إنت فوّت» — the newsreader sound. A sukun on the last
# letter of every word that ends in a consonant says "no vowel here", which is how Cairo ends a word. Not on a word that
# already ends in a diacritic, nor on ا / ى / آ / و / ي (long vowels), ه (the vowel of «كده» / «النهارده», the «-o» of
# «تاخده») or ة (Cairo says it -et before a noun, -a at a pause: the voice's own choice is kept). Twin: azure_tts.ts
# pausalAr (Sekka's runtime voice) — tests/azure_tts.test.ts holds the two to one output on a sample.
_PAUSAL_SKIP = set("اىآويهة")
_AR_LAST = re.compile("[ء-ي]")
_HARAKA_LAST = re.compile("[ً-ْٰ]")


def pausal(text):
    out = []
    for tok in text.split(" "):
        m = re.match(r"^([«(\"]*)(.*?)([.,!?؟،؛:»)\"…]*)$", tok)
        a, w, z = m.groups()
        if w and _AR_LAST.fullmatch(w[-1]) and w[-1] not in _PAUSAL_SKIP:
            w = w + "ْ"
        out.append(a + w + z)
    return " ".join(out)


def az_respell(text):
    """AZ_SAY on whole words, the punctuation around them kept («يمين،» → «يمينْ،»)."""
    out = []
    for tok in text.split(" "):
        m = re.match(r"^([«(\"]*)(.*?)([.,!?؟،؛:»)\"]*)$", tok)
        a, w, z = m.groups()
        out.append(a + AZ_SAY.get(w, w) + z)
    return " ".join(out)


def is_azure(rec):
    return str(rec.get("model", "")).startswith("azure")


def az_take(R, line, voice, cfg, group, ref=None, context=False):
    """ONE line in ONE request (cached by the SSML; nothing is asked twice). A miss is recorded, never raised."""
    from tools.voice_audition import azure as AZR
    model = az_model(cfg)
    tid = R.tid(group, line["id"], voice, model, 1)
    if tid in R.takes:
        return R.takes[tid]
    text = az_text(line, cfg, context)
    doc = AZR.ssml(text, voice, line["lang"], rate=cfg.get("rate"), pitch=cfg.get("pitch"), style=cfg.get("style"),
                   degree=cfg.get("degree"))
    try:
        wav, meta = AZR.synth_wav(doc, purpose=group)
    except (AZR.Exhausted, AZR.AzureError) as e:
        R.missing.append({"line": line["id"], "voice": voice, "model": model, "group": group, "why": scrub(e)[:200]})
        log("  MISSING %s %s %s — %s" % (line["id"], voice, model, scrub(e)[:160]))
        return None
    key = sha("azure", doc)
    raw_p = os.path.join(CACHE, key + ".wav")
    if not os.path.exists(raw_p):
        open(raw_p, "wb").write(wav)
    x, rate = acoustics.pcm(wav)
    m2 = dict(meta, model=model, voice=voice, take=1, ssml=doc, provider="azure",
              audio_s=round(len(x) / float(rate), 2))
    rec = R.record(line, voice, model, group, 1, text, key, wav, m2, ref=ref, context=context)
    log("  made %s %s %s (%d chars, %.1f s audio, %.2f s)" % (line["id"], voice, model, meta["chars"],
                                                            m2["audio_s"], meta["latency_s"]))
    return rec


AZ_SEPS = {   # how segments are separated in ONE request (each separator is billable markup)
    "break": '<break time="900ms"/>',
    "s": "</s><s>",            # each segment its own <s> sentence element
    "nl": "\n",                # a bare line break (1 character)
}


def az_batch_test(R, voice, cfg, seps=("break", "s", "nl")):
    """The bank question on Azure: the 12 batch segments in ONE request (for each separator in `seps`), cut by the
    aligned cut, against the same segments one request each — judged inside a line like every bank segment."""
    from tools.voice_audition import azure as AZR
    lines = batch_lines()
    texts = [az_text(ln, cfg, context=True) for ln in lines]
    for ln in lines:
        az_take(R, ln, voice, cfg, "azure-batch-single", context=True)
    single = sum(AZR.billable(AZR.ssml(t, voice, "ar", rate=cfg.get("rate"), pitch=cfg.get("pitch"))) for t in texts)
    for sep in seps:
        model = az_model(cfg) + " batch=%d sep=%s" % (len(lines), sep)
        if all(R.tid("azure-batch", ln["id"], voice, model, 1) in R.takes for ln in lines):
            continue
        body = AZ_SEPS[sep].join(escape_xml(t) for t in texts)
        if sep == "s":
            body = "<s>" + body + "</s>"
        doc = AZR.ssml(None, voice, "ar", rate=cfg.get("rate"), pitch=cfg.get("pitch"), body_xml=body)
        try:
            wav, meta = AZR.synth_wav(doc, purpose="azure-batch")
        except (AZR.Exhausted, AZR.AzureError) as e:
            log("  azure batch %s MISSING — %s" % (sep, scrub(e)[:160]))
            continue
        key = sha("azure", doc)
        rel = "clips/batch_%s.mp3" % key[:16]
        encode_mp3(processed(wav), os.path.join(OUT, rel))
        pieces = split_on_silence(wav, len(lines), weights=[letters_of(t) for t in texts])
        x, rate = acoustics.pcm(wav)
        info = {"group": "azure-batch", "voice": voice, "model": model, "sep": sep, "take": 1, "n": len(lines),
                "key": key, "clip": rel, "audio_s": round(len(x) / float(rate), 2), "latency_s": meta["latency_s"],
                "chars": meta["chars"], "chars_single": single, "cut_ok": pieces is not None,
                "lines": [ln["id"] for ln in lines]}
        R.state.setdefault("batches", {})["azure-batch|%s|%s|1" % (voice, model)] = info
        if pieces is None:
            log("  azure batch %s: could not cut" % sep)
            R.save()
            continue
        for i, (ln, pc) in enumerate(zip(lines, pieces)):
            m2 = dict(meta, model=model, voice=voice, take=1, provider="azure", batched=len(lines), batch_clip=rel,
                      piece=i, audio_s=round(len(acoustics.pcm(pc)[0]) / float(rate), 2))
            R.record(ln, voice, model, "azure-batch", 1, texts[i], key, pc, m2, context=True, sub="_p%d" % i)
        R.save()
        log("  azure batch sep=%s: %d segments in ONE request (%d chars vs %d one per request)" % (
            sep, len(lines), info["chars"], single))


def escape_xml(s):
    from xml.sax.saxutils import escape
    return escape(s)


def score_en_az(rec):
    """English Azure voices: what a driver hears — naturalness, warmth, clarity, a natural pace — and the words; the
    nearness to the old Journey-F voice counts only a little (RouteMind wants a voice of its own)."""
    if not rec.get("check"):
        return None
    j = rec["judge"] or {}
    s = _n(j.get("naturalness")) + 0.75 * _n(j.get("warmth")) + 0.75 * _n(j.get("clarity"))
    s += 0.25 * _n(j.get("sim_overall"))
    s -= 1.5 * ((j.get("pace") or "natural").lower() != "natural")
    s -= 3.0 * (not rec["check"]["text_ok"])
    s -= 0.5 * len([p for p in (j.get("problems") or []) if p])
    return round(s, 2)


def run_azure(R, args):
    stages = (args.azure_stages or "tune,final,screen,english,batch").split(",")
    from tools.voice_audition import azure as AZR
    if args.azure_cap:
        AZR.ledger().cap = args.azure_cap
    lines = {ln["id"]: ln for ln in all_lines()}
    ar = [probe_ar()] + arabic_lines()
    en = [probe_en()] + english_lines()
    if "tune" in stages:
        log("== Azure, Arabic settings: %d settings × %d lines × %d voices" % (len(AZ_AR_TUNE), len(AZ_TUNE_LINES),
                                                                           len(AZ_AR)))
        for v in AZ_AR:
            for cfg in AZ_AR_TUNE:
                for lid in AZ_TUNE_LINES:
                    az_take(R, lines[lid], v, cfg, "azure-tune")
    if "final" in stages:
        for v in AZ_AR:
            log("== Azure, Arabic, %s on every line (%s)" % (v, az_model(AZ_AR_FINAL[v])))
            for ln in ar:
                az_take(R, ln, v, AZ_AR_FINAL[v], "azure")
    if "screen" in stages:
        log("== Azure, English screen: %d voices on the Journey-F line" % len(AZ_EN_SCREEN))
        for v, cfg in AZ_EN_SCREEN:
            az_take(R, probe_en(), v, cfg, "azure-screen", ref=EN_REFS[0][1])
    if "english" in stages:
        for v, cfg in AZ_EN_PICK:
            log("== Azure, English, %s on every line" % v)
            for ln in en:
                az_take(R, ln, v, cfg, "azure")
    if "batch" in stages:
        v = AZ_BANK_AR
        log("== Azure, the batching measurement (%s)" % v)
        az_batch_test(R, v, AZ_AR_FINAL[v])
    if "judge" in stages:
        groups = set((args.azure_groups or "azure,azure-tune,azure-screen,azure-batch,azure-batch-single").split(","))
        log("== judging the Azure takes (%s; %d clips per request)" % (",".join(sorted(groups)), JUDGE_BATCH))
        judge_pending(R, groups=groups, force=args.judge_model)
    R.state["azure_ledger"] = {k: AZR.ledger().d[k] for k in ("chars", "requests", "failed", "by_purpose")}
    R.save()


# ════════════════════════════════════════════════════════════════════════════════════════════════
# SEKKA — the assistant's LIVE voice (2026-10-17, docs/azure-voice-2026-10.md §8)
#   python -m tools.voice_audition.audition sekka [--sekka-stages en-screen,en-degree,en-final,ar-azure,judge,
#          gemini,judge-gemini,merge] [--no-phones] [--sekka-log FILE]
# English: candidate voices WITH a speaking style at a moderate degree, on Sekka's six replies; Arabic: Azure ar-EG on
# the committed runtime text vs the fully diacritized text (+ a lexicon), against Gemini Despina (≤ 3 TTS requests,
# guarded below). Azure takes are counted in a SIDE ledger (azure/ledger_sekka.json, cap SEKKA_AZ_CAP) while the
# navigation bank renders on the same resource — its process holds the main ledger in memory — and merged into the
# main ledger once (`merge`).
# ════════════════════════════════════════════════════════════════════════════════════════════════
SEKKA_AZ_CAP = 8000              # this work's own Azure characters, all runs (the month is shared with the bank)
SEKKA_GEMINI_VOICE = "Despina"   # the audition's (and the Arabic bank's) Gemini voice, gemini-3.8-flash-tts
SEKKA_GEMINI_MAX = 3             # TTS requests this test may EVER make (the free tier: 10 a day, the bank's)
SEKKA_JUDGE_BATCH = 3            # clips per judge request (today's judges are the flash-lite family)
# English, screened on the traffic answer (the longest reply): the committed default, warm voices without a style,
# and voices WITH an upbeat style (Azure's StyleList on this resource: voices.json).
SEKKA_EN_SCREEN = [
    ("en-US-BrianMultilingualNeural", {}),          # the default committed with Sekka's Azure voice (dbf6487)
    ("en-US-AndrewMultilingualNeural", {}),
    ("en-US-AvaMultilingualNeural", {}),
    ("en-US-DavisNeural", {"style": "friendly"}),
    ("en-US-DavisNeural", {"style": "chat"}),
    ("en-US-JasonNeural", {"style": "friendly"}),
    ("en-US-AriaNeural", {"style": "friendly"}),
    ("en-US-NancyMultilingualNeural", {"style": "friendly"}),
    ("en-US-Ethan:MAI-Voice-2-Flash", {"style": "happy"}),
    ("en-US-Harper:MAI-Voice-2-Flash", {"style": "happy"}),
    # round 2: the voices Microsoft's own tags call upbeat / cheerful / bright (voices/list VoicePersonalities)
    ("en-US-EmmaMultilingualNeural", {}),           # Cheerful, Light-Hearted, Casual
    ("en-US-KaiNeural", {"style": "conversation"}),  # Sincere, Pleasant, Bright, Clear, Friendly, Warm (Assistant)
    ("en-US-PhoebeMultilingualNeural", {}),         # youthful, upbeat, confident — the English guidance bank's voice
    # round 3: a livelier style at a MODERATE degree (the degree probe: Aria and Ethan render differently at 0.5 / 1.5;
    # Ethan's "happy" at its own degree took 7.3 s for a 31-letter line — the MAI voices are out)
    ("en-US-AriaNeural", {"style": "cheerful", "degree": "0.6"}),
    ("en-US-DavisNeural", {"style": "cheerful", "degree": "0.6"}),
    ("en-US-JennyNeural", {"style": "cheerful", "degree": "0.6"}),
]
# Does the voice honour styledegree? The same short line at 0.5 / 1 / 1.5 (identical audio = the degree is ignored).
SEKKA_DEGREE_LINE = "bye"
SEKKA_DEGREES = ["0.5", "1.5"]
SEKKA_DEGREE_VOICES = [("en-US-AriaNeural", "friendly"), ("en-US-Ethan:MAI-Voice-2-Flash", "happy")]
# The finalists on all six replies — chosen from the screen (docs/azure-voice-2026-10.md §8 has the ranking).
# (the traffic answer's 6-voice ranking put Andrew, Jenny-cheerful-0.6, Aria-friendly first; over all six replies, in
# both clip orders — 12 rankings — Aria-friendly came first: Borda 29 vs Jenny 24 vs Andrew 19, "upbeat" 12 / 12,
# Andrew "calm" 6 / 12)
SEKKA_EN_FINAL = [
    ("en-US-AriaNeural", {"style": "friendly"}),                    # THE PICK
    ("en-US-JennyNeural", {"style": "cheerful", "degree": "0.6"}),  # a livelier style at a moderate degree
    ("en-US-AndrewMultilingualNeural", {}),                         # no style; male, as Sekka's Arabic grammar
]
SEKKA_EN_PICK = ("en-US-AriaNeural", {"style": "friendly"})   # the compiled default and the live row (tts.ts)
# The comparative ranking (one request per reply: the same reply by each voice of the set).
SEKKA_RANK_SET = [
    ("en-US-BrianMultilingualNeural", {}),
    ("en-US-AndrewMultilingualNeural", {}),
    ("en-US-EmmaMultilingualNeural", {}),
    ("en-US-AriaNeural", {"style": "friendly"}),
    ("en-US-DavisNeural", {"style": "cheerful", "degree": "0.6"}),
    ("en-US-JennyNeural", {"style": "cheerful", "degree": "0.6"}),
]
SEKKA_RANK_LINES = ["traffic", "greet", "stop", "search", "reroute", "bye"]
# Arabic on Azure: the committed runtime text, then the diacritized text, then + the lexicon (SEKKA_LEX).
SEKKA_AR_AZ = [
    ("ar-EG-ShakirNeural", {"spell": "sekka-runtime", "rate": "+20%", "pitch": "+2%"}),
    ("ar-EG-ShakirNeural", {"spell": "sekka-diac", "rate": "+20%", "pitch": "+2%"}),
]
# word (as written in `diac`, pausal applied) → the spelling that fixed it. The phone listener on the diacritized
# takes: «جامْدَة عَلى» came back «gamda-tun ʕala» and «بَنْزينَة وَطَنِيَّة» «banziːna-ti …» — a Modern-Standard
# ending on the ة before the next word; Cairo says «gamda», «banzīna». Two spellings tried: ه for ة (the phone
# listener heard «gamdeh ʕala» — the ending gone, a breathy h left) and ا for ة.
SEKKA_LEXES = {
    "sekka-diac-lex": {"جامْدَة": "جامْدَه", "بَنْزينَة": "بَنْزينَه"},
    "sekka-diac-lex2": {"جامْدَة": "جامْدا", "بَنْزينَة": "بَنْزينا"},
}
SEKKA_LEX = SEKKA_LEXES["sekka-diac-lex"]
_SK_AR_WORD = re.compile("[ء-ي٠-٩ٰ-ۓً-ٟ]+")   # azure_tts.ts AR_WORD


def sekka_runtime_ar(text):
    """The text Sekka's runtime sends Azure for an Arabic sentence (azure_tts.ts arabicForAzure: RESPELL_AR on whole
    words, then the pausal sukun) — the TS test holds the two to one output."""
    from tools.voice_audition import azure as AZR
    rs = AZR.runtime_respell()
    return pausal(_SK_AR_WORD.sub(lambda m: rs.get(m.group(0), m.group(0)), text))


def sekka_lex(text, spell="sekka-diac-lex"):
    lex = SEKKA_LEXES[spell]
    out = []
    for tok in text.split(" "):
        m = re.match(r"^([«(\"]*)(.*?)([.,!?؟،؛:»)\"…]*)$", tok)
        a, w, z = m.groups()
        out.append(a + lex.get(w, w) + z)
    return " ".join(out)


def _sekka_side_ledger():
    """The Azure ledger of this work alone (merged into the main one by `merge`)."""
    from tools.voice_audition import azure as AZR
    if getattr(AZR.LEDGER, "path", "").endswith("ledger_sekka.json"):
        return AZR.LEDGER
    L = AZR.Ledger.__new__(AZR.Ledger)
    os.makedirs(AZR.AZ, exist_ok=True)
    L.path = os.path.join(AZR.AZ, "ledger_sekka.json")
    L.d = json.load(io.open(L.path, encoding="utf-8")) if os.path.exists(L.path) else {
        "chars": 0, "requests": 0, "failed": 0, "by_day": {}, "by_purpose": {}, "by_month": {}, "errors": [],
        "merged_chars": 0, "merged_requests": 0, "merged_by_purpose": {}}
    L.cap = SEKKA_AZ_CAP
    AZR.LEDGER = L
    return L


def sekka_merge_ledger():
    """Add what the side ledger spent since the last merge to the main ledger (azure/ledger.json) — run when no other
    process holds the main ledger (the bank's daily run writes it from memory)."""
    from tools.voice_audition import azure as AZR
    side = _sekka_side_ledger()
    main_p = os.path.join(AZR.AZ, "ledger.json")
    M = json.load(io.open(main_p, encoding="utf-8"))
    dc = side.d["chars"] - side.d.get("merged_chars", 0)
    dr = side.d["requests"] - side.d.get("merged_requests", 0)
    if dc <= 0 and dr <= 0:
        log("sekka ledger: nothing new to merge (%d characters already merged)" % side.d.get("merged_chars", 0))
        return
    today = datetime.date.today().isoformat()
    M["chars"] += dc
    M["requests"] += dr
    M["by_day"][today] = M["by_day"].get(today, 0) + dc
    M["by_month"][today[:7]] = M["by_month"].get(today[:7], 0) + dc
    done = side.d.setdefault("merged_by_purpose", {})
    for p, v in side.d["by_purpose"].items():
        add = v - done.get(p, 0)
        if add:
            M["by_purpose"][p] = M["by_purpose"].get(p, 0) + add
            done[p] = v
    io.open(main_p, "w", encoding="utf-8").write(AZR.scrub(json.dumps(M, ensure_ascii=False, indent=1)))
    side.d["merged_chars"] = side.d["chars"]
    side.d["merged_requests"] = side.d["requests"]
    side.save()
    log("sekka ledger: merged %d characters / %d requests into %s (main total %d)" % (dc, dr, main_p, M["chars"]))


def _sk_lines(lang):
    return [ln for ln in sekka_lines() if ln["lang"] == lang]


def _sk_line(lang, key):
    return next(ln for ln in sekka_lines() if ln["lang"] == lang and ln["key"] == key)


def sekka_gemini_call(R, prompt, kind):
    """ONE generateContent on gemini-3.8-flash-tts with Despina (cached like `synth`; no second request on a
    no-audio answer) — and never more than SEKKA_GEMINI_MAX requests in this test's life (R.state counts them
    BEFORE each is sent)."""
    key = sha(MAIN_MODEL, SEKKA_GEMINI_VOICE, prompt)
    wav_p = os.path.join(CACHE, key + ".wav")
    meta_p = os.path.join(CACHE, key + ".json")
    if os.path.exists(wav_p) and os.path.exists(meta_p):
        return key, open(wav_p, "rb").read(), json.load(io.open(meta_p, encoding="utf-8"))
    used = R.state.setdefault("sekka_gemini_tts", [])
    if len(used) >= SEKKA_GEMINI_MAX:
        raise Exhausted("Sekka's Despina test has used its %d TTS requests" % SEKKA_GEMINI_MAX)
    used.append({"at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": kind})
    R.save()
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                                     "voiceName": SEKKA_GEMINI_VOICE}}}}}
    t0 = time.time()
    r = call(MAIN_MODEL, body, "tts")
    lat = time.time() - t0
    parts = ((r.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    aud = [p["inlineData"] for p in parts if "inlineData" in p]
    if not aud:
        used[-1]["result"] = "no audio"
        R.save()
        raise RuntimeError("no audio from %s/%s" % (MAIN_MODEL, SEKKA_GEMINI_VOICE))
    wav_b = to_wav(aud[0].get("mimeType", ""), b"".join(base64.b64decode(a["data"]) for a in aud))
    x, rate = acoustics.pcm(wav_b)
    meta = {"model": MAIN_MODEL, "voice": SEKKA_GEMINI_VOICE, "take": 1, "mime": aud[0].get("mimeType", ""),
            "latency_s": round(lat, 2), "usage": r.get("usageMetadata"), "audio_s": round(len(x) / float(rate), 2),
            "at": datetime.datetime.now().isoformat(timespec="seconds"), "provider": "gemini", "kind": kind}
    LIM.m(MAIN_MODEL)["audio_s"] += meta["audio_s"]
    LIM.save()
    os.makedirs(CACHE, exist_ok=True)
    open(wav_p, "wb").write(wav_b)
    io.open(meta_p, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False))
    used[-1]["result"] = "ok %.1f s audio in %.1f s" % (meta["audio_s"], lat)
    R.save()
    return key, wav_b, meta


def sekka_gemini(R):
    """(b) Despina live: the six Arabic replies in ONE request (the audition's batching: prompt_ar_batch, the
    aligned cut) for the quality, then two replies alone (prompt_ar) for the time to first audio."""
    lines = _sk_lines("ar")
    grp = "sekka-gemini"
    if not all(R.tid(grp, ln["id"], SEKKA_GEMINI_VOICE, MAIN_MODEL, 1) in R.takes for ln in lines):
        texts = [say_text(ln) for ln in lines]
        prompt = prompt_ar_batch("\n".join(texts), len(lines))
        try:
            key, wav_raw, meta = sekka_gemini_call(R, prompt, "batch")
        except (Exhausted, RuntimeError) as e:
            log("  Sekka / Despina batch MISSING — %s" % scrub(e)[:160])
            R.state["sekka_gemini_error"] = scrub(e)[:200]
            R.save()
            return
        rel = "clips/batch_%s.mp3" % key[:16]
        encode_mp3(processed(wav_raw), os.path.join(OUT, rel))
        pieces = split_on_silence(wav_raw, len(lines), weights=[letters_of(t) for t in texts])
        R.state.setdefault("batches", {})["%s|%s|%s|1" % (grp, SEKKA_GEMINI_VOICE, MAIN_MODEL)] = {
            "group": grp, "voice": SEKKA_GEMINI_VOICE, "model": MAIN_MODEL, "take": 1, "n": len(lines), "key": key,
            "clip": rel, "audio_s": meta.get("audio_s"), "latency_s": meta.get("latency_s"), "usage": meta.get("usage"),
            "cut_ok": pieces is not None, "lines": [ln["id"] for ln in lines]}
        if pieces is None:
            log("  Sekka / Despina batch: could not cut %d replies (%s)" % (len(lines), rel))
        else:
            rate = acoustics.pcm(wav_raw)[1]
            for i, (ln, pc) in enumerate(zip(lines, pieces)):
                m2 = dict(meta, batched=len(lines), batch_clip=rel, piece=i,
                          audio_s=round(len(acoustics.pcm(pc)[0]) / float(rate), 2))
                R.record(ln, SEKKA_GEMINI_VOICE, MAIN_MODEL, grp, 1, texts[i], key, pc, m2, sub="_p%d" % i)
        R.save()
    for k in ("greet", "traffic"):
        ln = _sk_line("ar", k)
        if R.tid("sekka-gemini-single", ln["id"], SEKKA_GEMINI_VOICE, MAIN_MODEL, 1) in R.takes:
            continue
        text = say_text(ln)
        try:
            key, wav_raw, meta = sekka_gemini_call(R, prompt_ar(text), "single " + k)
        except (Exhausted, RuntimeError) as e:
            log("  Sekka / Despina single %s MISSING — %s" % (k, scrub(e)[:160]))
            continue
        R.record(ln, SEKKA_GEMINI_VOICE, MAIN_MODEL, "sekka-gemini-single", 1, text, key, wav_raw, meta)
        log("  Despina single %s: %.1f s audio, first audio after %.2f s" % (k, meta["audio_s"], meta["latency_s"]))


def judge_sekka(R, groups, batch=None):
    """Every take of `groups` without a verdict, `batch` clips per request, the Sekka prompts ("ar-sekka" /
    "en-sekka"). A request whose clips all came back with ONE transcript (the flash-lite failure the bank saw) is
    thrown away and heard again one clip at a time."""
    batch = batch or SEKKA_JUDGE_BATCH
    lines = {ln["id"]: ln for ln in all_lines()}
    todo = [r for r in R.takes.values() if r["group"] in groups and r.get("judge") is None]
    todo.sort(key=lambda r: (r["lang"], r["group"], r["voice"], r["model"], r["line"]))
    left = 0
    for lang in ("ar", "en"):
        recs = [r for r in todo if r["lang"] == lang]
        i = 0
        while i < len(recs):
            chunk = recs[i:i + batch]
            jl = lang + "-sekka"
            try:
                vs = judge_batch(jl, [heard_mp3(r) for r in chunk])
            except Exhausted as e:
                left += len(recs) - i
                log("  sekka judge stopped: %s (%d takes wait)" % (scrub(e)[:160], len(recs) - i))
                break
            ts = [(v.get("transcript") or "").strip() for v in vs]
            if len(chunk) > 1 and len(set(ts)) == 1:
                log("  sekka judge: one transcript for %d clips — heard again one by one" % len(chunk))
                for r in chunk:
                    p = os.path.join(CACHE, "verdict_%s.json" % _verdict_key(jl, heard_mp3(r), None,
                                                                             JUDGE_FORCE["model"]))
                    if os.path.exists(p):
                        os.remove(p)
                vs = []
                try:
                    for r in chunk:
                        vs += judge_batch(jl, [heard_mp3(r)])
                except Exhausted as e:
                    left += len(recs) - i
                    log("  sekka judge stopped: %s" % scrub(e)[:160])
                    break
            for r, v in zip(chunk, vs):
                apply_verdict(r, lines[r["line"]], v)
                log("  %-4s %s %s %s  %s  %s" % ("PASS" if r["pass"] else "fail", r["line"], r["voice"], r["model"],
                                                (v.get("transcript") or "")[:60], ";".join(r["check"]["problems"])[:60]))
            R.save()
            i += batch
    return left


SEKKA_RANK_PROMPT = (
    "You will hear {n} clips, each announced by a text label \"CLIP k\": the SAME sentence spoken by {n} different "
    "voices of an in-car voice ASSISTANT. Compare the voices with each other. The ideal voice for this assistant is "
    "WARM AND UPBEAT — a friendly person who is glad to help — but NOT over the top (not salesy, not cartoonish, not "
    "a radio host) and NOT sleepy, flat or bored. Report only what you HEAR.\n"
    "Return a JSON object with \"clips\": an array of exactly {n} objects in clip order, each with \"clip\" (its "
    "number), \"gender\" (\"female\" / \"male\"), \"energy\" (1-10: 1 sleepy, 4 calm but low, 6-8 warm and upbeat, "
    "10 over the top), \"warmth\" (1-10), \"naturalness\" (1-10), \"feel\" (one of \"sleepy\", \"calm\", \"upbeat\", "
    "\"over the top\"), \"notes\" (one short sentence); and \"ranking\": the clip numbers from the BEST fit for the "
    "ideal to the worst.")


def sekka_rank_en(R, keys):
    """ONE comparative request: the same reply by several voices, ranked against the owner's brief (absolute 1-10
    verdicts of single clips came back flat — every voice 6 / 7 / 8)."""
    recs = [R.takes[k] for k in keys]
    wavs = [heard_mp3(r) for r in recs]
    h = sha("sekka-rank", SEKKA_RANK_PROMPT, *[hashlib.sha256(w).hexdigest() for w in wavs])
    p = os.path.join(CACHE, "sekka_rank_%s.json" % h[:24])
    if os.path.exists(p):
        return json.load(io.open(p, encoding="utf-8"))
    parts = []
    for i, w in enumerate(wavs):
        parts += [{"text": "CLIP %d" % (i + 1)}, bank_tts.audio_part(w, "audio/mpeg")]
    parts.append({"text": SEKKA_RANK_PROMPT.format(n=len(wavs))})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}}
    last = None
    for jm in [m for m in JUDGE_CHAIN if m not in _DEAD_JUDGES]:
        try:
            r = call(jm, body, "judge", max5xx=2)
        except (Exhausted, Overloaded, RuntimeError) as e:
            last = e
            continue
        txt = "".join(pp.get("text", "") for pp in ((r.get("candidates") or [{}])[0].get("content") or {})
                      .get("parts", []) if not pp.get("thought"))
        try:
            out = json.loads(txt)
        except ValueError:
            last = RuntimeError("%s: not JSON" % jm)
            continue
        if not isinstance(out, dict) or len(out.get("clips") or []) != len(wavs):
            last = RuntimeError("%s: %s clips" % (jm, len(out.get("clips") or []) if isinstance(out, dict) else "no"))
            continue
        out["_judge_model"] = jm
        out["_takes"] = [[r["voice"], r["model"]] for r in recs]
        io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False))
        return out
    raise Exhausted("no judge ranked the clips (%s)" % scrub(last))


def sekka_rejudge_failed(R, groups):
    """A take whose words did not match after a multi-clip verdict is heard again ALONE: the flash-lite judges shift
    transcripts between the clips of one request (2026-10-17: the sign-off's transcript ended with the greeting's
    first words, the greeting's carried the reroute's). A take already heard alone keeps its verdict."""
    again = [r for r in R.takes.values() if r["group"] in groups and r.get("check")
             and not r["check"]["text_ok"] and (r.get("judge") or {}).get("_batch", 1) > 1]
    for r in again:
        jl = r["lang"] + "-sekka"
        p = os.path.join(CACHE, "verdict_%s.json" % _verdict_key(jl, heard_mp3(r), None, JUDGE_FORCE["model"]))
        if os.path.exists(p):
            os.remove(p)
        r["judge"], r["check"], r["pass"] = None, None, None
    if again:
        log("== %d Sekka takes heard again one by one (their multi-clip verdicts did not match their words)" % len(again))
        judge_sekka(R, groups, batch=1)


def score_sekka_en(rec):
    """English Sekka: naturalness, warmth, clarity, and the owner's ENERGY target — warm and upbeat (7 of 10), not
    sleepy, not over the top; the words must match."""
    if not rec.get("check"):
        return None
    j = rec["judge"] or {}
    e = _n(j.get("energy"), 5.0)
    s = _n(j.get("naturalness")) + 0.75 * _n(j.get("warmth")) + 0.5 * _n(j.get("clarity"))
    s += 2.0 - abs(e - 7.0)
    s -= 1.5 * ((j.get("pace") or "natural").lower() != "natural")
    s -= 3.0 * (not rec["check"]["text_ok"])
    s -= 0.5 * len([p for p in (j.get("problems") or []) if p])
    return round(s, 2)


def score_sekka_ar(rec):
    s = score_ar(rec)
    if s is None:
        return None
    ce = ((rec.get("judge") or {}).get("case_endings") or "none").lower()
    return round(s - {"none": 0.0, "some": 2.0, "many": 4.0}.get(ce, 0.0), 2)


def _p50(xs):
    v = sorted(x for x in xs if isinstance(x, (int, float)))
    if not v:
        return None
    m = len(v) // 2
    return round(v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2.0, 2)


# cost model (docs/copilot-models-2026-10.md §6: a conversation = 3 voice turns; here a turn speaks ONE reply of the
# six's average length). Prices read 2026-10-17: Gemini API pricing page (updated 2026-10-16) — gemini-3.8-flash-tts
# input $0.50 / 1M text tokens, output $9.00 / 1M audio tokens through 2026-12-31 ($1.00 / $18.00 from 2027-01-01),
# "25 tokens per second of audio"; OpenAI list prices (developers.openai.com, read 2026-10-16) — gpt-6-luna $0.10 /
# $0.50, gpt-5.4-nano $0.20 / $1.25 per 1M input / output tokens; Azure F0: 500,000 characters a month at $0.
SK_TURNS = 3
SK_GEMINI_IN, SK_GEMINI_OUT, SK_GEMINI_TPS = 0.50e-6, 9.00e-6, 25
SK_DIAC_MODELS = {"gpt-6-luna": (0.10e-6, 0.50e-6), "gpt-5.4-nano": (0.20e-6, 1.25e-6)}
SK_DIAC_PROMPT_TOKENS = 220      # an instruction to diacritize the Cairene way (estimate: the call was not made)


def sekka_summary(R):
    """What the Sekka tab shows (computed here, so the page stays a renderer)."""
    takes = [r for r in R.takes.values() if r["group"].startswith("sekka-")]

    def by(group, voice=None, model=None):
        return [r for r in takes if r["group"] == group and (voice is None or r["voice"] == voice)
                and (model is None or r["model"] == model)]

    def lat(rs):
        return _p50([(r.get("meta") or {}).get("latency_s") for r in rs])

    en_rows = []
    for v, cfg in SEKKA_EN_SCREEN:
        m = az_model(cfg)
        rs = by("sekka-en-screen", v, m)
        r = rs[0] if rs else None
        j = (r or {}).get("judge") or {}
        en_rows.append({"voice": v, "cfg": cfg, "model": m, "made": bool(r), "score": score_sekka_en(r) if r else None,
                        "energy": j.get("energy"), "warmth": j.get("warmth"), "naturalness": j.get("naturalness"),
                        "clarity": j.get("clarity"), "pace": j.get("pace"), "gender": j.get("gender"),
                        "problems": j.get("problems"), "notes": j.get("notes"),
                        "judge_model": j.get("_judge_model"), "text_ok": ((r or {}).get("check") or {}).get("text_ok"),
                        "latency_s": ((r or {}).get("meta") or {}).get("latency_s"),
                        "chars": ((r or {}).get("meta") or {}).get("chars"),
                        "rate_lps": ((r or {}).get("stats") or {}).get("rate_lps"),
                        "f0_med": ((r or {}).get("stats") or {}).get("f0_med"),
                        "f0_iqr": ((r or {}).get("stats") or {}).get("f0_iqr"), "clip": (r or {}).get("clip")})
    deg = []
    for r in by("sekka-en-degree"):
        deg.append({"voice": r["voice"], "model": r["model"], "clip": r["clip"],
                    "sha": hashlib.sha256(open(os.path.join(CACHE, r["key"] + ".wav"), "rb").read()).hexdigest()[:12]
                    if os.path.exists(os.path.join(CACHE, r["key"] + ".wav")) else None,
                    "stats": r.get("stats"), "chars": (r.get("meta") or {}).get("chars")})
    finals = []
    for v, cfg in SEKKA_EN_FINAL:
        m = az_model(cfg)
        rs = [r for r in takes if r["group"] in ("sekka-en", "sekka-en-screen") and r["voice"] == v
              and r["model"] == m]
        sc = [score_sekka_en(r) for r in rs if r.get("judge")]
        ens = [e for e in (_n((r.get("judge") or {}).get("energy"), None) for r in rs if r.get("judge"))
               if e is not None]
        finals.append({"voice": v, "cfg": cfg, "model": m, "n": len(rs), "judged": len(sc),
                       "score": round(sum(sc) / len(sc), 2) if sc else None,
                       "energy": round(sum(ens) / len(ens), 1) if ens else None,
                       "warmth": _p50([_n((r.get("judge") or {}).get("warmth"), None) for r in rs]),
                       "rate_lps": _p50([(r.get("stats") or {}).get("rate_lps") for r in rs]),
                       "f0_iqr": _p50([(r.get("stats") or {}).get("f0_iqr") for r in rs]),
                       "read_back": sum(1 for r in rs if (r.get("check") or {}).get("text_ok")),
                       "latency_p50": lat(rs), "chars": sum((r.get("meta") or {}).get("chars") or 0 for r in rs),
                       "pick": SEKKA_EN_PICK is not None and (v, cfg) == SEKKA_EN_PICK})
    # Arabic: one row per variant
    n_rep = len(SEKKA_REPLIES)
    ar = []

    def ar_row(label, rs, how, ttfa, cost, note=""):
        js = [r for r in rs if r.get("judge")]
        ph = [r for r in rs if r.get("phones") is not None]
        flags = [f for r in rs for f in ((r.get("phones") or {}).get("hard") or [])]
        return {"label": label, "how": how, "n": len(rs), "of": n_rep, "judged": len(js),
                "read_back": sum(1 for r in js if (r.get("check") or {}).get("text_ok")),
                "dialect_ok": sum(1 for r in js if r.get("dialect_ok")),
                "accent": round(sum(_n(r["judge"].get("accent_score")) for r in js) / len(js), 1) if js else None,
                "jim": sorted({str(r["judge"].get("jim")) for r in js}), "qaf": sorted({str(r["judge"].get("qaf")) for r in js}),
                "case": sorted({str(r["judge"].get("case_endings")) for r in js}),
                "warmth": round(sum(_n(r["judge"].get("warmth")) for r in js) / len(js), 1) if js else None,
                "natural": round(sum(_n(r["judge"].get("naturalness")) for r in js) / len(js), 1) if js else None,
                "phones": len(ph), "flags": flags, "score": round(sum(score_sekka_ar(r) for r in js) / len(js), 2)
                if js else None, "ttfa": ttfa, "cost": cost, "note": note,
                "judge_models": sorted({(r.get("judge") or {}).get("_judge_model") for r in js} - {None})}

    from tools.voice_audition import azure as AZR
    for v, cfg in SEKKA_AR_AZ + [("ar-EG-ShakirNeural", {"spell": s, "rate": "+20%", "pitch": "+2%"})
                                 for s in SEKKA_LEXES]:
        m = az_model(cfg)
        rs = by("sekka-ar-azure", v, m)
        chars = [(r.get("meta") or {}).get("chars") for r in rs]
        cpr = sum(c for c in chars if c) / max(1, len([c for c in chars if c])) if rs else None
        diac = cfg["spell"] != "sekka-runtime"
        cost = {"azure_chars_per_conversation": round(cpr * SK_TURNS) if cpr else None, "azure_usd": 0.0}
        if diac:
            # the diacritizing model's tokens (estimated: ~2.5 characters a token in, ~2 out for marked Arabic)
            tin = SK_DIAC_PROMPT_TOKENS + sum(len(ln["plain"]) for ln in _sk_lines("ar")) / 2.5 / n_rep
            tout = sum(len(ln["diac"]) for ln in _sk_lines("ar")) / 2.0 / n_rep
            cost["diac_usd"] = {k: round(SK_TURNS * (tin * a + tout * b), 6) for k, (a, b) in SK_DIAC_MODELS.items()}
            cost["diac_tokens_per_reply"] = [round(tin), round(tout)]
        how = {"sekka-runtime": "the committed runtime text (RESPELL_AR + pausal sukun) — no diacritizer",
               "sekka-diac": "FULLY DIACRITIZED (the desk's best-case marking) + pausal sukun",
               "sekka-diac-lex": "diacritized + the lexicon (ة → ه on the 2 words that still broke)",
               "sekka-diac-lex2": "diacritized + the lexicon (ة → ا on the 2 words that still broke)"}[cfg["spell"]]
        ar.append(ar_row("Azure %s %s" % (v.replace("Neural", ""), m.replace("azure ", "")), rs, how,
                         {"azure_p50_s": lat(rs), "diacritizer_s": "not measured" if diac else 0}, cost))
    # (a) as the owner would run it: every reply diacritized, the lexicon where it changed the reply
    lexm = az_model({"spell": "sekka-diac-lex", "rate": "+20%", "pitch": "+2%"})
    diacm = az_model({"spell": "sekka-diac", "rate": "+20%", "pitch": "+2%"})
    comb = []
    for ln in _sk_lines("ar"):
        r = R.takes.get(R.tid("sekka-ar-azure", ln["id"], "ar-EG-ShakirNeural", lexm, 1)) or \
            R.takes.get(R.tid("sekka-ar-azure", ln["id"], "ar-EG-ShakirNeural", diacm, 1))
        if r:
            comb.append(r)
    if comb:
        base = next((a for a in ar if "spell=sekka-diac " in a["label"] + " "), None)
        ar.insert(0, ar_row("(a) Azure Shakir — diacritized + lexicon, all six", comb,
                            "every reply fully diacritized (Cairene vowels, ق → ء, ث → س, no case endings) + the "
                            "pausal sukun + the lexicon (ة → ه) on the 2 replies it changes",
                            {"azure_p50_s": lat(comb), "diacritizer_s": "not measured (≈ 0.5-1 s expected)"},
                            dict(base["cost"]) if base else {},
                            note="the OpenAI diacritizer was not run: a temporary function holding the copilot's "
                                 "key was refused by the desk's permission system — the marking is the desk's own "
                                 "(best case)"))
    gb = by("sekka-gemini", SEKKA_GEMINI_VOICE, MAIN_MODEL)
    gs = by("sekka-gemini-single", SEKKA_GEMINI_VOICE, MAIN_MODEL)
    gcost = None
    if gs:
        tin = [_n(((r.get("meta") or {}).get("usage") or {}).get("promptTokenCount"), None) for r in gs]
        tin = [t for t in tin if t]
        aud = [(r.get("meta") or {}).get("audio_s") for r in (gs + gb)]
        aud = [a for a in aud if a]
        if tin and aud:
            per = (sum(tin) / len(tin)) * SK_GEMINI_IN + (sum(aud) / len(aud)) * SK_GEMINI_TPS * SK_GEMINI_OUT
            gcost = {"usd_per_conversation": round(SK_TURNS * per, 5), "prompt_tokens_per_reply": round(sum(tin) / len(tin)),
                     "audio_s_per_reply": round(sum(aud) / len(aud), 2)}
    ttfa_b = {"gemini_p50_s": lat(gs), "n": len(gs)}
    how_b = ("the audition's Despina settings: prompt_ar_batch (6 replies, one request) / prompt_ar (alone), "
             "pronounce.py SAY")
    live = [r for r in gb if (r.get("meta") or {}).get("bank_request")]
    if live and not gs:
        # 2026-10-17: measured at ZERO extra requests — the six replies rode at the end of a navigation-bank request
        # (tools/voice_bank/sekka_live.py), STREAMED: the time to first audio is that request's first audio chunk
        m0 = live[0]["meta"]
        ttfa_b = {"gemini_p50_s": m0.get("first_audio_s"), "n": "1, streamed — the bank request's first audio chunk"}
        how_b = ("the audition's Despina text (pronounce.py SAY) at the end of one navigation-bank request of %s "
                 "lines (prompt_ar_batch), judged by the bank's quota-free judges: Azure STT read-back + the blind "
                 "phone listener (no accent / case-ending scores)" % m0.get("batched"))
        aud = [(r.get("meta") or {}).get("audio_s") for r in live if (r.get("meta") or {}).get("audio_s")]
        if aud:      # one live reply = one request: the director's notes (≈ 330 tokens, measured) + its audio
            per = 330 * SK_GEMINI_IN + (sum(aud) / len(aud)) * SK_GEMINI_TPS * SK_GEMINI_OUT
            gcost = {"usd_per_conversation": round(SK_TURNS * per, 5), "prompt_tokens_per_reply": 330,
                     "audio_s_per_reply": round(sum(aud) / len(aud), 2)}
    ar.append(ar_row("(b) Gemini %s · %s" % (SEKKA_GEMINI_VOICE, MAIN_MODEL), gb or gs, how_b, ttfa_b, gcost or {},
                     note=R.state.get("sekka_gemini_error") or ""))
    # the comparative rankings: Borda points over the finalists in every ranking (both clip orders)
    names = {"%s|%s" % (v, az_model(c)): (v, c) for v, c in SEKKA_EN_FINAL}
    agg = {n: {"borda": 0, "first": 0, "energy": [], "warmth": [], "natural": [], "feel": {}} for n in names}
    per = []
    for key, o in sorted((R.state.get("sekka_en_rank") or {}).items()):
        ids = ["|".join(t) for t in o.get("_takes") or []]
        rk = [ids[int(c) - 1] for c in o.get("ranking") or [] if str(c).isdigit() and 0 < int(c) <= len(ids)]
        per.append({"reply": key, "judge": o.get("_judge_model"),
                    "order": [ids[int(c) - 1].replace("en-US-", "").replace("|azure", " ") for c in
                              o.get("ranking") or [] if str(c).isdigit() and 0 < int(c) <= len(ids)]})
        fr = [n for n in rk if n in agg]
        for i, n in enumerate(fr):
            agg[n]["borda"] += len(fr) - i
        if fr:
            agg[fr[0]]["first"] += 1
        for n, c in zip(ids, o.get("clips") or []):
            if n in agg:
                for k_, f_ in (("energy", "energy"), ("warmth", "warmth"), ("natural", "naturalness")):
                    if isinstance(c.get(f_), (int, float)):
                        agg[n][k_].append(c[f_])
                agg[n]["feel"][c.get("feel")] = agg[n]["feel"].get(c.get("feel"), 0) + 1
    rank = [{"voice": names[n][0], "cfg": names[n][1], "borda": a["borda"], "first": a["first"],
             "energy": round(sum(a["energy"]) / len(a["energy"]), 2) if a["energy"] else None,
             "warmth": round(sum(a["warmth"]) / len(a["warmth"]), 2) if a["warmth"] else None,
             "natural": round(sum(a["natural"]) / len(a["natural"]), 2) if a["natural"] else None,
             "feel": a["feel"]} for n, a in agg.items()]
    sp = os.path.join(AZR.AZ, "ledger_sekka.json")     # read, never installed (write_page must not move the ledger)
    side = json.load(io.open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    return {"replies": [{"key": k, "label": lb, "ar": a, "diac": d, "en": e} for k, lb, a, _, d, e in SEKKA_REPLIES],
            "en_screen": en_rows, "degree": deg, "finals": finals, "rank": rank, "rank_per_reply": per,
            "pick": {"voice": SEKKA_EN_PICK[0], "cfg": SEKKA_EN_PICK[1], "model": az_model(SEKKA_EN_PICK[1])}
            if SEKKA_EN_PICK else None, "ar": ar,
            "lex": {s: " · ".join("%s → %s" % kv for kv in d.items()) for s, d in SEKKA_LEXES.items()},
            "gemini_used": R.state.get("sekka_gemini_tts") or [], "gemini_max": SEKKA_GEMINI_MAX,
            "gemini_schedule": R.state.get("sekka_gemini_schedule"), "ledger": {k: side.get(k) for k in (
                "chars", "requests", "failed", "by_purpose", "merged_chars")},
            "live": R.state.get("sekka_live") or {}}


def run_sekka(R, args):
    stages = (args.sekka_stages or "en-screen,en-degree,en-final,ar-azure,judge").split(",")
    _sekka_side_ledger()
    if "en-screen" in stages:
        ln = _sk_line("en", "traffic")
        log("== Sekka English screen: %d voices on «%s»" % (len(SEKKA_EN_SCREEN), ln["plain"][:50]))
        for v, cfg in SEKKA_EN_SCREEN:
            az_take(R, ln, v, cfg, "sekka-en-screen")
    if "en-degree" in stages:
        ln = _sk_line("en", SEKKA_DEGREE_LINE)
        for v, st in SEKKA_DEGREE_VOICES:
            for d in [None] + SEKKA_DEGREES:
                c = {"style": st}
                if d:
                    c["degree"] = d
                az_take(R, ln, v, c, "sekka-en-degree")
    if "en-final" in stages:
        for v, cfg in SEKKA_EN_FINAL:
            log("== Sekka English final: %s %s on the six replies" % (v, az_model(cfg)))
            for ln in _sk_lines("en"):
                if ln["key"] == "traffic" and cfg in [c for vv, c in SEKKA_EN_SCREEN if vv == v]:
                    continue                      # the screen's take of this reply is the same SSML
                az_take(R, ln, v, cfg, "sekka-en")
    if "ar-azure" in stages:
        variants = SEKKA_AR_AZ + [("ar-EG-ShakirNeural", {"spell": s, "rate": "+20%", "pitch": "+2%"})
                                  for s in SEKKA_LEXES]
        for v, cfg in variants:
            log("== Sekka Arabic on Azure: %s %s" % (v, az_model(cfg)))
            for ln in _sk_lines("ar"):
                if cfg["spell"] in SEKKA_LEXES and sekka_lex(pausal(ln["diac"]), cfg["spell"]) == pausal(ln["diac"]):
                    continue                      # the lexicon changes nothing in this reply
                az_take(R, ln, v, cfg, "sekka-ar-azure")
    if "gemini" in stages:
        log("== Sekka Arabic on Gemini: %s (at most %d TTS requests, %d used before this run)" % (
            SEKKA_GEMINI_VOICE, SEKKA_GEMINI_MAX, len(R.state.get("sekka_gemini_tts") or [])))
        # a scheduled run may wake a minute before the recorded reset (midnight Pacific + 1 min): wait it out
        # (≤ 15 min) instead of failing without a request
        for m in (MAIN_MODEL, JUDGE_MODEL):
            w = LIM.m(m)["blocked_until"] - time.time()
            if 0 < w <= 900:
                log("  %s: the quota day starts in %.0f s — waiting" % (m, w))
                time.sleep(w + 5)
        sekka_gemini(R)
    if "judge" in stages:
        log("== judging Sekka's Azure takes (%d clips per request)" % SEKKA_JUDGE_BATCH)
        judge_sekka(R, {"sekka-en-screen", "sekka-en", "sekka-ar-azure"})
    if "judge-gemini" in stages:
        # (a) and (b) heard by the SAME judge on the same day: the main judge (gemini-3.8-flash) when it answers —
        # the Azure takes were first judged by the flash-lite family (the only judges left on 2026-10-17)
        log("== judging Sekka's Arabic takes, (a) Azure and (b) Gemini, with one judge (%s)" % JUDGE_MODEL)
        groups = {"sekka-gemini", "sekka-gemini-single", "sekka-ar-azure"}
        # the Azure takes heard again: (a) as it would run (diacritized, the ه lexicon where it changes the reply)
        # and the committed runtime text — not the superseded variants (the judge's quota is the bank's too)
        keep = {az_model({"spell": s, "rate": "+20%", "pitch": "+2%"}) for s in ("sekka-runtime", "sekka-diac-lex")}
        lexed = {ln["id"] for ln in _sk_lines("ar") if sekka_lex(pausal(ln["diac"])) != pausal(ln["diac"])}
        keep_diac = az_model({"spell": "sekka-diac", "rate": "+20%", "pitch": "+2%"})

        def wanted(r):
            if r["group"] != "sekka-ar-azure":
                return True
            return r["model"] in keep or (r["model"] == keep_diac and r["line"] not in lexed)

        for r in R.takes.values():
            if r["group"] in groups and wanted(r) and r.get("judge") and \
                    r["judge"].get("_judge_model") != JUDGE_MODEL:
                r.setdefault("judge_first", r["judge"])        # the earlier verdict is kept beside it
                r["judge"], r["check"], r["pass"] = None, None, None
        R.save()
        JUDGE_FORCE["model"] = JUDGE_MODEL
        left = judge_sekka(R, groups, batch=6)                 # the main judge's measured size (§2 of the doc)
        JUDGE_FORCE["model"] = None
        if left:
            log("  %s did not answer for %d takes — the judge chain hears them" % (JUDGE_MODEL, left))
            judge_sekka(R, groups)
        for r in R.takes.values():                          # a take nobody could hear again keeps its first verdict
            if r["group"] in groups and r.get("judge") is None and r.get("judge_first"):
                apply_verdict(r, {ln["id"]: ln for ln in all_lines()}[r["line"]], r["judge_first"])
        sekka_rejudge_failed(R, groups)
    if "rejudge-fail" in stages:
        sekka_rejudge_failed(R, {"sekka-en-screen", "sekka-en", "sekka-ar-azure"})
    if "en-rank" in stages:
        # the same reply by the screen's contenders, compared in ONE request
        for key in SEKKA_RANK_LINES:
            ks = [R.tid(g, "sk_en_" + key, v, az_model(c), 1) for v, c in SEKKA_RANK_SET
                  for g in ("sekka-en-screen", "sekka-en") if R.tid(g, "sk_en_" + key, v, az_model(c), 1) in R.takes]
            if len(ks) < 2:
                continue
            # twice, the second time in the REVERSE clip order: a judge that favours a clip position (the first
            # pass put the voice in position 2 first on 4 of 6 replies) cancels out in the sum
            for tag, order in ((key, ks), (key + "_rev", list(reversed(ks)))):
                try:
                    out = sekka_rank_en(R, order)
                except Exhausted as e:
                    log("  sekka rank %s: %s" % (tag, scrub(e)[:160]))
                    continue
                R.state.setdefault("sekka_en_rank", {})[tag] = out
                log("  rank %s (%s): %s" % (tag, out["_judge_model"], " > ".join(
                    "%s %s" % (out["_takes"][int(c) - 1][0].replace("en-US-", ""), out["_takes"][int(c) - 1][1]
                               .replace("azure", "")) for c in out.get("ranking") or [] if 0 < int(c) <= len(order))))
                R.save()
    if "merge" in stages:
        sekka_merge_ledger()
    R.save()


# ════════════════════════════════════════════════════════════════════════════════════════════════
# RANKING
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _n(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def score_ar(rec):
    if not rec.get("check"):
        return None
    j = rec["judge"] or {}
    jim = (j.get("jim") or "none").lower()
    qaf = (j.get("qaf") or "none").lower()
    s = _n(j.get("accent_score")) + 0.5 * _n(j.get("warmth")) + 0.5 * _n(j.get("naturalness"))
    s -= 4.0 * (jim not in ("g", "none", ""))
    s -= 4.0 * (qaf not in ("glottal", "none", ""))
    s -= 3.0 * (not rec["check"]["text_ok"])
    s -= 1.5 * len((rec.get("phones") or {}).get("hard") or [])
    return round(s, 2)


def score_en(rec, ref_f0):
    if not rec.get("check"):
        return None
    j = rec["judge"] or {}
    s = _n(j.get("sim_overall")) + 0.5 * _n(j.get("warmth")) + 0.5 * _n(j.get("naturalness"))
    s -= 3.0 * (not rec["check"]["text_ok"])
    f0 = (rec.get("stats") or {}).get("f0_med")
    if f0 and ref_f0:
        s -= min(4.0, 12.0 * abs(math.log2(f0 / ref_f0)))   # 1 semitone ≈ 0.69 points
    return round(s, 2)


def best_take(run, group, line_id, voice, model):
    cands = [r for r in run.takes.values() if r["group"] == group and r["line"] == line_id and r["voice"] == voice
             and r["model"] == model and not r.get("raw_spelling")]
    if not cands:
        return None
    return sorted(cands, key=lambda r: (not r.get("pass"), not (r.get("check") or {}).get("text_ok"), r["take"]))[0]


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE RUN
# ════════════════════════════════════════════════════════════════════════════════════════════════
def ref_f0():
    st = acoustics.clip_stats(mp3_to_wav(os.path.join(ROOT, EN_REFS[0][1])), EN_REF_TEXT)
    return st["f0_med"]


def run(args):
    global LIM
    LIM = Limits()
    STATS["max_calls"] = args.max_calls
    if args.no_phones:
        PHONES["on"] = False
    R = Run()
    stages = args.stages.split(",")
    pa, pe = probe_ar(), probe_en()
    ar, en = arabic_lines(), english_lines()

    if "prescreen" in stages:
        log("== pre-screen, Arabic: %d voices on the probe line (%s)" % (len(AR_SCREEN), MAIN_MODEL))
        for v in AR_SCREEN:
            R.take(pa, v, MAIN_MODEL, "prescreen")
        enm = args.en_screen_model or MAIN_MODEL
        log("== pre-screen, English: %d voices on the Journey-F reference line (%s)" % (len(EN_SCREEN), enm))
        for v in EN_SCREEN:
            R.take(pe, v, enm, "prescreen", ref=EN_REFS[0][1])
    if "judge" in stages:
        log("== judging the pre-screen (one request per language)")
        judge_pending(R, groups={"prescreen"})

    # the picks — from the judged pre-screen; the September finalists when the judge could not answer yet
    R.state["ref_f0"] = R.state.get("ref_f0") or ref_f0()
    rerank(R)
    ar_rank, en_rank = R.state["ar_rank"], R.state["en_rank"]
    ar_judged = len(ar_rank) == sum(1 for r in R.takes.values() if r["group"] == "prescreen" and r["lang"] == "ar")
    en_judged = len(en_rank) == sum(1 for r in R.takes.values() if r["group"] == "prescreen" and r["lang"] == "en")
    if args.voices_ar:
        ar_pick, why_ar = args.voices_ar.split(","), "given"
    elif ar_rank and ar_judged:
        ar_pick, why_ar = [v for _, v in ar_rank[:3]], "judged"
        if INCUMBENT_AR not in ar_pick:
            ar_pick.append(INCUMBENT_AR)
    else:
        ar_pick, why_ar = R.state.get("ar_pick") or ["Zephyr", "Leda", "Sulafat", "Kore"], "unjudged (2026-09 finalists)"
    if args.voices_en:
        en_pick, why_en = args.voices_en.split(","), "given"
    elif en_rank and en_judged:
        en_pick, why_en = [v for _, v in en_rank[:3]], "judged"
    else:
        en_pick, why_en = R.state.get("en_pick") or ["Aoede", "Kore", "Leda"], "unjudged"
    R.state.update({"ar_pick": ar_pick, "en_pick": en_pick, "pick_basis": {"ar": why_ar, "en": why_en}})
    R.save()
    log("== picks: Arabic %s (%s) | English %s (%s)" % (ar_pick, why_ar, en_pick, why_en))
    best = ar_pick[0]

    # every audition take is made N lines per request (10 requests a day per TTS model on the free tier); the probe
    # line rides along, so each voice's batched probe sits beside its single-request pre-screen take
    if "compare" in stages:
        for model in COMPARE_MODELS:
            log("== model comparison, %s on %s (%d lines, one request)" % (best, model, len(ar) + 1))
            R.batched([pa] + ar, best, model, "compare")
    if "audition" in stages:
        log("== audition, Arabic, best voice %s: %d lines in one request" % (best, len(ar) + 1))
        R.batched([pa] + ar, best, MAIN_MODEL, "audition")
    if "english" in stages:
        for v in en_pick:
            log("== audition, English, %s: %d lines in one request" % (v, len(en) + 1))
            R.batched([pe] + en, v, MAIN_MODEL, "audition")
    if "audition" in stages:
        for v in ar_pick[1:]:
            log("== audition, Arabic, %s: %d lines in one request" % (v, len(ar) + 1))
            R.batched([pa] + ar, v, MAIN_MODEL, "audition")
    if "raw" in stages:
        log("== the respelling question: %s reads the probe with the ق written as ق" % best)
        R.take(pa, best, args.test_model or MAIN_MODEL, "raw-spelling", raw=True)
    if "batch" in stages:
        log("== the batching question: %d bank segments in ONE request, cut, judged inside a line (%s)"
            % (len(BATCH_SEGS), best))
        R.batched(batch_lines(), best, args.test_model or MAIN_MODEL, "batch-piece")
    if "judge" in stages:
        log("== judging every take without a verdict (%d clips per request)" % JUDGE_BATCH)
        left = judge_pending(R)
        if not left and "retake" in stages:
            n = R.retakes()
            log("== %d retakes (the words did not match)" % n)
            if n:
                judge_pending(R)
    # what is still missing: this run's misses + earlier runs' that no take has filled since
    have = {(r["group"], r["line"], r["voice"], r["model"]) for r in R.takes.values()}
    old = [m for m in (R.state.get("missing") or []) if (m["group"], m["line"], m["voice"], m["model"]) not in have]
    new = [m for m in R.missing if (m["group"], m["line"], m["voice"], m["model"]) not in have]
    seen, merged = set(), []
    for m in new + old:
        k = (m["group"], m["line"], m["voice"], m["model"])
        if k not in seen:
            seen.add(k)
            merged.append(m)
    R.missing = merged
    R.state["missing"] = R.missing
    R.state["unjudged"] = sum(1 for r in R.takes.values() if r.get("judge") is None)
    R.state["phones_error"] = PHONES["err"]
    R.state["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    R.save()
    write_page(R)
    log("calls this run: %d; page: %s" % (STATS["calls"], os.path.join(OUT, "index.html")))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE CURRENT BANK — joined exactly as VoiceBank joins it
# ════════════════════════════════════════════════════════════════════════════════════════════════
def current_clip(line):
    if line["lang"] != "ar":
        return None, []
    clips, missing = [], []
    for s in line["segments"]:
        p = os.path.join(BANK_DIR, mv.fnv64(s) + ".mp3")
        if os.path.exists(p):
            clips.append(open(p, "rb").read())
        else:
            missing.append(s)
    if missing:
        return None, missing
    rel = "clips/current_%s.mp3" % line["id"]
    out = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(mp3mod.concat(clips, gaps=[mv.pause_after(s) for s in line["segments"]]))
    return rel, []


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE PAGE
# ════════════════════════════════════════════════════════════════════════════════════════════════
def still_missing(R, lim):
    """Every planned take not made yet, with why (the plan: the pre-screens, every pick on every line, the model
    comparison, the respelling and batching tests on the main model)."""
    pa, pe, ar, en = probe_ar(), probe_en(), arabic_lines(), english_lines()
    ar_pick = R.state.get("ar_pick") or []
    en_pick = R.state.get("en_pick") or []
    best = ar_pick[0] if ar_pick else INCUMBENT_AR
    en_model = next((r["model"] for r in R.takes.values() if r["group"] == "prescreen" and r["lang"] == "en"),
                    MAIN_MODEL)
    plan = [("prescreen", pa, v, MAIN_MODEL) for v in AR_SCREEN]
    plan += [("prescreen", pe, v, en_model) for v in EN_SCREEN]
    plan += [("audition", ln, v, MAIN_MODEL) for v in ar_pick for ln in [pa] + ar]
    plan += [("audition", ln, v, MAIN_MODEL) for v in en_pick for ln in [pe] + en]
    plan += [("compare", ln, best, m) for m in COMPARE_MODELS for ln in [pa] + ar]
    plan += [("raw-spelling", pa, best, MAIN_MODEL)]
    plan += [("batch-piece", ln, best, MAIN_MODEL) for ln in batch_lines()]
    have = {(r["group"], r["line"], r["voice"], r["model"]) for r in R.takes.values()}
    out = []
    for g, ln, v, m in plan:
        if (g, ln["id"], v, m) in have:
            continue
        bu = ((lim.get("models") or {}).get(m) or {}).get("blocked_until") or 0
        why = ("%s: free-tier daily quota spent — resumes after %s UTC (midnight Pacific)"
               % (m, time.strftime("%Y-%m-%d %H:%M", time.gmtime(bu)))) if bu > time.time() else "not made yet"
        out.append({"group": g, "line": ln["id"], "voice": v, "model": m, "why": why})
    return out


def write_page(R):
    lines = all_lines()
    for ln in lines:
        ln["current"], ln["current_missing"] = current_clip(ln)
        ln["tts_text"] = say_text(ln)
    refs = []
    for name, p in EN_REFS:
        rel = "clips/ref_%s.mp3" % re.sub(r"[^A-Za-z0-9]+", "_", name)[:30]
        shutil.copyfile(os.path.join(ROOT, p), os.path.join(OUT, rel))
        refs.append({"name": name, "clip": rel, "text": EN_REF_TEXT, "source": p})
    lim = json.load(io.open(os.path.join(OUT, "limits.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(OUT, "limits.json")) else {"models": {}}
    R.state["missing"] = still_missing(R, lim)
    R.state["unjudged"] = sum(1 for r in R.takes.values() if r.get("judge") is None)
    takes = list(R.takes.values())
    for t in takes:
        if is_azure(t) and t["lang"] == "en":
            t["score"] = score_en_az(t)
        else:
            t["score"] = score_ar(t) if t["lang"] == "ar" else score_en(t, R.state.get("ref_f0"))
    az_led = {}
    lp = os.path.join(OUT, "azure", "ledger.json")
    if os.path.exists(lp):
        d = json.load(io.open(lp, encoding="utf-8"))
        az_led = {k: d.get(k) for k in ("chars", "requests", "failed", "by_day", "by_purpose")}
    data = {"generated": datetime.datetime.now().isoformat(timespec="minutes"), "main_model": MAIN_MODEL,
            "judge_model": JUDGE_MODEL, "lines": lines, "takes": takes, "refs": refs, "state": R.state,
            "limits": lim, "ar_screen": AR_SCREEN, "en_screen": EN_SCREEN,
            "prompt_ar_example": prompt_ar(say_text(lines[1])), "prompt_en_example": prompt_en(EN_REF_TEXT),
            "azure": {"ledger": az_led, "ar_final": AZ_AR_FINAL, "ar_tune": [az_model(c) for c in AZ_AR_TUNE],
                      "en_pick": [[v, az_model(c)] for v, c in AZ_EN_PICK], "say": AZ_SAY,
                      "tune_lines": AZ_TUNE_LINES, "monthly": 500000}}
    try:
        data["sekka"] = sekka_summary(R)
    except Exception as e:                    # the page never fails on the Sekka tab
        data["sekka"] = {"error": scrub("%s: %s" % (type(e).__name__, e))[:300]}
    for t in takes:
        if t["group"].startswith("sekka-"):
            t["score"] = score_sekka_en(t) if t["lang"] == "en" else score_sekka_ar(t)
    blob = scrub(json.dumps(data, ensure_ascii=False)).replace("</", "<\\/")
    page = PAGE.replace("/*DATA*/null", blob)
    io.open(os.path.join(OUT, "index.html"), "w", encoding="utf-8", newline="\n").write(page)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Audition</title>
<style>
:root{--bg:#f6f5f2;--panel:#ffffff;--ink:#1c1d21;--muted:#5d6170;--line:#e2e0da;--accent:#2457c5;--accent-ink:#ffffff;
--pass:#1f7a4a;--pass-bg:#e3f3ea;--fail:#b3261e;--fail-bg:#fbe6e4;--warn:#8a5a00;--warn-bg:#fdf1d6;--chip:#eeece6;
--cur:#5b3fa8;--cur-bg:#efeafb;--az:#0a6e8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121317;--panel:#1b1d23;--ink:#e9e8e4;--muted:#a3a6b2;
--line:#2c2f37;--accent:#7aa2ff;--accent-ink:#0d1020;--pass:#6fd39d;--pass-bg:#173326;--fail:#ff8a80;--fail-bg:#3a1c1a;
--warn:#f0c46b;--warn-bg:#3a2e12;--chip:#262932;--cur:#b9a3ff;--cur-bg:#2a2340;--az:#5fc8e6}}
:root[data-theme="dark"]{--bg:#121317;--panel:#1b1d23;--ink:#e9e8e4;--muted:#a3a6b2;--line:#2c2f37;--accent:#7aa2ff;
--accent-ink:#0d1020;--pass:#6fd39d;--pass-bg:#173326;--fail:#ff8a80;--fail-bg:#3a1c1a;--warn:#f0c46b;--warn-bg:#3a2e12;
--chip:#262932;--cur:#b9a3ff;--cur-bg:#2a2340;--az:#5fc8e6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 16px 8px;max-width:1280px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:22px 0 8px}
h3{font-size:15px;margin:0 0 4px}
.sub{color:var(--muted)}
main{max-width:1280px;margin:0 auto;padding:0 16px 48px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}
nav button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:7px 14px;border-radius:999px;
cursor:pointer;font:inherit}
nav button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.take{border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--panel)}
.take.cur{background:var(--cur-bg);border-color:var(--cur)}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chip{background:var(--chip);border-radius:6px;padding:1px 7px;font-size:12px;white-space:nowrap}
.pass{background:var(--pass-bg);color:var(--pass);font-weight:600}
.fail{background:var(--fail-bg);color:var(--fail);font-weight:600}
.warn{background:var(--warn-bg);color:var(--warn)}
audio{width:100%;height:34px;margin:6px 0}
.ar{direction:rtl;text-align:right;font-family:"Noto Naskh Arabic","Segoe UI","Tahoma",sans-serif;font-size:17px;line-height:1.7}
.say{direction:rtl;text-align:right;color:var(--muted);font-size:14px}
.small{font-size:12px;color:var(--muted)}
.heard{border-top:1px dashed var(--line);margin-top:6px;padding-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto}
pre{white-space:pre-wrap;background:var(--chip);padding:10px;border-radius:8px;font-size:12px;overflow-x:auto}
select{font:inherit;padding:5px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.toggle{float:right}
h4.sect{margin:14px 0 6px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);border-top:1px solid var(--line);padding-top:10px}
h4.sect.gem{color:var(--accent)}
h4.sect.azs{color:var(--az)}
.az{background:var(--az);color:var(--panel);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<header>
<button class="toggle chip" id="theme" type="button">theme</button>
<h1>RouteMind voice audition</h1>
<div class="sub" id="sub"></div>
<nav id="tabs"></nav>
</header>
<main id="main"></main>
<script>
const D = /*DATA*/null;
const $ = (s, el=document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const LINES = Object.fromEntries(D.lines.map(l => [l.id, l]));
const short = m => m.replace("gemini-","").replace("-preview","·pv");
const isAz = t => String(t.model||"").startsWith("azure");
const AZG = {"azure":1,"azure-tune":1,"azure-screen":1,"azure-batch":1,"azure-batch-single":1};
const tabs = [["line","By line"],["sekka","Sekka (live)"],["azure","Azure"],["voice","By voice"],["screen","Pre-screen"],["batch","Batch test"],["limits","Limits & missing"],["about","How"]];
const isSk = t => String(t.group||"").startsWith("sekka-");
let cur = "line";
try { cur = localStorage.getItem("va_tab") || "line"; } catch(e) {}
function chips(t){
  const j=t.judge||{}, c=t.check||{}, out=[];
  out.push(t.judge==null?`<span class="chip warn">VERDICT PENDING</span>`:`<span class="chip ${t.pass?'pass':'fail'}">${t.pass?'PASS':'FAIL'}</span>`);
  if(isAz(t)) out.push(`<span class="chip az">AZURE</span><span class="chip">${esc(t.voice)}</span><span class="chip">${esc(t.model.replace(/^azure ?/,"")||"default")}</span>${t.meta&&t.meta.chars!=null?`<span class="chip">${t.meta.chars} chars · ${t.meta.latency_s}s</span>`:""}`);
  else out.push(`<span class="chip">${esc(t.voice)}</span><span class="chip">${esc(short(t.model))}</span>`);
  if(t.take>1) out.push(`<span class="chip warn">retake ${t.take}</span>`);
  if(t.raw_spelling) out.push(`<span class="chip warn">raw ق spelling</span>`);
  if(t.meta&&t.meta.batched) out.push(`<span class="chip">line ${t.meta.piece+1} of ${t.meta.batched} in one request</span>`);
  if(j._judge_model&&j._judge_model!==D.judge_model) out.push(`<span class="chip warn">judged by ${esc(short(j._judge_model))}</span>`);
  if(t.judge==null){ if(t.stats&&t.stats.f0_med) out.push(`<span class="chip">F0 ${t.stats.f0_med} Hz</span>`); return out.join(" "); }
  if(t.lang==="ar"){
    out.push(`<span class="chip ${t.dialect_ok?'pass':'fail'}">${esc(j.accent||'?')} ${esc(j.accent_score??'')}/10</span>`);
    out.push(`<span class="chip">ج ${esc(j.jim||'?')}</span><span class="chip">ق ${esc(j.qaf||'?')}</span>`);
    out.push(`<span class="chip ${j.addressee==='feminine'?'fail':''}">to: ${esc(j.addressee||'?')}</span>`);
    out.push(`<span class="chip ${c.text_ok?'pass':'fail'}">CER ${esc(c.cer)}</span>`);
  } else {
    out.push(`<span class="chip">${esc(j.accent||'?')}</span>`);
    out.push(`<span class="chip ${c.text_ok?'pass':'fail'}">WER ${esc(c.wer)}</span>`);
    if(j.sim_overall!=null) out.push(`<span class="chip">≈Journey-F ${esc(j.sim_overall)}/10</span>`);
  }
  out.push(`<span class="chip">warmth ${esc(j.warmth??'?')}</span><span class="chip">natural ${esc(j.naturalness??'?')}</span>`);
  if(j.pace) out.push(`<span class="chip">pace ${esc(j.pace)}</span>`);
  if(t.stats&&t.stats.f0_med) out.push(`<span class="chip">F0 ${t.stats.f0_med} Hz</span>`);
  return out.join(" ");
}
function takeCard(t, opts={}){
  const j=t.judge||{}, c=t.check||{}, L=LINES[t.line]||{};
  const probs=[...(c.problems||[]), ...((t.phones&&t.phones.hard)||[]).map(p=>"phones: "+p)];
  return `<div class="take">
    ${opts.title?`<h3>${esc(opts.title)}</h3>`:""}
    <div class="row">${chips(t)}</div>
    <audio controls preload="none" src="${esc(t.clip)}"></audio>
    ${t.lang==="ar"&&t.tts_text!==L.plain?`<div class="say" title="what the voice was given">${esc(t.tts_text)}</div>`:""}
    <div class="heard"><div class="small">judge heard:</div>
    <div class="${t.lang==='ar'?'ar':''}">${esc(j.transcript||'—')}</div>
    ${probs.length?`<div class="small" style="color:var(--fail)">${esc(probs.join(' · '))}</div>`:""}
    ${(j.problems&&j.problems.length)?`<div class="small">judge flags: ${esc([].concat(j.problems).join(' · '))}</div>`:""}
    <div class="small">${esc(j.notes||'')}</div>
    ${j.sim_notes?`<div class="small">vs Journey-F: ${esc(j.sim_notes)}</div>`:""}
    ${t.phones&&t.phones.ipa?`<details class="small"><summary>phone listener (IPA)</summary>${esc(t.phones.ipa)}</details>`:""}
    </div></div>`;
}
function currentCard(L){
  if(L.lang!=="ar") return "";
  if(L.current) return `<div class="take cur"><div class="row"><span class="chip" style="background:var(--cur);color:var(--panel)">CURRENT BANK</span><span class="chip">Zephyr · 2.5-pro-tts</span></div>
    <audio controls preload="none" src="${esc(L.current)}"></audio><div class="small">the bundled clips joined as VoiceBank joins them</div></div>`;
  return `<div class="take cur"><div class="row"><span class="chip" style="background:var(--cur);color:var(--panel)">CURRENT BANK</span></div>
    <p><b>no current clip</b> — the bank has no clip for: <span class="ar" style="font-size:14px">${esc(L.current_missing.join(' | '))}</span>
    (on the phone this whole line falls back to the cloud / device voice)</p></div>`;
}
function lineHead(L){
  return `<h3>${esc(L.label)} <span class="small">${esc(L.id)}</span></h3>
   <div class="${L.lang==='ar'?'ar':''}">${esc(L.plain)}</div>
   ${L.lang==='ar'&&L.tts_text!==L.plain?`<div class="small">the voice reads (pronounce.py SAY):</div><div class="say">${esc(L.tts_text)}</div>`:""}
   ${L.source?`<div class="small">source: ${esc(L.source)}</div>`:""}`;
}
function order(a,b){
  const g={"audition":0,"compare":1,"prescreen":2,"raw-spelling":3,"batch-single":4,"batch-piece":5,"azure":6,"azure-tune":7,"azure-screen":8,"azure-batch-single":9,"azure-batch":10};
  return ((g[a.group]??11)-(g[b.group]??11))||a.voice.localeCompare(b.voice)||a.model.localeCompare(b.model)||a.take-b.take;
}
function viewLine(){
  let h=`<div class="panel row"><label>language <select id="lf"><option value="all">all</option><option value="ar">Arabic</option><option value="en">English</option></select></label>
   <label><input type="checkbox" id="onlybest"> hide retakes / failed first takes</label></div>`;
  for(const L of D.lines){
    if(L.sekka) continue;   // Sekka's replies have their own tab
    const all=D.takes.filter(t=>t.line===L.id).sort(order);
    const gem=all.filter(t=>!isAz(t)), az=all.filter(t=>isAz(t)&&t.group==="azure");
    h+=`<section class="panel" data-lang="${L.lang}">${lineHead(L)}`;
    if(L.lang==="ar"||L.id==="en00_probe"){
      h+=`<h4 class="sect">${L.lang==="ar"?"Today's bundled clip":"Reference — the English voice the app used to have"}</h4><div class="grid">${currentCard(L)}`;
      if(L.id==="en00_probe") for(const r of D.refs) h+=`<div class="take cur"><div class="row"><span class="chip" style="background:var(--cur);color:var(--panel)">REFERENCE</span><span class="chip">${esc(r.name)}</span></div><audio controls preload="none" src="${esc(r.clip)}"></audio></div>`;
      h+=`</div>`;
    }
    h+=`<h4 class="sect gem">Gemini voices <span class="small">(${gem.length})</span></h4><div class="grid">`;
    for(const t of gem) h+=`<div data-take="${t.take}" data-pass="${t.pass}">${takeCard(t)}</div>`;
    if(!gem.length) h+=`<div class="take"><b>no Gemini take yet</b><div class="small">see “Limits & missing”</div></div>`;
    h+=`</div><h4 class="sect azs">Azure voices <span class="small">(${az.length})</span></h4><div class="grid">`;
    for(const t of az) h+=`<div data-take="${t.take}" data-pass="${t.pass}">${takeCard(t)}</div>`;
    if(!az.length) h+=`<div class="take"><b>no Azure take for this line</b>${L.id.startsWith("bt")?'<div class="small">bank segments: see the Azure tab (batching)</div>':""}</div>`;
    h+=`</div></section>`;
  }
  return h;
}
function viewVoice(){
  const groups={};
  for(const t of D.takes){ if(t.group==="prescreen"||isSk(t)||(isAz(t)&&t.group!=="azure")) continue; const k=`${t.lang}|${t.voice}|${t.model}|${t.group}`; (groups[k]=groups[k]||[]).push(t); }
  const keys=Object.keys(groups).sort();
  const avg=(xs)=>{const v=xs.filter(x=>typeof x==="number"); return v.length?(v.reduce((a,b)=>a+b,0)/v.length).toFixed(1):"–";};
  let h=`<section class="panel"><h2 style="margin-top:0">Summary — every voice × model (first takes; the pre-screen is on its own tab)</h2><div class="scroll"><table><tr><th>lang</th><th>voice</th><th>model</th><th>group</th><th>judged</th><th>pass</th><th>accent /10</th><th>warmth</th><th>natural</th><th>F0 Hz</th><th>made</th></tr>`;
  for(const k of keys){ const [lang,v,m,g]=k.split("|"); const f=groups[k].filter(t=>t.take===1); const j=f.filter(t=>t.judge!=null);
    h+=`<tr><td>${lang}</td><td>${esc(v)}</td><td>${esc(short(m))}</td><td>${esc(g)}</td><td>${j.length}/${f.length}</td><td>${j.filter(t=>t.pass).length}</td><td>${avg(j.map(t=>Number(t.judge.accent_score)))}</td><td>${avg(j.map(t=>Number(t.judge.warmth)))}</td><td>${avg(j.map(t=>Number(t.judge.naturalness)))}</td><td>${avg(f.map(t=>t.stats&&t.stats.f0_med))}</td><td>${f.some(t=>t.meta&&t.meta.batched)?"N per request":"one per request"}</td></tr>`; }
  h+=`</table></div></section>`;
  for(const k of keys){
    const [lang,v,m,g]=k.split("|"); const ts=groups[k];
    const best={}; for(const t of ts){ const b=best[t.line]; if(!b||(t.pass&&!b.pass)||(t.pass===b.pass&&t.take<b.take)) best[t.line]=t; }
    const firsts=ts.filter(t=>t.take===1), passed=firsts.filter(t=>t.pass).length;
    h+=`<section class="panel"><h2 style="margin-top:0">${esc(v)} · ${esc(m)} · ${esc(g)} <span class="small">${lang==="ar"?"Arabic":"English"} — ${passed}/${firsts.length} first takes pass</span></h2><div class="grid">`;
    for(const L of D.lines.filter(l=>best[l.id])){
      h+=`<div><div class="small">${esc(L.label)}</div><div class="${L.lang==='ar'?'ar':''}" style="font-size:15px">${esc(L.plain)}</div>${takeCard(best[L.id])}
      ${L.current?`<audio controls preload="none" src="${esc(L.current)}" title="current bank"></audio><div class="small">↑ current bank clip</div>`:(L.lang==='ar'?`<div class="small">no current clip</div>`:"")}</div>`;
    }
    h+=`</div></section>`;
  }
  return h||`<div class="panel">no auditions yet</div>`;
}
function viewScreen(){
  const st=D.state||{};
  let h=`<section class="panel"><h2 style="margin-top:0">Arabic pre-screen — ${D.ar_screen.length} female voices, one probe line, ${esc(D.main_model)}</h2>
   <div class="ar">${esc((LINES["ar00_probe"]||{}).plain)}</div>
   <p class="small">Score = accent score + ½ warmth + ½ naturalness − 4 if ج is not g − 4 if ق is not a glottal stop − 3 if the words do not match − 1.5 per phone-listener hard flag. Picks: <b>${esc((st.ar_pick||[]).join(", "))}</b> (the incumbent Zephyr is always auditioned).</p>
   <div class="grid">`;
  const ar=D.takes.filter(t=>t.group==="prescreen"&&t.line==="ar00_probe").sort((a,b)=>b.score-a.score);
  for(const t of ar) h+=takeCard(t,{title:`${t.voice} — score ${t.score}`});
  h+=`${currentCard(LINES["ar00_probe"])}</div></section>`;
  h+=`<section class="panel"><h2 style="margin-top:0">English pre-screen — nearest to en-US-Journey-F</h2>
   <p class="small">Score = judge similarity to the Journey-F clip (overall, 1-10) + ½ warmth + ½ naturalness − 3 if the words do not match − a pitch penalty (12 points per octave of median-F0 distance from the reference's ${esc(st.ref_f0)} Hz, capped at 4). Picks: <b>${esc((st.en_pick||[]).join(", "))}</b></p><div class="grid">`;
  for(const r of D.refs) h+=`<div class="take cur"><div class="row"><span class="chip" style="background:var(--cur);color:var(--panel)">REFERENCE</span><span class="chip">${esc(r.name)}</span></div><audio controls preload="none" src="${esc(r.clip)}"></audio><div class="small">${esc(r.text)}</div></div>`;
  const en=D.takes.filter(t=>t.group==="prescreen"&&t.line==="en00_probe").sort((a,b)=>b.score-a.score);
  for(const t of en) h+=takeCard(t,{title:`${t.voice} — score ${t.score}`});
  h+=`</div></section>`;
  const raw=D.takes.filter(t=>t.group==="raw-spelling");
  if(raw.length){ h+=`<section class="panel"><h2 style="margin-top:0">Does the new model still need the ق respelling?</h2><p class="small">The same probe with ق written as ق (no pronounce.py SAY spelling).</p><div class="grid">`; for(const t of raw) h+=takeCard(t); h+=`</div></section>`; }
  return h;
}
function viewBatch(){
  const B=Object.values((D.state||{}).batches||{});
  let h=`<section class="panel"><h2 style="margin-top:0">One request, many lines</h2>
  <p class="small">The free tier gives a TTS model 10 requests a day, so every audition take below the pre-screen was made N lines per request and cut at the pauses (the pre-screen takes are one request each — compare a voice's probe in both). The bank segments at the bottom were made the same way and are judged as build_bank judges a direction-word clip: inside a line, behind the bank's own lead «بعد تلتمية متر،», next to the CURRENT single-request bank clip where the bank has one.</p>
  <div class="scroll"><table><tr><th>group</th><th>voice</th><th>model</th><th>take</th><th>lines</th><th>audio</th><th>cut</th><th>pieces that pass</th><th>the whole request</th></tr>`;
  for(const b of B){
    const ts=D.takes.filter(t=>t.meta&&t.meta.batch_clip===b.clip);
    const judged=ts.filter(t=>t.judge!=null), ok=judged.filter(t=>t.pass).length;
    h+=`<tr><td>${esc(b.group)}</td><td>${esc(b.voice)}</td><td>${esc(short(b.model))}</td><td>${b.take}</td><td>${b.n}</td><td>${esc(b.audio_s)} s</td><td>${b.cut_ok?'ok':'<b>failed</b>'}</td><td>${judged.length?ok+'/'+judged.length:'pending'}</td><td><audio controls preload="none" src="${esc(b.clip)}"></audio></td></tr>`;
  }
  h+=`</table></div></section>`;
  for(const L of D.lines.filter(l=>l.id.startsWith("bt"))){
    const ts=D.takes.filter(t=>t.line===L.id).sort((a,b)=>a.take-b.take);
    h+=`<section class="panel">${lineHead(L)}<div class="grid" style="margin-top:8px">${currentCard(L)}`;
    for(const t of ts) h+=takeCard(t);
    if(!ts.length) h+=`<div class="take"><b>no take yet</b></div>`;
    h+=`</div></section>`;
  }
  return h;
}
function viewLimits(){
  const L=D.limits.models||{}; let h=`<section class="panel"><h2 style="margin-top:0">What the free tier answered</h2><div class="scroll"><table><tr><th>model</th><th>ok</th><th>ok by Pacific day</th><th>429</th><th>5xx</th><th>avg latency</th><th>audio s</th><th>tokens in / out</th><th>quota answers (latest)</th></tr>`;
  for(const [m,x] of Object.entries(L)){
    const v=(x.violations||[]).slice(-3).map(r=>`${r.at}: ${(r.violations||[]).map(q=>`${q.quotaId}=${q.quotaValue}`).join(", ")||esc(r.message)} (after ${r.ok_today_before} ok today${r.retryDelay_s!=null?`, retry ${r.retryDelay_s}s`:""})`).join("<br>");
    h+=`<tr><td>${esc(m)}</td><td>${x.ok}</td><td>${esc(JSON.stringify(x.ok_by_ptday))}</td><td>${x.e429}</td><td>${x.e5xx}</td><td>${x.lat_n?(x.lat_sum/x.lat_n).toFixed(1)+" s":""}</td><td>${(x.audio_s||0).toFixed(0)}</td><td>${x.tokens_in} / ${x.tokens_out}</td><td class="small">${v||"—"}</td></tr>`;
  }
  h+=`</table></div></section>`;
  const miss=(D.state&&D.state.missing)||[];
  h+=`<section class="panel"><h2 style="margin-top:0">Missing from this run (${miss.length})</h2>`;
  h+=miss.length?`<div class="scroll"><table><tr><th>line</th><th>voice</th><th>model</th><th>group</th><th>why</th></tr>${miss.map(m=>`<tr><td>${esc(m.line)}</td><td>${esc(m.voice)}</td><td>${esc(m.model)}</td><td>${esc(m.group)}</td><td class="small">${esc(m.why)}</td></tr>`).join("")}</table></div>`:"<p>nothing — every planned take was made.</p>";
  if(D.state&&D.state.phones_error) h+=`<p class="small">phone listener: ${esc(D.state.phones_error)}</p>`;
  return h+`</section>`;
}
function viewAzure(){
  const A=D.azure||{}, L=A.ledger||{}, med=xs=>{const v=xs.filter(x=>typeof x==="number"&&!isNaN(x)).sort((a,b)=>a-b); return v.length?v[Math.floor(v.length/2)]:null;};
  const avg=xs=>{const v=xs.filter(x=>typeof x==="number"&&!isNaN(x)); return v.length?(v.reduce((a,b)=>a+b,0)/v.length).toFixed(1):"–";};
  let h=`<section class="panel"><h2 style="margin-top:0">Azure Speech — the free F0 resource</h2>
   <p>Characters spent by this desk (every audition, tuning and Sekka test, counted before each request): <b>${(L.chars||0).toLocaleString()}</b> of the month's ${(A.monthly||500000).toLocaleString()} (${((L.chars||0)/(A.monthly||500000)*100).toFixed(1)} %), ${L.requests||0} requests${L.failed?`, ${L.failed} failed`:""}.</p>
   <div class="small">by purpose: ${esc(Object.entries(L.by_purpose||{}).map(([k,v])=>k+" "+v).join(" · "))}</div>
   <p class="small">Final Arabic settings: ${esc(Object.entries(A.ar_final||{}).map(([v,c])=>v+" "+JSON.stringify(c)).join(" | "))}. English voices on every line: ${esc((A.en_pick||[]).map(p=>p[0]+(p[1]!=="azure"?" ("+p[1].replace(/^azure /,"")+")":"")).join(", ")||"not chosen yet")}.</p></section>`;
  // Arabic tuning
  const tune=D.takes.filter(t=>t.group==="azure-tune");
  const keys=[...new Set(tune.map(t=>t.voice+"|"+t.model))];
  h+=`<section class="panel"><h2 style="margin-top:0">Arabic settings tried (${tune.length} takes)</h2><p class="small">Each setting on the same ${esc((A.tune_lines||[]).join(", "))}. Judge = the blind Gemini listener (words, accent, ج / ق, warmth, naturalness, pace); phones = the blind phone listener (hard flags: ق heard as q, ج as dʒ, garbled consonants); rate = letters per second of speech (the Gemini bank speaks 12.6); F0 IQR = how much the pitch moves (flat newsreader → small).</p><div class="scroll"><table><tr><th>voice</th><th>setting</th><th>pass</th><th>accent</th><th>ق</th><th>ج</th><th>warmth</th><th>natural</th><th>pace</th><th>phone flags</th><th>rate l/s</th><th>F0 med</th><th>F0 IQR</th><th>chars</th></tr>`;
  for(const k of keys){ const [v,m]=k.split("|"); const ts=tune.filter(t=>t.voice===v&&t.model===m); const j=ts.filter(t=>t.judge);
    const flags=ts.reduce((a,t)=>a+(((t.phones||{}).hard)||[]).length,0);
    h+=`<tr><td>${esc(v.replace("Neural",""))}</td><td>${esc(m.replace(/^azure ?/,"")||"default")}</td><td class="num">${j.filter(t=>t.pass).length}/${j.length}</td><td class="num">${avg(j.map(t=>Number(t.judge.accent_score)))}</td><td>${esc([...new Set(j.map(t=>t.judge.qaf))].join("/"))}</td><td>${esc([...new Set(j.map(t=>t.judge.jim))].join("/"))}</td><td class="num">${avg(j.map(t=>Number(t.judge.warmth)))}</td><td class="num">${avg(j.map(t=>Number(t.judge.naturalness)))}</td><td>${esc([...new Set(j.map(t=>t.judge.pace))].join("/"))}</td><td class="num">${flags}</td><td class="num">${med(ts.map(t=>(t.stats||{}).rate_lps))??"–"}</td><td class="num">${med(ts.map(t=>(t.stats||{}).f0_med))??"–"}</td><td class="num">${med(ts.map(t=>(t.stats||{}).f0_iqr))??"–"}</td><td class="num">${ts.reduce((a,t)=>a+((t.meta||{}).chars||0),0)}</td></tr>`; }
  h+=`</table></div>`;
  for(const lid of (A.tune_lines||[])){ const Ln=LINES[lid]; if(!Ln) continue;
    h+=`<h3 style="margin-top:16px">${esc(Ln.label)}</h3><div class="ar" style="font-size:15px">${esc(Ln.plain)}</div><div class="grid" style="margin-top:6px">${currentCard(Ln)}`;
    for(const t of tune.filter(t=>t.line===lid).sort(order)) h+=takeCard(t);
    h+=`</div>`; }
  h+=`</section>`;
  // English screen
  const scr=D.takes.filter(t=>t.group==="azure-screen").sort((a,b)=>(b.score??-99)-(a.score??-99));
  h+=`<section class="panel"><h2 style="margin-top:0">English voices tried (${scr.length}) — ranked</h2><p class="small">Score = naturalness + ¾ warmth + ¾ clarity + ¼ nearness to the old Journey-F voice − 1.5 if the pace is not natural − 3 if the words do not match − ½ per judge flag. RouteMind wants a voice of its own, so the nearness to Journey-F counts little; the pitch is shown, not penalised.</p><div class="scroll"><table><tr><th>#</th><th>voice</th><th>setting</th><th>score</th><th>natural</th><th>warmth</th><th>clarity</th><th>≈Journey-F</th><th>pace</th><th>F0 Hz</th><th>rate l/s</th><th>words</th><th>listen</th></tr>`;
  scr.forEach((t,i)=>{ const j=t.judge||{}; h+=`<tr><td class="num">${i+1}</td><td>${esc(t.voice)}</td><td>${esc(t.model.replace(/^azure ?/,"")||"default")}</td><td class="num"><b>${t.score??"pending"}</b></td><td class="num">${esc(j.naturalness??"")}</td><td class="num">${esc(j.warmth??"")}</td><td class="num">${esc(j.clarity??"")}</td><td class="num">${esc(j.sim_overall??"")}</td><td>${esc(j.pace??"")}</td><td class="num">${esc((t.stats||{}).f0_med??"")}</td><td class="num">${esc((t.stats||{}).rate_lps??"")}</td><td>${t.check?(t.check.text_ok?"ok":"<b>"+esc(t.check.problems.join(" "))+"</b>"):""}</td><td style="min-width:220px"><audio controls preload="none" src="${esc(t.clip)}"></audio></td></tr>`; });
  h+=`</table></div><div class="grid" style="margin-top:8px">`;
  for(const r of D.refs) h+=`<div class="take cur"><div class="row"><span class="chip" style="background:var(--cur);color:var(--panel)">REFERENCE</span><span class="chip">${esc(r.name)}</span></div><audio controls preload="none" src="${esc(r.clip)}"></audio><div class="small">${esc(r.text)}</div></div>`;
  h+=`</div></section>`;
  // batching
  const B=Object.values((D.state||{}).batches||{}).filter(b=>b.group==="azure-batch");
  h+=`<section class="panel"><h2 style="margin-top:0">Azure and the bank: one segment per request, or many?</h2><p class="small">The 12 bank segments of the batch test in ONE request (a &lt;break&gt; between them — markup Azure bills), cut by the aligned cut, against one request per segment; both judged inside a line behind the bank's lead clip, as build_bank judges a direction word.</p>`;
  for(const b of B){ const ts=D.takes.filter(t=>t.meta&&t.meta.batch_clip===b.clip); const jd=ts.filter(t=>t.judge); const sg=D.takes.filter(t=>t.group==="azure-batch-single"&&t.voice===b.voice); const sj=sg.filter(t=>t.judge);
    h+=`<p>${esc(b.voice)}: ONE request = <b>${b.chars}</b> billable characters for ${b.n} segments (one per request: <b>${b.chars_single}</b>); cut ${b.cut_ok?"ok":"FAILED"}; pieces passing ${jd.filter(t=>t.pass).length}/${jd.length}; single takes passing ${sj.filter(t=>t.pass).length}/${sj.length}. <audio controls preload="none" src="${esc(b.clip)}"></audio></p>`; }
  for(const L2 of D.lines.filter(l=>l.id.startsWith("bt"))){ const ts=D.takes.filter(t=>t.line===L2.id&&(t.group==="azure-batch"||t.group==="azure-batch-single")); if(!ts.length) continue;
    h+=`<h3 style="margin-top:14px">«${esc(L2.seg)}»</h3><div class="grid">${currentCard(L2)}${ts.map(t=>takeCard(t,{title:t.group==="azure-batch"?"one request, cut":"one request each"})).join("")}</div>`; }
  return h+`</section>`;
}
function viewSekka(){
  const S=D.sekka||{}; if(S.error) return `<div class="panel">Sekka summary failed: ${esc(S.error)}</div>`;
  const live=S.live||{}, pk=S.pick;
  const cfgTxt=c=>{const o=c||{}; return [o.style?`style ${o.style}`:"", o.degree?`degree ${o.degree}`:""].filter(Boolean).join(", ")||"no style";};
  let h=`<section class="panel"><h2 style="margin-top:0">Sekka's live voice</h2>
   <p><b>English:</b> ${pk?`${esc(pk.voice)} (${esc(cfgTxt(pk.cfg))}) — the compiled default (tts.ts) and the live row`:"not picked yet"}. <b>Arabic:</b> the phone's own device voice — the per-language switch <code>voice_ar_enabled</code> is OFF (compiled default and live row); nothing in the Arabic section below is live.</p>
   ${live.row_at?`<p class="small">live row merged ${esc(live.row_at)} UTC</p>`:""}
   <p class="small">Azure characters spent by this work: ${esc((S.ledger||{}).chars??0)} in ${esc((S.ledger||{}).requests??0)} requests (side ledger azure/ledger_sekka.json; merged into the desk ledger: ${esc((S.ledger||{}).merged_chars??0)}).</p></section>`;
  const scr=(S.en_screen||[]).slice().sort((a,b)=>(b.score??-99)-(a.score??-99));
  h+=`<section class="panel"><h2 style="margin-top:0">English — the voices tried on the traffic answer</h2><p class="small">Score = naturalness + ¾ warmth + ½ clarity + (2 − |energy − 7|) − 1.5 if the pace is not natural − 3 if the words do not match − ½ per judge flag. Energy (the blind judge, 1-10): 1 sleepy … 6-8 warm and upbeat … 10 over the top. Rate = letters per second of speech; F0 IQR = how much the pitch moves (Hz). Latency = Azure's whole answer from the desk.</p><div class="scroll"><table><tr><th>#</th><th>voice</th><th>style</th><th>score</th><th>energy</th><th>warmth</th><th>natural</th><th>clarity</th><th>pace</th><th>rate l/s</th><th>F0 IQR</th><th>latency</th><th>chars</th><th>listen</th></tr>`;
  scr.forEach((r,i)=>{ h+=`<tr><td class="num">${i+1}</td><td>${esc(r.voice)}</td><td>${esc(cfgTxt(r.cfg))}</td><td class="num"><b>${r.score??"pending"}</b></td><td class="num">${esc(r.energy??"")}</td><td class="num">${esc(r.warmth??"")}</td><td class="num">${esc(r.naturalness??"")}</td><td class="num">${esc(r.clarity??"")}</td><td>${esc(r.pace??"")}</td><td class="num">${esc(r.rate_lps??"")}</td><td class="num">${esc(r.f0_iqr??"")}</td><td class="num">${r.latency_s!=null?esc(r.latency_s)+" s":""}</td><td class="num">${esc(r.chars??"")}</td><td style="min-width:220px">${r.clip?`<audio controls preload="none" src="${esc(r.clip)}"></audio>`:"not made"}</td></tr>`; });
  h+=`</table></div>`;
  if((S.degree||[]).length){ h+=`<h3 style="margin-top:14px">Does the voice honour styledegree?</h3><p class="small">The same short line at the voice's own degree and at 0.5 / 1.5 — identical audio (the same hash) = the degree is ignored by that voice.</p><div class="scroll"><table><tr><th>voice</th><th>setting</th><th>audio hash</th><th>F0 IQR</th><th>rate l/s</th><th>listen</th></tr>`;
    for(const d of S.degree) h+=`<tr><td>${esc(d.voice)}</td><td>${esc(d.model.replace(/^azure ?/,""))}</td><td><code>${esc(d.sha||"")}</code></td><td class="num">${esc((d.stats||{}).f0_iqr??"")}</td><td class="num">${esc((d.stats||{}).rate_lps??"")}</td><td style="min-width:220px"><audio controls preload="none" src="${esc(d.clip)}"></audio></td></tr>`;
    h+=`</table></div>`; }
  h+=`</section>`;
  if((S.finals||[]).length){
    h+=`<section class="panel"><h2 style="margin-top:0">English — the finalists on Sekka's six replies (the owner may veto)</h2><div class="scroll"><table><tr><th>voice</th><th>style</th><th>mean score</th><th>energy</th><th>warmth</th><th>words right</th><th>rate l/s</th><th>F0 IQR</th><th>latency p50</th><th>chars (6 replies)</th><th></th></tr>`;
    for(const f of S.finals) h+=`<tr><td>${esc(f.voice)}</td><td>${esc(cfgTxt(f.cfg))}</td><td class="num"><b>${f.score??"pending"}</b></td><td class="num">${esc(f.energy??"")}</td><td class="num">${esc(f.warmth??"")}</td><td class="num">${f.read_back}/${f.n}</td><td class="num">${esc(f.rate_lps??"")}</td><td class="num">${esc(f.f0_iqr??"")}</td><td class="num">${f.latency_p50!=null?esc(f.latency_p50)+" s":""}</td><td class="num">${f.chars}</td><td>${f.pick?'<span class="chip pass">PICK — live</span>':""}</td></tr>`;
    h+=`</table></div>`;
    if((S.rank||[]).length){
      h+=`<h3 style="margin-top:14px">The comparative rankings (how the pick was made)</h3><p class="small">The single-clip verdicts came back flat (every voice energy 6 / warmth 7 / natural 8), so each reply was heard by ALL the finalists in ONE request and ranked against the brief ("warm and upbeat, not over the top, not sleepy") — twice, the second time in the reverse clip order (a judge's position bias cancels in the sum). Borda = 3 points for a first place among the three, 2 for a second, 1 for a third. The judges left today were the flash-lite family: treat this as a tie-breaker the page lets you overrule by ear.</p><div class="scroll"><table><tr><th>voice</th><th>style</th><th>Borda</th><th>first places</th><th>energy</th><th>warmth</th><th>natural</th><th>heard as</th></tr>`;
      for(const r of S.rank.slice().sort((a,b)=>b.borda-a.borda)) h+=`<tr><td>${esc(r.voice)}</td><td>${esc(cfgTxt(r.cfg))}</td><td class="num"><b>${r.borda}</b></td><td class="num">${r.first}</td><td class="num">${esc(r.energy??"")}</td><td class="num">${esc(r.warmth??"")}</td><td class="num">${esc(r.natural??"")}</td><td>${esc(Object.entries(r.feel||{}).map(([k,v])=>k+" "+v).join(", "))}</td></tr>`;
      h+=`</table></div><div class="small">${esc((S.rank_per_reply||[]).map(p=>p.reply+": "+p.order.join(" > ")).join(" | "))}</div>`;
    }
    for(const R of (S.replies||[])){
      h+=`<h3 style="margin-top:14px">${esc(R.label)}</h3><div>${esc(R.en)}</div><div class="grid" style="margin-top:6px">`;
      for(const f of S.finals){ const t=D.takes.find(t=>(t.group==="sekka-en"||t.group==="sekka-en-screen")&&t.line==="sk_en_"+R.key&&t.voice===f.voice&&t.model===f.model); h+=t?takeCard(t,{title:`${f.voice} — ${cfgTxt(f.cfg)}${f.pick?" (PICK)":""}`}):`<div class="take"><b>${esc(f.voice)}</b> — not made</div>`; }
      h+=`</div>`;
    }
    h+=`</section>`;
  }
  h+=`<section class="panel"><h2 style="margin-top:0">Arabic — Azure ar-EG with diacritization vs Gemini Despina (NOT live: the owner decides)</h2>
   <p class="small">Pronunciation = the blind judge (read-back, accent, ج / ق, Modern-Standard case endings) + the blind phone listener (tools/masri_voice/phones.py: ق heard as q, ج as dʒ, ث/ذ as θ/ð, garbled consonants — the authority on sounds). Time to first audio = the TTS answer for one reply (p50; neither API streams partial audio on this path). A conversation = 3 voice turns × one reply of the six's average length (docs/copilot-models-2026-10.md §6).</p>
   <div class="scroll"><table><tr><th>variant</th><th>text sent</th><th>takes</th><th>words right</th><th>Cairene (judge)</th><th>accent /10</th><th>ج</th><th>ق</th><th>case endings</th><th>phone-listener flags</th><th>time to first audio</th><th>cost per conversation</th></tr>`;
  for(const a of (S.ar||[])){
    const tt=a.ttfa||{}, c=a.cost||{};
    const ttTxt = tt.azure_p50_s!=null ? `${tt.azure_p50_s} s (Azure)${tt.diacritizer_s&&tt.diacritizer_s!==0?" + the diacritizer ("+esc(tt.diacritizer_s)+")":""}` : (tt.gemini_p50_s!=null?`${tt.gemini_p50_s} s (n=${tt.n})`:"pending");
    const cTxt = c.azure_chars_per_conversation!=null ? `$0 on Azure F0 (${c.azure_chars_per_conversation} of the month's 500,000 characters)${c.diac_usd?` + the diacritizer ≈ $${esc(c.diac_usd["gpt-6-luna"])} (gpt-6-luna) / $${esc(c.diac_usd["gpt-5.4-nano"])} (gpt-5.4-nano), estimated`:""}` : (c.usd_per_conversation!=null?`$${c.usd_per_conversation} on the paid tier (${c.prompt_tokens_per_reply} prompt tokens + ${c.audio_s_per_reply} s of audio a reply); the free tier is 10 requests a day`:"pending");
    h+=`<tr><td><b>${esc(a.label)}</b>${a.note?`<div class="small">${esc(a.note)}</div>`:""}</td><td class="small">${esc(a.how)}</td><td class="num">${a.n}/${a.of}</td><td class="num">${a.judged?a.read_back+"/"+a.judged:"–"}</td><td class="num">${a.judged?a.dialect_ok+"/"+a.judged:"–"}</td><td class="num">${esc(a.accent??"–")}</td><td>${esc((a.jim||[]).join("/"))}</td><td>${esc((a.qaf||[]).join("/"))}</td><td>${esc((a.case||[]).join("/"))}</td><td class="small">${a.phones?(a.flags.length?esc(a.flags.join(" · ")):"none ("+a.phones+" heard)"):"–"}</td><td class="num">${ttTxt}</td><td class="small">${cTxt}</td></tr>`;
  }
  h+=`</table></div>`;
  const gu=S.gemini_used||[]; h+=`<p class="small">Gemini TTS requests used by this test: ${gu.length} of ${S.gemini_max}${gu.length?" — "+esc(gu.map(u=>u.at+" "+u.kind+" "+(u.result||"")).join("; ")):""}${S.gemini_schedule?`. Scheduled: ${esc(S.gemini_schedule)}`:""}.</p>`;
  if(S.lex&&Object.keys(S.lex).length) h+=`<p class="small">Lexicon: ${esc(Object.entries(S.lex).map(([a,b])=>a+" → "+b).join(" · "))}</p>`;
  for(const R of (S.replies||[])){
    h+=`<h3 style="margin-top:14px">${esc(R.label)}</h3><div class="ar">${esc(R.ar)}</div><div class="small">fully diacritized (the text the diacritized Azure row reads, before the pausal sukun):</div><div class="say">${esc(R.diac)}</div><div class="grid" style="margin-top:6px">`;
    const ts=D.takes.filter(t=>t.line==="sk_ar_"+R.key&&isSk(t)).sort((a,b)=>a.group.localeCompare(b.group)||a.model.localeCompare(b.model));
    for(const t of ts) h+=takeCard(t,{title:t.group==="sekka-ar-azure"?`Azure ${t.voice.replace("Neural","")} — ${t.model.replace(/^azure ?/,"")}`:(t.group==="sekka-gemini"?"Gemini Despina — six replies in one request":"Gemini Despina — this reply alone")});
    if(!ts.length) h+=`<div class="take"><b>no take yet</b></div>`;
    h+=`</div>`;
  }
  return h+`</section>`;
}
function viewAbout(){
  return `<section class="panel"><h2 style="margin-top:0">How these takes were made</h2>
  <p>Every take: <b>${esc(D.main_model)}</b> (or the compared model), one prebuilt voice, a director's-notes prompt (only the transcript is spoken). Arabic lines are the app's own (tools/masri_voice/masri_voice.py, masculine addressee); the voice reads the bank build's spelling (pronounce.py SAY). Each take is trimmed and levelled like a bank clip (build_bank.process) and encoded in the bank's format (MP3 24 kHz mono 32 kbps), so it plays at the same level as the CURRENT clip beside it.</p>
  <p>Judge: <b>${esc(D.judge_model)}</b> hears each take blind and returns a transcript and accent notes; the transcript is compared with the text (CER for Arabic with ق/ء, alef, ya, ta-marbuta and digits folded; WER for English); a mismatch, a lost or swapped direction word / ordinal, or a feminine form of the line is a FAIL and the line is regenerated once (the retake reads the next SAY spelling, as build_bank does). Arabic also needs the judge to call the accent Cairene with ج = g and ق = glottal stop.</p>
  <h3>Arabic prompt (example)</h3><pre>${esc(D.prompt_ar_example)}</pre><h3>English prompt (example)</h3><pre>${esc(D.prompt_en_example)}</pre></section>`;
}
function render(){
  $("#tabs").innerHTML=tabs.map(([k,n])=>`<button type="button" data-k="${k}" class="${k===cur?'on':''}">${n}</button>`).join("");
  $("#main").innerHTML=({line:viewLine,sekka:viewSekka,azure:viewAzure,voice:viewVoice,screen:viewScreen,batch:viewBatch,limits:viewLimits,about:viewAbout}[cur]||viewLine)();
  const lf=$("#lf"), ob=$("#onlybest");
  const apply=()=>{ document.querySelectorAll("section[data-lang]").forEach(s=>s.style.display=(lf.value==="all"||s.dataset.lang===lf.value)?"":"none");
    document.querySelectorAll("[data-take]").forEach(d=>{ const sec=d.closest("section"); const hasRetake=sec.querySelector('[data-take="2"]'); d.style.display=(ob.checked&&((d.dataset.take==="1"&&hasRetake&&d.dataset.pass==="false")))?"none":""; }); };
  if(lf){ lf.onchange=apply; ob.onchange=apply; }
}
$("#tabs").addEventListener("click",e=>{const b=e.target.closest("button"); if(!b) return; cur=b.dataset.k; try{localStorage.setItem("va_tab",cur);}catch(err){} render();});
$("#theme").onclick=()=>{const r=document.documentElement; const d=r.dataset.theme==="dark"||(!r.dataset.theme&&matchMedia("(prefers-color-scheme: dark)").matches); r.dataset.theme=d?"light":"dark";};
const nAr=D.takes.filter(t=>t.lang==="ar").length, nEn=D.takes.filter(t=>t.lang==="en").length;
$("#sub").textContent=`${D.generated} · ${nAr} Arabic + ${nEn} English takes · main model ${D.main_model} · judge ${D.judge_model} · picks: Arabic ${(D.state.ar_pick||[]).join(", ")} | English ${(D.state.en_pick||[]).join(", ")}`;
render();
</script>
</body>
</html>
"""


def rerank(R):
    f0r = R.state.get("ref_f0") or ref_f0()
    R.state["ar_rank"] = sorted(((score_ar(r), r["voice"]) for r in R.takes.values()
                                 if r["group"] == "prescreen" and r["line"] == "ar00_probe" and r["take"] == 1
                                 and r.get("check")), reverse=True)
    R.state["en_rank"] = sorted(((score_en(r, f0r), r["voice"]) for r in R.takes.values()
                                 if r["group"] == "prescreen" and r["line"] == "en00_probe" and r["take"] == 1
                                 and r.get("check")), reverse=True)


def cmd_rejudge(group, model):
    """Every take of `group` heard again by ONE judge model, batched (a consistent ranking after a fall-back)."""
    global LIM
    LIM = Limits()
    R = Run()
    for rec in R.takes.values():
        if not group or rec["group"] == group:
            rec["judge"], rec["check"], rec["pass"] = None, None, None
    left = judge_pending(R, groups={group} if group else None, force=model)
    R.state["ref_f0"] = R.state.get("ref_f0") or ref_f0()
    rerank(R)
    R.state["rejudged"] = {"group": group, "model": model, "left": left,
                           "at": datetime.datetime.now().isoformat(timespec="minutes")}
    R.save()
    write_page(R)
    log("rejudged with %s (%d left); Arabic rank %s; English rank %s" % (model, left, R.state["ar_rank"],
                                                                        R.state["en_rank"]))


def cmd_recut():
    """Cut every one-request take again (no network): the aligned cut replaces a cut made by pause length alone."""
    R = Run()
    lines = {ln["id"]: ln for ln in all_lines()}
    for bk, b in sorted((R.state.get("batches") or {}).items()):
        raw_p = os.path.join(CACHE, b["key"] + ".wav")
        if not os.path.exists(raw_p):
            continue
        wav_raw = open(raw_p, "rb").read()
        meta = json.load(io.open(os.path.join(CACHE, b["key"] + ".json"), encoding="utf-8"))
        todo = [lines[i] for i in b["lines"]]
        take = b["take"]
        texts = [pronounce.say(ln["seg"], (take - 1) % pronounce.variants(ln["seg"])) if ln.get("seg")
                 else say_text(ln, take) for ln in todo]
        pieces = split_on_silence(wav_raw, len(todo), weights=[letters_of(t) for t in texts])
        b["cut_ok"] = pieces is not None
        b["cut"] = "aligned"
        if pieces is None:
            log("  %s: no aligned cut" % bk)
            continue
        rate = acoustics.pcm(wav_raw)[1]
        for i, (ln, pc) in enumerate(zip(todo, pieces)):
            R.takes.pop(R.tid(b["group"], ln["id"], b["voice"], b["model"], take), None)
            m2 = dict(meta, batched=len(todo), batch_clip=b["clip"], piece=i,
                      audio_s=round(len(acoustics.pcm(pc)[0]) / float(rate), 2))
            R.record(ln, b["voice"], b["model"], b["group"], take, texts[i], b["key"], pc, m2,
                     context=bool(ln.get("seg")), sub="_q%d" % i)
        log("  %s: re-cut into %d pieces: %s" % (bk, len(pieces), " ".join(
            "%.1f" % (len(acoustics.pcm(pc)[0]) / float(rate)) for pc in pieces)))
    R.save()
    write_page(R)


def cmd_phones_azure(groups=("azure", "azure-tune", "azure-batch", "azure-batch-single")):
    """The blind phone listener over the Azure Arabic takes — on exactly what the judge heard (heard_wav: the
    processed take, behind the bank's lead for a bank segment). CPU only, no network; resumable (cached per clip)."""
    R = Run()
    lines = {ln["id"]: ln for ln in all_lines()}
    todo = [r for r in R.takes.values() if r["lang"] == "ar" and r["group"] in groups and not r.get("phones")]
    todo.sort(key=lambda r: (r["group"] != "azure", r["voice"], r["model"], r["line"]))
    n = 0
    for rec in todo:
        hp = os.path.join(CACHE, rec.get("heard_wav") or "-")
        if not os.path.exists(hp):
            continue
        ph = phones_verdict(lines[rec["line"]]["plain"], open(hp, "rb").read())
        if ph is None:
            break
        rec["phones"] = ph
        n += 1
        if n % 5 == 0:
            log("  phones: %d / %d" % (n, len(todo)))
            R.save()
    R.state["phones_error"] = PHONES["err"]
    R.save()
    write_page(R)
    log("phones (azure): %d takes%s" % (n, (" — " + PHONES["err"]) if PHONES["err"] else ""))


def cmd_phones():
    """Back-fill the blind phone listener over every Arabic take already made (CPU only, no network)."""
    R = Run()
    text = {ln["id"]: ln["plain"] for ln in all_lines()}
    n = 0
    for rec in R.takes.values():
        if rec["lang"] != "ar" or rec.get("phones"):
            continue
        wav_raw = open(os.path.join(CACHE, rec["key"] + ".wav"), "rb").read()
        ph = phones_verdict(text[rec["line"]], processed(wav_raw))
        if ph is None:
            break
        rec["phones"] = ph
        n += 1
        if n % 10 == 0:
            log("  phones: %d takes" % n)
            R.save()
    R.state["phones_error"] = PHONES["err"]
    R.save()
    write_page(R)
    log("phones: %d takes judged%s" % (n, (" — " + PHONES["err"]) if PHONES["err"] else ""))


# ════════════════════════════════════════════════════════════════════════════════════════════════
def cmd_lines():
    for ln in all_lines():
        print(ln["id"], "|", ln["label"])
        print("   text :", ln["plain"])
        if ln["lang"] == "ar":
            print("   says :", say_text(ln))
            miss = [s for s in ln["segments"] if not os.path.exists(os.path.join(BANK_DIR, mv.fnv64(s) + ".mp3"))]
            print("   bank :", "complete" if not miss else "MISSING " + " | ".join(miss))


def main():
    global LIM
    if sys.stdout is not None:               # pythonw (the scheduled Sekka run) has no console: --sekka-log
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lines", "run", "page", "limits", "phones", "phones-azure", "rejudge", "recut",
                                    "azure", "sekka"])
    ap.add_argument("--sekka-stages", default=None,
                    help="sekka: en-screen,en-degree,en-final,ar-azure,judge,gemini,judge-gemini,merge")
    ap.add_argument("--sekka-log", default=None, help="sekka: append everything printed to this file")
    ap.add_argument("--sekka-not-before", default=None,
                    help="sekka: an ISO UTC time; the gemini stage refuses to start before it (the quota day)")
    ap.add_argument("--ffmpeg", default=None, help="the ffmpeg executable (a scheduled run's PATH may lack it)")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--stages", default="prescreen,judge,audition,english,compare,raw,batch,retake")
    ap.add_argument("--voices-ar", default=None)
    ap.add_argument("--voices-en", default=None)
    ap.add_argument("--no-phones", action="store_true")
    ap.add_argument("--judge-model", default=None, help="one judge for every verdict (default: the chain)")
    ap.add_argument("--en-screen-model", default=None, help="the English pre-screen's TTS model (default: the main)")
    ap.add_argument("--test-model", default=None, help="the raw-spelling and batch tests' TTS model (default: the main)")
    ap.add_argument("--group", default="prescreen", help="rejudge: the take group ('' = every take)")
    ap.add_argument("--azure-stages", default=None,
                    help="azure: tune,final,screen,english,batch,judge (default: all but judge)")
    ap.add_argument("--azure-cap", type=int, default=None, help="azure: this desk's character cap (all runs)")
    ap.add_argument("--azure-groups", default=None, help="azure judge: the take groups to judge (comma list)")
    ap.add_argument("--judge-batch", type=int, default=None, help="clips per judge request (default JUDGE_BATCH)")
    a = ap.parse_args()
    if a.ffmpeg:
        global FFMPEG
        FFMPEG = a.ffmpeg
        os.environ["FFMPEG"] = a.ffmpeg       # azure.py reads it at call time
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(CLIPS, exist_ok=True)
    JUDGE_FORCE["model"] = a.judge_model
    if a.judge_batch:
        global JUDGE_BATCH
        JUDGE_BATCH = a.judge_batch
    if a.cmd == "lines":
        cmd_lines()
    elif a.cmd == "rejudge":
        try:
            cmd_rejudge(a.group, a.judge_model or JUDGE_MODEL)
        except Exception:
            import traceback
            log("rejudge stopped: " + scrub(traceback.format_exc()))
            raise SystemExit(1)
    elif a.cmd == "run":
        try:
            run(a)
        except Exception:           # never let a traceback carry the key out
            import traceback
            log("run stopped: " + scrub(traceback.format_exc()))
            raise SystemExit(1)
    elif a.cmd == "page":
        R = Run()
        write_page(R)
        print(os.path.join(OUT, "index.html"))
    elif a.cmd == "recut":
        cmd_recut()
    elif a.cmd == "azure":
        LIM = Limits()
        if a.no_phones:
            PHONES["on"] = False
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        R = Run()
        try:
            run_azure(R, a)
        except Exception:
            import traceback
            log("azure stopped: " + scrub(traceback.format_exc()))
            raise SystemExit(1)
        finally:
            R.state["phones_error"] = PHONES["err"]
            R.save()
            write_page(R)
    elif a.cmd == "sekka":
        if a.sekka_log:
            os.makedirs(os.path.dirname(os.path.abspath(a.sekka_log)), exist_ok=True)
            fh = io.open(a.sekka_log, "a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = fh
            log("==== sekka run %s UTC (%s)" % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                                a.sekka_stages))
        LIM = Limits()
        if a.no_phones:
            PHONES["on"] = False
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        R = Run()
        try:
            stages = (a.sekka_stages or "").split(",")
            if "gemini" in stages and a.sekka_not_before:
                nb = datetime.datetime.fromisoformat(a.sekka_not_before.replace("Z", "+00:00"))
                if datetime.datetime.now(datetime.timezone.utc) < nb:
                    raise SystemExit("sekka: the gemini stage may not start before %s" % a.sekka_not_before)
            run_sekka(R, a)
        except SystemExit as e:
            log(str(e))
        except Exception:
            import traceback
            log("sekka stopped: " + scrub(traceback.format_exc()))
        finally:
            R.state["phones_error"] = PHONES["err"]
            R.save()
            write_page(R)
            log("page: %s" % os.path.join(OUT, "index.html"))
    elif a.cmd == "phones":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        cmd_phones()
    elif a.cmd == "phones-azure":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        cmd_phones_azure()
    elif a.cmd == "limits":
        print(scrub(json.dumps(Limits().d, ensure_ascii=False, indent=1)))


if __name__ == "__main__":
    main()
