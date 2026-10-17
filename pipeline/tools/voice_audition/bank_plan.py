# -*- coding: utf-8 -*-
"""The numbers behind docs/voice-bank-plan-2026-10.md — every count in the plan is computed here (no key, no audio).

    python -m tools.voice_audition.bank_plan [--json out.json]

  • ARABIC: the segments MasriVoice can compose (tools/masri_voice/inventory.all_segments — the set the parity gate
    counts) and which have a clip in assets/voice/ar; plus the proposed road-name segments.
  • ENGLISH: the PROPOSED composed inventory (english_segments below) — the staged route lines from the phrasebook's
    English tables (masri_voice.json "english", the same distance bands as Arabic), every app event line the
    Android code speaks (cameras, reports, bridges, forks, TrafficAdvisor, reroute / GPS / offline / arrival), and the
    road names. The provider's free text (Mapbox's English with arbitrary road names) cannot be banked: it is replaced
    by the composed first line (+ a road name when the road is on the list).
  • ROADS: G:/grad_all/.p5work/voice_audition/major_roads.json (tools/voice_audition/major_roads.py).
  • THE FREE TIER: the quota answers the audition recorded (limits.json beside the page).
"""
import argparse
import collections
import io
import json
import math
import os
import statistics
import sys

from tools.masri_voice import inventory as inv
from tools.masri_voice import masri_voice as mv

ROOT = mv.ROOT
OUT = os.environ.get("VOICE_AUDITION_OUT", "G:/grad_all/.p5work/voice_audition")
BANK = os.path.join(ROOT, "assets", "voice", "ar")
EN = mv.EN
BYTES_PER_S = 4000          # the bank's format: MPEG-2 L3 24 kHz mono 32 kbps CBR = 4,000 bytes a second
ROAD_MIN_KM = 2.0


# ── English, composed ────────────────────────────────────────────────────────────────────────────────
def en_minutes_words(band):
    b = int(band)
    if b < 60:
        return "1 minute" if b == 1 else "%d minutes" % b
    h, r = b // 60, b % 60
    hs = "1 hour" if h == 1 else "%d hours" % h
    return hs if r == 0 else hs + (" and %d minutes" % r)


def ta_dist(m):
    """TrafficAdvisor.en_dist's own banding (100 m steps under 950 m, then half kilometres)."""
    if m < 950.0:
        v = max(100, int(math.floor(m / 100.0 + 0.5)) * 100)
        return "%d meters" % v
    half = int(math.floor(m / 500.0 + 0.5))
    if half % 2 == 0:
        k = half // 2
        return "1 kilometer" if k == 1 else "%d kilometers" % k
    return "%d.5 kilometers" % (half // 2)


def english_segments(roads=()):
    """{segment: category} — the proposed English bank. Like Arabic, a line = segments joined with the pause their
    punctuation asks for; every segment that can be followed by more has its own comma / open form."""
    segs = {}

    def add(cat, s):
        segs.setdefault(s, cat)

    bands = inv.all_bands()
    for b in bands:
        add("lead", EN["lead"].replace("{D}", mv.en_dist(b)).strip())            # "In 200 meters,"
        add("continue", "Continue for %s." % mv.en_dist(b))                     # the CONTINUE line
    for fam in inv.all_families():
        rem, now = mv.en_action(fam, 0), mv.en_action(fam, 1)
        add("action", rem + ".")                                                 # closes a line
        add("action", rem)                                                       # a road name follows
        add("then", "Then " + rem + ".")
        add("now", now)
        add("now", now.rstrip(".") + ",")                                        # "Turn right," + Then / a road
    add("action", "at the roundabout,")
    # cameras (phone: 4 kinds × 500 m / 1 km, the limit; car: "Speed camera ahead")
    kinds = ["Speed camera", "Mobile speed camera", "Red light camera", "Average speed zone"]
    for k in kinds:
        for d in ("500 meters", "1 kilometer"):
            add("camera", "%s in %s." % (k, d))
            add("camera", "%s in %s," % (k, d))
    add("camera", "Speed camera ahead.")
    add("camera", "Speed camera ahead,")
    for sp in inv.all_speeds():
        add("camera", "limit %d kilometers per hour." % sp)
    # user reports (5 kinds × 2 distances)
    for d in ("500 meters", "1 kilometer"):
        for t in ("Speed camera reported, %s ahead.", "Accident reported %s ahead, drive carefully.",
                  "Police reported %s ahead.", "Road hazard %s ahead.", "Incident reported %s ahead."):
            add("report", t % d)
    # bridges / underpasses (BridgeAnalyzer): leads are the shared "In N meters," segments
    for obj in ("the bridge", "the underpass"):
        add("bridge", "take %s." % obj)
        for side in ("right", "left"):
            add("bridge", "take %s on your %s." % (obj, side))
        add("bridge", "Don't take %s on your right, stay on the main road." % obj)
        add("bridge", "Don't take %s on your left, keep straight on the road." % obj)
        add("bridge", "Don't take %s ahead, stay on the main road." % obj)
        add("bridge", "Optional: you can take %s to skip ahead. It rejoins your road, either one works." % obj)
    for side in ("right", "left"):
        add("bridge", "turn %s, then" % side)
    # forks (ForkSense) — early, final, final after the early call
    for side in ("left", "right"):
        add("fork", "at the fork, keep %s." % side)
        add("fork", "Keep %s at the fork." % side)
    for s in ("Keep left.", "Keep right.", "Keep straight."):
        add("fork", s)
    # TrafficAdvisor — fixed phrases + minutes + its own distances
    for s in ("The road ahead is closed.", "There's another route.", "Want to take it?", "Heavy traffic ahead.",
              "There's a faster route that saves about", "You're in heavy traffic.", "A faster route saves about",
              "Found a faster route that saves about", "The road is closed in", "Accident reported in",
              "Road incident in", "It continues for about", "Heavy traffic in", "Slow down.", "Take care.",
              "of delay."):
        add("traffic", s)
    for mn in inv.all_minutes():
        if mn > 120:
            continue
        add("traffic", en_minutes_words(mn) + ".")
        add("traffic", "about " + en_minutes_words(mn))
    for m in range(100, 8001, 50):
        add("traffic", ta_dist(float(m)) + ".")
        add("traffic", ta_dist(float(m)) + ",")
    # events
    for s in ("Finding a new route.", "You missed the turn.", "Wrong way. Make a U-turn to return to the route.",
              "The GPS signal is weak here. I'll keep guiding you.",
              "You've left your downloaded maps. I'll keep guiding on the current route.", "Okay, taking this route.",
              "Okay, switched to the faster route.", "You've arrived.", "You're here.", "Stop reached.",
              "Good morning.", "Good afternoon.", "Good evening."):
        add("event", s)
    for r in roads:
        if r.get("en"):
            add("road", "onto %s." % r["en"])
            add("road", "onto %s," % r["en"])
    return segs


def en_seconds(s, cps):
    return max(0.6, len(s) / cps)


# ── the plan ─────────────────────────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(OUT, "bank_plan.json"))
    a = ap.parse_args()
    roads_all = json.load(io.open(os.path.join(OUT, "major_roads.json"), encoding="utf-8"))
    roads = [r for r in roads_all["roads"] if r["km"] >= ROAD_MIN_KM]
    manifest = json.load(io.open(os.path.join(ROOT, "tools", "masri_voice", "bank_manifest.json"), encoding="utf-8"))
    clips = manifest["clips"]

    # Arabic
    segs = sorted(inv.all_segments())
    files = set(os.listdir(BANK))
    have = [s for s in segs if mv.fnv64(s) + ".mp3" in files]
    miss = [s for s in segs if mv.fnv64(s) + ".mp3" not in files]
    cat = inv.categorized()
    by_cat = collections.Counter(cat.get(s, "other") for s in segs)
    miss_cat = collections.Counter(cat.get(s, "other") for s in miss)
    ms = [c["ms"] for c in clips.values()]
    ar_mean_s = statistics.mean(ms) / 1000.0
    ar_cps = sum(len(c["text"]) for c in clips.values()) / (sum(ms) / 1000.0)
    ar_road = []
    for r in roads:
        ar_road += ["على " + r["ar"], mv.comma("على " + r["ar"])]
    ar_road_s = sum(len(x) / ar_cps for x in ar_road)

    # English
    en = english_segments(roads)
    en_cat = collections.Counter(en.values())
    # English characters per second of speech: measured on the audition's English takes (median), else 13
    EN_CPS = 13.0
    tk_p = os.path.join(OUT, "takes.json")
    if os.path.exists(tk_p):
        rates = []
        for t in json.load(io.open(tk_p, encoding="utf-8")).values():
            st = t.get("stats") or {}
            if t.get("lang") == "en" and st.get("speech_s"):
                txt = t.get("tts_text") or ""
                rates.append(len(txt) / st["speech_s"])
        if rates:
            EN_CPS = statistics.median(rates)
    en_s = sum(en_seconds(s, EN_CPS) for s in en)
    en_s_noroad = sum(en_seconds(s, EN_CPS) for s, c in en.items() if c != "road")

    full_ar = len(segs) + len(ar_road)
    full_en = len(en)
    total = full_ar + full_en
    ar_s = sum(len(s) / ar_cps for s in segs) + ar_road_s

    res = {
        "arabic": {"segments": len(segs), "with_clip": len(have), "missing": len(miss),
                   "by_category": dict(by_cat), "missing_by_category": dict(miss_cat),
                   "road_segments": len(ar_road), "roads": len(roads), "total": full_ar,
                   "bank_mean_clip_s": round(ar_mean_s, 2), "chars_per_s": round(ar_cps, 1),
                   "audio_s": round(ar_s), "bytes": int(ar_s * BYTES_PER_S),
                   "core_audio_s": round(ar_s - ar_road_s), "core_bytes": int((ar_s - ar_road_s) * BYTES_PER_S),
                   "road_audio_s": round(ar_road_s), "road_bytes": int(ar_road_s * BYTES_PER_S)},
        "english": {"segments": full_en, "chars_per_s": round(EN_CPS, 1), "by_category": dict(en_cat), "audio_s": round(en_s),
                    "audio_s_without_roads": round(en_s_noroad), "bytes": int(en_s * BYTES_PER_S),
                    "core_bytes": int(en_s_noroad * BYTES_PER_S), "road_bytes": int((en_s - en_s_noroad) * BYTES_PER_S),
                    "roads_with_en_name": sum(1 for r in roads if r.get("en"))},
        "roads": {"rule": roads_all["rule"], "names_total": roads_all["names_total"], "selected": len(roads),
                  "city": sum(1 for r in roads if set(r["in"]) & {"cairo_giza", "alexandria"}),
                  "national_only": sum(1 for r in roads if r["in"] == ["egypt_motorway_trunk"])},
        "total_segments": total,
        "bytes_total": int((ar_s + en_s) * BYTES_PER_S),
        "current_bank_bytes": sum(c.get("bytes", 0) for c in clips.values()),
    }
    # the request arithmetic (free tier: TTS 10 requests / day / model, a text judge 20 / day / model — measured)
    lim_p = os.path.join(OUT, "limits.json")
    lim = json.load(io.open(lim_p, encoding="utf-8")) if os.path.exists(lim_p) else {"models": {}}
    measured = {}
    for m, x in lim["models"].items():
        for v in x.get("violations") or []:
            for q in v.get("violations") or []:
                measured[m] = {"quotaId": q.get("quotaId"), "quotaValue": q.get("quotaValue")}
    res["measured_quota"] = measured
    takes_per_seg = 1.3        # build_bank's experience: most clips pass first time; a retake for the rest
    judged_per_seg = 2.0       # the clip alone + inside a line (direction words / ordinals), as build_bank
    scen = []
    for per_req in (1, 10, 25, 40):
        tts_req = math.ceil(total * takes_per_seg / per_req)
        for judge_models, clips_per_judge in ((3, 10), (5, 15)):
            judge_req = math.ceil(total * takes_per_seg * judged_per_seg / clips_per_judge)
            days = max(math.ceil(tts_req / 10.0), math.ceil(judge_req / (20.0 * judge_models)))
            scen.append({"segments_per_tts_request": per_req, "tts_requests": tts_req,
                         "judge_models": judge_models, "clips_per_judge_request": clips_per_judge,
                         "judge_requests": judge_req, "days": days})
    res["scenarios"] = scen
    io.open(a.json, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scenarios"}, ensure_ascii=False, indent=1))
    for s in scen:
        print(s)


if __name__ == "__main__":
    main()
