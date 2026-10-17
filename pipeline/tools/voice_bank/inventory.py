# -*- coding: utf-8 -*-
"""The navigation bank's inventory, RANKED by how often a driver hears each segment — so the first build batch covers
the most spoken lines and can ship before the whole bank is done (docs/voice-bank-plan-2026-10.md §h).

    python -m tools.voice_bank.inventory [--json OUT] [--show 30]

WHAT IS COUNTED (no network, no key):
  • the REAL route corpora already on the desk — tools/voice_sim/data/corpus/*.json (18 intercity routes) and
    tools/fork_sense/results/*.json (the 32-trip Cairo / Giza fork corpus): every voice instruction of every step,
    composed exactly as the phone composes it (masri_voice.route_line with a Told per route, so the staged remind /
    now lines appear as often as they are spoken) → the Arabic segments of every line; the proposed English line
    for the same instruction (lead + action, the staged forms; bank_plan.english_segments' vocabulary) → the English
    segments; every step's road name matched against the major-road list → the road segments.
  • what no route corpus carries (the app's own events: arrival, trip start, reroute, cameras, bridges, forks,
    traffic offers, quiet moments) is ranked by how often a TRIP says it — a prior per category (EVENT_PER_TRIP),
    stated here, not measured.

THE ORDER: segments are sorted by expected plays per 100 trips (corpus count normalised to the corpus' trips; events
by their prior). Because a line plays from the bank only when EVERY segment of it is there, the report also gives the
LINE coverage of the corpus after the first N segments — the size of a first batch that already speaks most lines.
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import unicodedata

from tools.masri_voice import inventory as inv
from tools.masri_voice import masri_voice as mv
from tools.masri_voice import pronounce
from tools.masri_voice import route_check as rc
from tools.voice_audition import bank_plan as bp

OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
ROADS_JSON = os.path.join(os.environ.get("VOICE_AUDITION_OUT", "G:/grad_all/.p5work/voice_audition"),
                          "major_roads.json")

# Plays per trip of the lines no route corpus carries — a PRIOR (a typical 20-minute Cairo trip), used only to order
# them against the corpus-counted maneuver lines. Arrival / trip start once per trip; a reroute every other trip; a
# camera / bridge / fork call or two; traffic offers and quiet moments rarely.
EVENT_PER_TRIP = {
    "arrival": 1.0, "trip start": 1.0, "distance leads «بعد … / كمان …»": 0.0, "events": 0.5, "bridges & tunnels": 1.5,
    "cameras": 1.5, "reports": 0.3, "faster route": 0.3, "TrafficAdvisor": 0.3, "«المشوار حوالي …»": 0.5,
    "quiet moments": 0.2, "other": 0.05,
    # English categories (bank_plan.english_segments)
    "event": 0.5, "camera": 1.5, "report": 0.3, "bridge": 1.5, "fork": 0.5, "traffic": 0.3,
}


def norm_name(s):
    """The fold road names are matched by (the search engine's spirit: alef / ya / ta-marbuta, no tatweel, spaces)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub("[\u064B-\u0652\u0640]", "", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    s = re.sub(r"[,،.]", " ", s)
    return " ".join(s.lower().split())


def roads():
    if not os.path.exists(ROADS_JSON):
        return []
    d = json.load(io.open(ROADS_JSON, encoding="utf-8"))
    return [r for r in d["roads"] if r["km"] >= bp.ROAD_MIN_KM]


def en_line(kind, fam, band, told):
    """The proposed composed English line's segments for one instruction (bank_plan's vocabulary)."""
    if kind == mv.CONTINUE:
        return ["Continue for %s." % mv.en_dist(band)]
    if kind == mv.FINAL:
        return [mv.en_action(fam, 1)]
    return [mv.EN["lead"].replace("{D}", mv.en_dist(band)).strip(), mv.en_action(fam, 0) + "."]


def corpus_counts(road_list):
    """(arabic Counter, english Counter, road Counter, lines [(lang, [segments])], trips)."""
    ar, en, rd = collections.Counter(), collections.Counter(), collections.Counter()
    lines = []
    by_norm = {}
    for r in road_list:
        for nm in [r["ar"]] + list(r.get("names") or []):
            by_norm[norm_name(nm)] = r
        if r.get("en"):
            by_norm[norm_name(r["en"])] = r
    routes = rc.load_routes()
    for name, route in routes:
        told = mv.Told()
        told.sync(name)
        told_en = set()
        for li, leg in enumerate(route.get("legs") or []):
            mans = rc.maneuvers_of(leg)
            steps = leg.get("steps") or []
            last = li == len(route.get("legs") or []) - 1
            for si, st in enumerate(steps):
                rn = by_norm.get(norm_name(st.get("name") or ""))
                if rn:
                    rd[rn["ar"]] += 1
                vis = st.get("voiceInstructions") or []
                for vj, vi in enumerate(vis):
                    d = float(vi.get("distanceAlongGeometry") or 0.0)
                    a = mv.route_line(mans, si, vj, len(vis), d, fem=False, last_leg=last, told=told, leg=li)
                    if not a:
                        continue
                    segs = mv.segments(a)
                    ar.update(segs)
                    lines.append(("ar", segs))
                    if si + 1 < len(mans):
                        kind = mv.instruction_kind(vj, len(vis), d)
                        fam = mv.family(mans[si + 1][0], last)
                        try:
                            e = en_line(kind, fam, mv.dist_band(d), (li, si + 1) in told_en)
                        except (KeyError, ValueError):
                            e = []
                        if kind != mv.CONTINUE:
                            told_en.add((li, si + 1))
                        en.update(e)
                        if e:
                            lines.append(("en", e))
    return ar, en, rd, lines, max(1, len(routes))


def phrasebook_sha():
    import hashlib
    return hashlib.sha256(open(os.path.join(mv.ROOT, "nav-spec", "masri_voice.json"), "rb").read()).hexdigest()


def freeze(path):
    """The ranked inventory as data (segment texts, categories, ranks, the corpus lines as segment lists, the trip
    count — no route geometry): the GitHub Actions build (tools/voice_bank/ci/voice-bank.yml) reads THIS instead of
    the route corpora and the road list, which stay on the desk. Written by tools/voice_bank/mirror.py."""
    order, lines, trips = _ranked_live()
    doc = {"phrasebook": phrasebook_sha(), "trips": trips, "order": order, "lines": [[lg, s] for lg, s in lines]}
    io.open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, ensure_ascii=False, indent=0) + "\n")
    return doc


def ranked():
    """[{seg, lang, cat, per100, n, chars_say}] in build order + the corpus lines (for coverage). With
    $VOICE_BANK_INVENTORY (the Actions build) the frozen copy — refused when it was made from another phrasebook."""
    frozen = os.environ.get("VOICE_BANK_INVENTORY")
    if frozen:
        d = json.load(io.open(frozen, encoding="utf-8"))
        if d["phrasebook"] != phrasebook_sha():
            raise RuntimeError("the frozen inventory %s was made from another phrasebook — run "
                               "`python tools/voice_bank/mirror.py push` in the app repository" % frozen)
        return d["order"], [(lg, s) for lg, s in d["lines"]], d["trips"]
    return _ranked_live()


def _ranked_live():
    road_list = roads()
    ar_c, en_c, rd_c, lines, trips = corpus_counts(road_list)
    cat_ar = inv.categorized()
    out = []
    for s in sorted(inv.all_segments()):
        cat = cat_ar.get(s, "other")
        per100 = 100.0 * ar_c[s] / trips if ar_c[s] else 100.0 * EVENT_PER_TRIP.get(cat, 0.02) / 4.0
        out.append({"seg": s, "lang": "ar", "cat": cat, "n": ar_c[s], "per100": round(per100, 3),
                    "say": pronounce.say(s, 0)})
    for r in road_list:
        for s in ("على " + r["ar"], mv.comma("على " + r["ar"])):
            n = rd_c[r["ar"]]
            out.append({"seg": s, "lang": "ar", "cat": "road", "n": n,
                        "per100": round(100.0 * n / trips + r["km"] / 1000.0, 3), "say": pronounce.say(s, 0)})
    for s, cat in bp.english_segments(road_list).items():
        if cat == "road":
            nm = s[len("onto "):-1]
            r = next((x for x in road_list if x.get("en") == nm), None)
            n = rd_c[r["ar"]] if r else 0
            per100 = 100.0 * n / trips + ((r or {}).get("km", 0) / 1000.0)
        else:
            n = en_c[s]
            per100 = 100.0 * n / trips if n else 100.0 * EVENT_PER_TRIP.get(cat, 0.02) / 4.0
        out.append({"seg": s, "lang": "en", "cat": cat, "n": n, "per100": round(per100, 3), "say": s})
    out.sort(key=lambda x: (-x["per100"], x["lang"], x["seg"]))
    for i, x in enumerate(out):
        x["rank"] = i + 1
    return out, lines, trips


def coverage(order, lines, lang):
    """Fraction of the corpus' `lang` lines fully playable after the first N segments of `order` (that language)."""
    pos = {}
    k = 0
    for x in order:
        if x["lang"] == lang:
            k += 1
            pos[x["seg"]] = k
    need = []
    for lg, segs in lines:
        if lg != lang:
            continue
        need.append(max(pos.get(s, 10 ** 9) for s in segs))
    need.sort()
    return need, k


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(OUT, "inventory.json"))
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    order, lines, trips = ranked()
    rep = {"trips_in_corpus": trips, "segments": len(order), "by_lang": {}, "coverage": {}}
    for lang in ("ar", "en"):
        xs = [x for x in order if x["lang"] == lang]
        need, k = coverage(order, lines, lang)
        cov = {}
        for frac in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
            if need:
                idx = min(len(need) - 1, max(0, int(round(frac * len(need))) - 1))
                cov["%d%%" % int(frac * 100)] = need[idx] if need[idx] < 10 ** 9 else None
        rep["coverage"][lang] = {"corpus_lines": len(need), "segments_needed_for_line_share": cov,
                                 "lines_not_coverable": sum(1 for n in need if n >= 10 ** 9)}
        rep["by_lang"][lang] = {
            "segments": len(xs), "chars": sum(len(x["say"]) for x in xs),
            "chars_without_roads": sum(len(x["say"]) for x in xs if x["cat"] != "road"),
            "road_segments": sum(1 for x in xs if x["cat"] == "road"),
            "seen_in_corpus": sum(1 for x in xs if x["n"]),
        }
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    io.open(a.json, "w", encoding="utf-8").write(json.dumps({"report": rep, "order": order}, ensure_ascii=False,
                                                            indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    for x in order[:a.show]:
        print("%5d %s %-8s %7.2f/100 trips  n=%-4d %s" % (x["rank"], x["lang"], x["cat"][:8], x["per100"], x["n"],
                                                         x["seg"]))


if __name__ == "__main__":
    main()
