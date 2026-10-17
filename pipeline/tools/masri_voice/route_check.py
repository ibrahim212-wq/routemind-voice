# -*- coding: utf-8 -*-
"""Real routes through MasriVoice: does the line composed from the DATA say what the provider meant?

  python -m tools.masri_voice.route_check [--show N] [--json out.json]

Corpus (already on the desk, no new API calls): every Mapbox driving-traffic response in
tools/voice_sim/data/corpus/*.json (18 intercity routes) and tools/fork_sense/results/*.json (the
32-trip Cairo/Giza fork corpus, `mapbox` key) — every step, every voice instruction.

For each instruction the provider's ENGLISH announcement is parsed into what it tells the driver
(kind: continue / approach / final; the action: turn / keep at a fork / ramp / exit / roundabout
exit N / U-turn / merge / arrive …; its side; the distance; a "Then" clause) and compared with the
Egyptian line composed from the maneuver data alone. A disagreement is a real finding either way:
either the composer chose a different action than the data carries, or the English template says
something the data does not (both are listed).
"""
import glob
import io
import json
import os
import re
import sys

from tools.masri_voice import masri_voice as mv

ROOT = mv.ROOT
CORPORA = [os.path.join(ROOT, "tools", "voice_sim", "data", "corpus", "*.json"),
           os.path.join(ROOT, "tools", "fork_sense", "results", "*.json")]

ORD = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8,
       "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}


def load_routes():
    out = []
    for pat in CORPORA:
        for p in sorted(glob.glob(pat)):
            if p.endswith(".osm.json") or p.endswith(".ar.json") or os.path.basename(p) == "evidence.json":
                continue
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            r = d.get("mapbox") if isinstance(d, dict) and "mapbox" in d else d
            if not isinstance(r, dict) or not r.get("routes"):
                continue
            out.append((os.path.basename(p)[:-5], r["routes"][0]))
    return out


def english_kind(t):
    s = t.strip()
    if re.match(r"(?i)^continue for ", s):
        return "continue"
    if re.match(r"(?i)^in [\d.,]+ (meters?|kilometers?|miles?|feet)", s):
        return "approach"
    return "final"


def english_distance(t):
    m = re.match(r"(?i)^(?:continue for|in) ([\d.,]+) (meters?|kilometers?)", t.strip())
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * (1000.0 if m.group(2).lower().startswith("k") else 1.0)


def main_clause(t):
    s = re.sub(r"(?i)^in [\d.,]+ (meters?|kilometers?),\s*", "", t.strip())
    return re.split(r"(?i)\.\s*then,?\s*", s)[0]


def english_action(t):
    """→ (action, side, exit) from the provider's English, or None when it names no maneuver."""
    s = main_clause(t).lower()
    exit_n = None
    m = re.search(r"take the (\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth) exit", s)
    if m:
        exit_n = ORD.get(m.group(1))
    side = "right" if re.search(r"\bright\b", s) else ("left" if re.search(r"\bleft\b", s) else "")
    if "roundabout" in s or "rotary" in s or "traffic circle" in s:
        if s.startswith("exit the") or re.match(r"^exit ", s):
            return ("rbleave", "", None)
        return ("rb", side, exit_n)
    if "u-turn" in s:
        return ("uturn", "", None)
    if "arrive" in s or "destination" in s:
        return ("arrive", side, None)
    if re.search(r"\bkeep (left|right|straight)\b", s):
        return ("fork", side, None)
    if "take the ramp" in s:
        return ("ramp", side, None)
    if re.search(r"take the exit|take exit", s):
        return ("offramp", side, None)
    if s.startswith("merge"):
        return ("merge", side, None)
    if re.search(r"\bturn\b", s) or re.search(r"\bgo (left|right)\b", s):
        return ("turn", side, None)
    if re.search(r"\b(continue|head|drive|go) straight\b", s):
        return ("straight", "", None)
    if re.search(r"^(continue|head|drive|go)\b", s):
        return ("continue", side, None)
    return None


FAMILY_OF = {"turn": "turn", "sharp": "turn", "bend": "turn", "curve": "turn", "endofroad": "turn",
             "fork": "fork", "onramp": "ramp", "offramp": "offramp", "merge": "merge", "uturn": "uturn",
             "straight": "straight", "arrive": "arrive", "waypoint": "arrive", "rb": "rb", "rbturn": "rb",
             "rbleave": "rbleave"}


def maneuvers_of(leg):
    steps = leg.get("steps") or []
    out = []
    for k, st in enumerate(steps):
        man = st.get("maneuver") or {}
        deg = None
        if k > 0:
            b = (steps[k - 1].get("bannerInstructions") or [])
            if b:
                deg = (b[0].get("primary") or {}).get("degrees")
        out.append((mv.Maneuver(man.get("type"), man.get("modifier"), man.get("exit"), deg,
                                st.get("driving_side") or "right"), float(st.get("distance") or 0.0)))
    return out


def run(show=0, json_out=None):
    routes = load_routes()
    rows = []
    for name, route in routes:
        legs = route.get("legs") or []
        for li, leg in enumerate(legs):
            mans = maneuvers_of(leg)
            steps = leg.get("steps") or []
            for si, st in enumerate(steps):
                vis = st.get("voiceInstructions") or []
                for vj, vi in enumerate(vis):
                    en = vi.get("announcement") or ""
                    d = float(vi.get("distanceAlongGeometry") or 0.0)
                    ar = mv.route_line(mans, si, vj, len(vis), d, fem=False, last_leg=(li == len(legs) - 1))
                    if not ar:
                        rows.append(dict(route=name, step=si, en=en, ar="", verdict="no-next-maneuver"))
                        continue
                    kind = mv.instruction_kind(vj, len(vis), d)
                    fam = mv.family(mans[si + 1][0], li == len(legs) - 1)
                    then_ar = mv.pick("then") in ar
                    ek = english_kind(en)
                    ed = english_distance(en)
                    ea = english_action(en)
                    then_en = bool(re.search(r"(?i)\bthen\b", en))
                    row = dict(route=name, step=si, instr=vj, of=len(vis), d=round(d), en=en, ar=mv.plain(ar),
                               kind=kind, en_kind=ek, fam=list(fam), en_act=list(ea) if ea else None,
                               then=then_ar, en_then=then_en)
                    # kind: our "continue" vs the provider's "Continue for"; approach/final alike
                    row["kind_ok"] = (kind == ek) or (kind == mv.APPROACH and ek == "final" and d <= 400) or \
                                     (kind == mv.FINAL and ek == "approach" and d <= mv.FINAL_MAX_M)
                    if ed is not None and kind in (mv.APPROACH, mv.CONTINUE):
                        row["dist_ok"] = abs(mv.dist_band(d) - mv.dist_band(ed)) <= max(50, 0.1 * ed)
                    if ea and kind != mv.CONTINUE:
                        f = FAMILY_OF.get(fam[0], fam[0])
                        a = ea[0]
                        same = (f == a) or (a == "continue" and f in ("straight", "turn")) or \
                               (a == "turn" and f == "rb" and fam[0] == "rbturn") or \
                               (a == "ramp" and f in ("offramp", "ramp")) or (a == "offramp" and f == "ramp") or \
                               (a == "straight" and f in ("straight", "fork"))
                        row["act_ok"] = same
                        if same and ea[1] and fam[0] not in ("rb", "rbleave", "uturn", "straight"):
                            row["side_ok"] = (len(fam) > 1 and fam[1] == ea[1]) or fam[0] == "rbturn"
                        if a == "rb" and ea[2] is not None:
                            row["exit_ok"] = fam[0] == "rb" and fam[2] == ea[2]
                    if kind in (mv.FINAL, mv.APPROACH):
                        row["then_ok"] = then_ar == then_en
                    rows.append(row)
    keys = ["kind_ok", "dist_ok", "act_ok", "side_ok", "exit_ok", "then_ok"]
    print("routes %d  instructions %d" % (len(routes), len(rows)))
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            print("  %-8s %5d / %5d  = %.1f%%" % (k, sum(vals), len(vals), 100.0 * sum(vals) / len(vals)))
    if show:
        for k in keys:
            bad = [r for r in rows if r.get(k) is False]
            print("\n── %s disagreements (%d) ──" % (k, len(bad)))
            for r in bad[:show]:
                print("  %s s%d/%d d=%s  EN: %s\n      AR: %s   fam=%s en=%s then=%s/%s" % (
                    r["route"], r["step"], r.get("instr", 0), r.get("d"), r["en"], r["ar"], r.get("fam"),
                    r.get("en_act"), r.get("then"), r.get("en_then")))
    if json_out:
        json.dump(rows, io.open(json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    show = 0
    jo = None
    a = sys.argv[1:]
    if "--show" in a:
        show = int(a[a.index("--show") + 1])
    if "--json" in a:
        jo = a[a.index("--json") + 1]
    run(show, jo)
