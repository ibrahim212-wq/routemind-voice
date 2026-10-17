# -*- coding: utf-8 -*-
"""Everything MasriVoice can say — the segment inventory the clip bank must cover.

`all_segments()` walks EVERY input domain of the composer (every provider maneuver type × modifier
× exit × angle, every distance band, every event, every quiet-moment line, both genders, every
variant) and returns the set of segment texts that can reach a speaker. The bank builder renders
exactly this set; tool/masri_voice_parity.py proves (a) the bank has a clip for each, and (b)
random compositions never produce a segment outside it.
"""
from tools.masri_voice import masri_voice as mv

TYPES = ["turn", "new name", "depart", "arrive", "merge", "on ramp", "off ramp", "fork", "end of road",
         "continue", "roundabout", "rotary", "roundabout turn", "notification", "exit roundabout",
         "exit rotary", "use lane"]
MODIFIERS = [None, "uturn", "sharp right", "right", "slight right", "straight", "slight left", "left", "sharp left"]
EXITS = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DEGREES = [None, 20.0, 45.0, 90.0, 135.0, 160.0, 180.0, 200.0, 225.0, 270.0, 315.0, 340.0, 360.0]
SIDES = ["right", "left"]
MAX_KM = 1500


def all_bands():
    out, m = set(), 0.0
    while m <= MAX_KM * 1000.0:
        out.add(mv.dist_band(m))
        m += 10.0 if m < 5000 else (250.0 if m < 110000 else 2500.0)
    return sorted(out)


def all_traffic_distances():
    """TrafficAdvisor's distances: to a closure / an incident ahead (≤ 8 km) and what is left of a jam."""
    return [b for b in all_bands() if b <= 20000]


def all_minutes():
    return sorted({mv.minutes_band(x) for x in range(0, 24 * 60 + 1)})


def all_speeds():
    return sorted({mv.speed_band(x) for x in range(0, 221)})


def all_families():
    fams = {}
    for t in TYPES:
        for m in MODIFIERS:
            for e in EXITS:
                for d in DEGREES:
                    for ds in ("right", "left"):
                        for last in (True, False):
                            f = mv.family(mv.Maneuver(t, m, e, d, ds), last)
                            fams[repr(f)] = f
    return list(fams.values())


def variants(n=4):
    return range(n)


def all_segments():
    segs = set()

    def add_line(text):
        for s in mv.segments(text):
            segs.add(s)

    fams = all_families()
    for f in fams:
        for form in ("approach", "final", "remind", "now"):
            for v in variants():
                for fem in (False, True):
                    for s in mv.action(f, form, v, fem):
                        segs.add(s)
                        segs.add(mv.comma(s))
    for b in all_bands():
        for v in variants(2):
            segs.add(mv.seg_lead(b, v))
        segs.add(mv.seg_cont_dist(b))
    for v in variants():
        for fem in (False, True):
            segs.add(mv.comma(mv.pick("cont.verb", v, fem)))
            segs.add(mv.pick("then", v, fem))
    # events
    for v in variants():
        for fem in (False, True):
            for tunnel in (False, True):
                for side in ("right", "left", ""):
                    for pre in ("right", "left", ""):
                        add_line(mv.ev_bridge_take(tunnel, side, None, pre, v, fem))
                        add_line(mv.ev_bridge_take(tunnel, side, 300.0, pre, v, fem))
                    add_line(mv.ev_bridge_dont(tunnel, side, None, v, fem))
                    add_line(mv.ev_bridge_dont(tunnel, side, 300.0, v, fem))
                add_line(mv.ev_bridge_optional(tunnel, v, fem))
            for side in ("right", "left", "", "straight"):
                add_line(mv.ev_fork(side, None, v, fem))
                add_line(mv.ev_fork(side, None, v, fem, True))
                add_line(mv.ev_fork(side, 400.0, v, fem))
            for k in mv.CAMERA_KINDS:
                add_line(mv.ev_camera(k, 500.0, None, v, fem))
                for sp in all_speeds():
                    add_line(mv.ev_camera(k, 500.0, sp, v, fem))
            for k in mv.REPORT_KINDS:
                add_line(mv.ev_report(k, 500.0, None, v, fem))
                add_line(mv.ev_report(k, 500.0, 60, v, fem))
            for k in ("wrongway", "offroute", "offroute.missed", "offline.coverage_exit", "gps.lost",
                      "traffic.switched", "alt.switched", "stop.reached", "mark"):
                add_line(mv.ev_simple(k, v, fem))
            for mn in all_minutes():
                add_line(mv.ev_faster_route(mn, v, fem))
                for k in ("offer.faster", "offer.jam", "offer.stuck"):
                    add_line(mv.ev_traffic(k, None, mn, v, fem))
                add_line(mv.ev_traffic("jam", 2000.0, mn, v, fem))
                add_line(mv.ev_traffic("stuck", None, mn, v, fem))
            add_line(mv.ev_traffic("offer.closed", None, None, v, fem))
            add_line(mv.ev_traffic("jam", 2000.0, None, v, fem))
            add_line(mv.ev_traffic("stuck", None, None, v, fem))
            for dm in all_traffic_distances():
                for k in ("closed", "accident", "incident"):
                    add_line(mv.ev_traffic(k, dm, None, v, fem))
                add_line(mv.ev_traffic("stuck", dm, None, v, fem))
            for side in ("right", "left", "straight", "none"):
                for last in (True, False):
                    for extra in (None, "qm.arrive", "qm.arrive.night", "qm.arrive.long"):
                        add_line(mv.ev_arrival(side, last, v, fem, extra))
    # quiet moments
    for v in variants(6):
        for fem in (False, True):
            for hour in range(24):
                for wd in range(7):
                    for hm, hd in ((1, 1), (9, 5), (10, 1), (12, 10)):
                        for trip in (None, 1, 12):
                            add_line(mv.start_line(hour, wd, hm, hd, trip, v, fem))
                            add_line(mv.start_line(hour, wd, hm, hd, trip, v, fem, False))
            for cat in ("long", "jam", "iftar"):
                add_line(mv.quiet_line(cat, v, fem))
    for mn in all_minutes():
        segs.add(mv.seg_eta(mn))
    return segs


CATEGORIES = [
    ("dist", "المسافة — «بعد …» / «كمان …»"),
    ("about", "«حوالي …» — كمّل على طول"),
    ("action", "اللفّات — يمين، شمال، ميّل، التفريعة، الطلعة، المخرج، الصينية، الملف"),
    ("then", "كمّل / وبعدها"),
    ("bridge", "الكباري والأنفاق"),
    ("fork", "التفريعة (ForkSense)"),
    ("camera", "الرادارات والسرعة"),
    ("report", "البلاغات"),
    ("event", "الطريق اتغيّر، عكس الاتجاه، الـ GPS، الخرايط، الوقفات"),
    ("faster", "طريق أسرع — هيوفّرلك …"),
    ("arrive", "الوصول"),
    ("start", "أول المشوار"),
    ("eta", "«المشوار حوالي …»"),
    ("quiet", "لحظات هادية — طريق طويل، زحمة، المغرب"),
]


def categorized():
    """{segment: category key} — the same segments as all_segments(), each under the FIRST situation
    that produces it (in CATEGORIES order), for the listening page."""
    out = {}

    def put(cat, segs):
        for s in segs:
            out.setdefault(s, cat)

    def line(text):
        return mv.segments(text)

    put("dist", [mv.seg_lead(b, v) for b in all_bands() for v in variants(2)])
    put("about", [mv.seg_cont_dist(b) for b in all_bands()])
    acts = []
    for f in all_families():
        for form in ("approach", "final", "remind", "now"):
            for v in variants():
                for fem in (False, True):
                    for s in mv.action(f, form, v, fem):
                        acts += [s, mv.comma(s)]
    put("action", acts)
    put("then", [x for v in variants() for fem in (False, True)
                 for x in (mv.comma(mv.pick("cont.verb", v, fem)), mv.pick("then", v, fem))])
    for v in variants():
        for fem in (False, True):
            for tunnel in (False, True):
                for side in ("right", "left", ""):
                    for pre in ("right", "left", ""):
                        put("bridge", line(mv.ev_bridge_take(tunnel, side, None, pre, v, fem)))
                        put("bridge", line(mv.ev_bridge_take(tunnel, side, 300.0, pre, v, fem)))
                    put("bridge", line(mv.ev_bridge_dont(tunnel, side, None, v, fem)))
                    put("bridge", line(mv.ev_bridge_dont(tunnel, side, 300.0, v, fem)))
                put("bridge", line(mv.ev_bridge_optional(tunnel, v, fem)))
            for side in ("right", "left", "", "straight"):
                put("fork", line(mv.ev_fork(side, None, v, fem)))
                put("fork", line(mv.ev_fork(side, None, v, fem, True)))
                put("fork", line(mv.ev_fork(side, 400.0, v, fem)))
            for k in mv.CAMERA_KINDS:
                put("camera", line(mv.ev_camera(k, 500.0, None, v, fem)))
                for sp in all_speeds():
                    put("camera", line(mv.ev_camera(k, 500.0, sp, v, fem)))
            for k in mv.REPORT_KINDS:
                put("report", line(mv.ev_report(k, 500.0, None, v, fem)))
                put("report", line(mv.ev_report(k, 500.0, 60, v, fem)))
            for k in ("wrongway", "offroute", "offroute.missed", "offline.coverage_exit", "gps.lost",
                      "traffic.switched", "alt.switched", "stop.reached", "mark"):
                put("event", line(mv.ev_simple(k, v, fem)))
            for mn in all_minutes():
                put("faster", line(mv.ev_faster_route(mn, v, fem)))
                put("traffic", line(mv.ev_traffic("offer.jam", None, mn, v, fem)))
                put("traffic", line(mv.ev_traffic("jam", 2000.0, mn, v, fem)))
            for dm in all_traffic_distances():
                put("traffic", line(mv.ev_traffic("stuck", dm, None, v, fem)))
                put("traffic", line(mv.ev_traffic("closed", dm, None, v, fem)))
            for side in ("right", "left", "straight", "none"):
                for last in (True, False):
                    for extra in (None, "qm.arrive", "qm.arrive.night", "qm.arrive.long"):
                        put("arrive", line(mv.ev_arrival(side, last, v, fem, extra)))
    for v in variants(6):
        for fem in (False, True):
            for hour in range(24):
                for wd in range(7):
                    for hm, hd in ((1, 1), (9, 5), (10, 1), (12, 10)):
                        for trip in (None, 1, 12):
                            put("start", line(mv.start_line(hour, wd, hm, hd, trip, v, fem)))
                            put("start", line(mv.start_line(hour, wd, hm, hd, trip, v, fem, False)))
            for cat in ("long", "jam", "iftar"):
                put("quiet", line(mv.quiet_line(cat, v, fem)))
    put("eta", [mv.seg_eta(mn) for mn in all_minutes()])
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    s = all_segments()
    chars = sum(len(x) for x in s)
    print(len(s), "segments,", chars, "chars")
    print(len(all_bands()), "distance bands;", len(all_minutes()), "minute bands;", len(all_speeds()), "speeds")
