# -*- coding: utf-8 -*-
"""MasriVoice — the navigation voice in Egyptian colloquial Arabic, COMPOSED from maneuver data.

REFERENCE copy (Python). Mirrors android/app/src/main/java/com/routemind/app/nav/MasriVoice.kt and
ios/Runner/MasriVoice.swift — the same rules, the same rounding, the same variant choice. The words
live in ONE file, nav-spec/masri_voice.json; the Kotlin / Swift tables are generated from it
(tools/masri_voice/gen_tables.py). tool/masri_voice_parity.py writes the fixture every copy must
reproduce (MasriVoiceTest replays it on the JVM) and proves the clip bank covers every segment.

WHY (2026-09-23). The Arabic voice used to be the provider's Arabic text, rewritten: Mapbox's
template Fusha ("بمسافة 400 أمتار، الاتجاه نحو اليمين") through a regex table (ArabicVoice). A
rewrite can only ever be as good as the template it matches, it already cut the direction out of
212 of 273 lines once, and it read digits as Fusha numerals. Here nothing the provider WROTE is
spoken: the provider's maneuver DATA (type, modifier, exit, degrees, the distance to the maneuver)
picks a phrase an Egyptian would say, and numbers are Egyptian words ("ميتين متر", "كيلو ونص").

THE LINE. A spoken line is a list of SEGMENTS, each one exact phrasebook text, joined by U+2063
and led by one (`SEP`) — the marker every speaker uses to recognise a composed line. Each segment
has a pre-rendered clip in assets/voice/ar/<fnv64(segment)>.mp3, so a composed line is played by
concatenating clips: no network, no latency, the same voice on the phone, Android Auto and CarPlay.
`plain()` is the same line as ordinary text (the fallback engines and the notification read that).

THE ROUTE MODEL (Mapbox, verified 2026-09-23 on real responses): a step's voice instructions
announce the maneuver at the END of that step, i.e. the NEXT step's maneuver; each carries
`distanceAlongGeometry` = the distance from its trigger to that maneuver.
"""
import io
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC_PATH = os.path.join(ROOT, "nav-spec", "masri_voice.json")
SPEC = json.load(io.open(SPEC_PATH, encoding="utf-8"))
N = SPEC["numbers"]
W = SPEC["words"]
TPL = SPEC["templates"]
PH = SPEC["phrases"]
PAUSE = SPEC["pauseMs"]
EN = SPEC["english"]

SEP = "\u2063"

# ── decision constants (mirrored in the Kotlin / Swift copies; the parity tool reads them) ────────
# Calibrated on the provider's own choices over 2,102 real instructions (route_check.py):
#  • a step's FIRST line, when more follow, is "keep straight for X" from 1 km out — the provider
#    does the same for 95 % of first lines ≥ 1 km and announces the maneuver for 84 % under 1 km;
#  • "then" rides ONLY on the final line (the provider: 240 of 915 finals, 0 of 750 approach lines)
#    and only when the next maneuver is ≤ 150 m on (the provider's rate falls from ~80 % under
#    100 m to 34 % at 150-250 m and 10 % beyond).
CONTINUE_MIN_M = 1000.0   # a step's first line (of several) this far out: "keep straight, about X"
FINAL_MAX_M = 250.0       # a step's LAST line this close is spoken at the maneuver, no distance
THEN_MAX_M = 150.0        # the next maneuver within this of the announced one: "وبعدها على طول…"

ROUNDABOUT_TYPES = ("roundabout", "rotary", "roundabout turn", "exit roundabout", "exit rotary")
MODIFIER_DEG = {"sharp right": 45.0, "right": 90.0, "slight right": 135.0, "straight": 180.0,
                "slight left": 225.0, "left": 270.0, "sharp left": 315.0, "uturn": 360.0}
ORDINAL_MAX = 8


# ════════════════════════════════════════════════════════════════════════════════════════════════
# NUMBERS — Cairene words, every value the app can produce
# ════════════════════════════════════════════════════════════════════════════════════════════════
NUM_MAX = 999999


def num(n):
    """0..999,999 as Egyptian words: 25 → خمسة وعشرين, 150 → مية وخمسين, 3000 → تلات تلاف.
    Clamped to that range — the voice never needs more (1,500 km is its longest distance)."""
    n = min(abs(int(n)), NUM_MAX)
    if n <= 10:
        return N["units"][n]
    if n < 20:
        return N["teens"][n - 10]
    if n < 100:
        u, t = n % 10, n // 10
        return N["tens"][t] if u == 0 else N["units"][u] + " " + N["and"] + N["tens"][t]
    if n < 1000:
        h, r = n // 100, n % 100
        return N["hundreds"][h] if r == 0 else N["hundreds"][h] + " " + N["and"] + num(r)
    k, r = n // 1000, n % 1000
    if k == 1:
        head = N["thousand"]
    elif k == 2:
        head = N["thousandDual"]
    elif k <= 10:
        head = N["short"][k] + " " + N["thousandPlural"]
    else:
        head = num(k) + " " + N["thousand"]
    return head if r == 0 else head + " " + N["and"] + num(r)


def counted(n, sing, dual, plural):
    """A counted noun, Egyptian agreement: 1 دقيقة · 2 دقيقتين · 3-10 short + plural · 11+ singular."""
    n = int(n)
    if n == 1:
        return sing
    if n == 2:
        return dual
    if 3 <= n <= 10:
        return N["short"][n] + " " + plural
    return num(n) + " " + sing


def half_up(x):
    """Round half UP — the same in all three copies (Python's round() is half-even)."""
    return int(math.floor(x + 0.5))


def dist_band(m):
    """The distance a driver is TOLD: 50 m steps to 300, 100 m to 900, then kilometres
    (½ km to 3.5, 1 km to 20, 5 km to 100, 10 km to 300, 50 km to 1000, 100 km beyond)."""
    m = max(0.0, float(m))
    if m < 75.0:
        return 50
    if m < 325.0:
        return half_up(m / 50.0) * 50
    if m < 950.0:
        return half_up(m / 100.0) * 100
    k = m / 1000.0
    if k < 3.75:
        return half_up(k * 2.0) * 500
    if k < 20.5:
        return half_up(k) * 1000
    if k < 102.5:
        return half_up(k / 5.0) * 5000
    if k < 305.0:
        return half_up(k / 10.0) * 10000
    if k < 1025.0:
        return half_up(k / 50.0) * 50000
    return half_up(k / 100.0) * 100000


def dist_words(band):
    """ميتين متر · تلتمية متر · كيلو · كيلو ونص · اتنين كيلو ونص · خمستاشر كيلو · مية وعشرين كيلو."""
    band = int(band)
    if band < 1000:
        return num(band) + " " + W["meter"]
    whole, rest = band // 1000, band % 1000
    head = W["kilo"] if whole == 1 else num(whole) + " " + W["kilo"]
    return head + (" " + W["half"] if rest == 500 else "")


def minutes_band(minutes):
    """Durations as a person says them: exact to 14 min, then 5-min steps, ¼ h to 3 h, ½ h beyond."""
    m = max(1, half_up(float(minutes)))
    if m <= 14:
        return m
    if m < 60:
        return min(60, half_up(m / 5.0) * 5)
    if m < 180:
        return half_up(m / 15.0) * 15
    return min(24 * 60, half_up(m / 30.0) * 30)


def hours_words(h):
    return counted(h, W["hour"], W["hourDual"], W["hourPlural"])


def minutes_words(band):
    """دقيقة · دقيقتين · تلات دقايق · حداشر دقيقة · ربع ساعة · تلت ساعة · نص ساعة · ساعة إلا ربع ·
    ساعة وربع · ساعتين ونص · تلات ساعات."""
    b = int(band)
    if b < 60:
        special = {15: W["quarterHour"], 20: W["thirdHour"], 30: W["halfHour"],
                   40: W["hour"] + " " + W["lessThird"], 45: W["hour"] + " " + W["lessQuarter"],
                   50: W["hour"] + " " + W["lessTen"], 55: W["hour"] + " " + W["lessFive"]}
        if b in special:
            return special[b]
        return counted(b, W["minute"], W["minuteDual"], W["minutePlural"])
    h, r = b // 60, b % 60
    if r == 0:
        return hours_words(h)
    if r == 15:
        return hours_words(h) + " " + W["andQuarter"]
    if r == 30:
        return hours_words(h) + " " + W["andHalf"]
    if r == 45:
        return hours_words(h + 1) + " " + W["lessQuarter"]
    return hours_words(h)   # unreachable with minutes_band


SPEED_MIN, SPEED_MAX = 20, 140   # every posted limit in Egypt sits inside this
SAVE_MAX_MIN = 120               # a "faster route" never claims more than two hours
ETA_MAX_MIN = 16 * 60            # the longest drive in the country, with margin


def speed_band(kmh):
    return max(SPEED_MIN, min(SPEED_MAX, half_up(float(kmh) / 5.0) * 5))


def speed_words(band):
    return num(int(band))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# SEGMENTS
# ════════════════════════════════════════════════════════════════════════════════════════════════
def pick(key, variant=0, fem=False):
    vs = PH[key]
    v = vs[int(variant) % len(vs)]
    if isinstance(v, str):
        return v
    return v["f"] if fem else v["m"]


def n_variants(key):
    return len(PH[key])


def comma(seg):
    """The segment as it is said when more follows it (a clip of its own)."""
    return seg if seg[-1:] in ("،", ".", "!", "؟") else seg + "،"


def tpl(key, **kw):
    s = TPL[key]
    for k, v in kw.items():
        s = s.replace("{" + k + "}", v)
    return s


def seg_lead(band, variant=0):
    """بعد ميتين متر، · كمان ميتين متر،  ("كمان" only for the second variant and up to a kilometre)."""
    use_more = (int(variant) % 2 == 1) and band <= 1000
    return tpl("lead.more" if use_more else "lead.after", D=dist_words(band))


def seg_cont_dist(band):
    return tpl("cont.dist", D=dist_words(band))


def seg_limit(kmh_band):
    return tpl("cam.limit", S=speed_words(kmh_band))


def seg_save(minutes, fem=False):
    return tpl("traffic.save.f" if fem else "traffic.save",
               T=minutes_words(minutes_band(min(float(minutes), SAVE_MAX_MIN))))


def seg_eta(minutes):
    return tpl("start.eta", T=minutes_words(minutes_band(min(float(minutes), ETA_MAX_MIN))))


def line(segs):
    segs = [s for s in segs if s]
    if not segs:
        return ""
    # every segment but the last is said as "more follows"
    out = [comma(s) for s in segs[:-1]] + [segs[-1]]
    return SEP + SEP.join(out)


def is_composed(text):
    return bool(text) and text[0] == SEP


def segments(text):
    return [s for s in text.split(SEP) if s]


def plain(text):
    """The composed line as ordinary text (fallback TTS engines, notifications, logs)."""
    return " ".join(segments(text)) if is_composed(text) else text


def pause_after(seg):
    """Milliseconds of silence after a segment inside a line — from its own punctuation."""
    last = seg[-1:]
    if last in (".", "!", "؟"):
        return PAUSE["stop"]
    if last == "،":
        return PAUSE["comma"]
    return PAUSE["join"]


def fnv64(text):
    """FNV-1a 64 over the UTF-8 bytes — the clip's file name. Same function in all three copies."""
    h = 0xcbf29ce484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


# ════════════════════════════════════════════════════════════════════════════════════════════════
# MANEUVERS — the provider's data → one family
# ════════════════════════════════════════════════════════════════════════════════════════════════
def norm_type(t):
    return " ".join((t or "").strip().lower().replace("_", " ").split())


def norm_modifier(m):
    m = (m or "").strip().lower().replace("-", "").replace("_", " ")
    m = " ".join(m.split())
    return "uturn" if m.replace(" ", "") == "uturn" else m


def side_of(m):
    if "right" in m:
        return "right"
    if "left" in m:
        return "left"
    return ""


def rb_direction(angle, lht):
    """The way out of a roundabout from its exit angle (degrees of circulation, 180 = straight on)."""
    a = angle
    if a < 150.0:
        d = "right"
    elif a <= 210.0:
        d = "straight"
    elif a < 330.0:
        d = "left"
    else:
        d = "uturn"
    if lht and d in ("right", "left"):   # a clockwise circle: the first exits are on the LEFT
        d = "left" if d == "right" else "right"
    return d


class Maneuver:
    __slots__ = ("type", "modifier", "exit", "degrees", "driving_side")

    def __init__(self, type=None, modifier=None, exit=None, degrees=None, driving_side="right"):
        self.type, self.modifier, self.exit, self.degrees, self.driving_side = type, modifier, exit, degrees, driving_side


def family(man, last_leg=True):
    """(kind, detail…): turn/sharp/bend/straight/curve/uturn/fork/onramp/offramp/merge/endofroad/
    arrive/waypoint (side) · rb (intro, exit, dir) · rbturn (intro, inner family) · rbleave (intro)."""
    t = norm_type(man.type)
    m = norm_modifier(man.modifier)
    side = side_of(m)
    lht = (man.driving_side or "right").strip().lower() == "left"
    if t in ROUNDABOUT_TYPES:
        intro = "rotary" if t in ("rotary", "exit rotary") else "roundabout"
        if t in ("exit roundabout", "exit rotary"):
            return ("rbleave", intro)
        if t == "roundabout turn":
            inner = family(Maneuver("turn", man.modifier, None, None, man.driving_side), last_leg)
            return ("rbturn", intro, inner)
        deg = man.degrees
        if deg is not None and isinstance(deg, (int, float)) and math.isfinite(deg) and 0.0 < deg <= 360.0:
            angle = float(deg)
            ok_angle = True
        else:
            mm = m
            if lht and side:
                mm = m.replace("right", "#").replace("left", "right").replace("#", "left")
            angle = MODIFIER_DEG.get(mm, 180.0)
            ok_angle = mm in MODIFIER_DEG
        ex = man.exit if (man.exit is not None and 1 <= int(man.exit) <= ORDINAL_MAX) else 0
        return ("rb", intro, int(ex), rb_direction(angle, lht) if ok_angle else "")
    if t == "arrive":
        s = side or ("straight" if m == "straight" else "none")
        return ("arrive" if last_leg else "waypoint", s)
    if m == "uturn":
        return ("uturn",)
    if t == "fork":
        return ("fork", side or "straight")
    if t == "on ramp":
        return ("onramp", side or "straight")
    if t == "off ramp":
        return ("offramp", side or "straight")
    if t == "merge":
        return ("merge", side or "straight")
    if t == "end of road":
        return ("endofroad", side) if side else ("straight",)
    if t in ("continue", "new name", "notification", "use lane", "depart") and side:
        if "sharp" in m:
            return ("sharp", side)
        if "slight" in m:
            return ("bend", side)
        return ("curve", side)
    if side and "sharp" in m:
        return ("sharp", side)
    if side and "slight" in m:
        return ("bend", side)
    if side:
        return ("turn", side)
    return ("straight",)


def action(fam, form, variant=0, fem=False):
    """The maneuver's own words as segments.

    form = "approach" | "final" — the WHOLE picture (the first line that names the maneuver);
           "remind" — a later approach line once the driver has been told: only the direction
                      («هتخش شمال»; a roundabout's way out «هتلف شمال»);
           "now"    — the line at the maneuver once told: only what is needed this moment
                      («دلوقتي يمين»; a roundabout's exit «خُد تالت مخرج»)."""
    k = fam[0]
    if k in ("turn", "sharp", "bend", "curve", "fork", "onramp", "offramp", "merge", "endofroad"):
        return [pick("act.%s.%s.%s" % (k, fam[1], form), variant, fem)]
    if k in ("straight", "uturn"):
        return [pick("act.%s.%s" % (k, form), variant, fem)]
    if k in ("arrive", "waypoint"):
        # the destination's side IS the news at the arrival: "now" is the full final line
        return [pick("act.%s.%s.%s" % (k, fam[1], "final" if form == "now" else form), variant, fem)]
    if k == "rbleave":
        if form == "remind":
            return [pick("rb.leave.%s.remind" % fam[1], variant, fem)]
        if form == "now":
            return [pick("rb.leave.now", variant, fem)]
        return [pick("rb.leave." + fam[1], variant, fem)]
    if k == "rbturn":
        if form in ("remind", "now"):
            return action(fam[2], form, variant, fem)
        return [pick("rb.intro." + fam[1], variant, fem)] + action(fam[2], "approach", variant, fem)
    if k == "rb":
        intro, ex, d = fam[1], fam[2], fam[3]
        if form == "remind":
            # the way out is what a reminder says; with no angle, the exit itself
            if d:
                return [pick("rb.remind." + d, variant, fem)]
            return [pick("rb.exit.%d" % ex, variant, fem)] if ex >= 1 else [pick("rb.dir.straight", variant, fem)]
        if form == "now":
            # at the circle: which exit (or, unnumbered, which way)
            return [pick("rb.exit.%d" % ex, variant, fem)] if ex >= 1 else [pick("rb.dir." + (d or "straight"), variant, fem)]
        out = [pick("rb.intro." + intro, variant, fem)]
        if ex >= 1:
            out.append(pick("rb.exit.%d" % ex, variant, fem))
            if ex >= 2 and d:
                out.append(pick("rb.hint." + d, variant, fem))
        else:
            out.append(pick("rb.dir." + (d or "straight"), variant, fem))
        return out
    raise ValueError(fam)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE ROUTE'S OWN ANNOUNCEMENTS
# ════════════════════════════════════════════════════════════════════════════════════════════════
CONTINUE, APPROACH, FINAL = "continue", "approach", "final"


def instruction_kind(instr_idx, instr_count, dist_m):
    if instr_idx == 0 and instr_count > 1 and dist_m >= CONTINUE_MIN_M:
        return CONTINUE
    if instr_idx == instr_count - 1 and dist_m <= FINAL_MAX_M:
        return FINAL
    return APPROACH


def then_included(kind, man, then_man, then_gap_m, last_leg=True):
    """Whether the line for `man` also announces `then_man` («وبعدها على طول، …»): a FINAL line, the
    next maneuver within THEN_MAX_M, neither an arrival nor a plain straight-on."""
    if kind != FINAL or then_man is None or then_gap_m is None or then_gap_m > THEN_MAX_M:
        return False
    if family(man, last_leg)[0] in ("arrive", "waypoint"):
        return False
    return family(then_man, last_leg)[0] not in ("arrive", "waypoint", "straight")


def compose(kind, man, dist_m, then_man=None, then_gap_m=None, variant=0, fem=False, last_leg=True, told=False):
    """One announcement. `then_man` = the maneuver after `man`, `then_gap_m` = how far after it.
    `told` = the driver has already been given this maneuver's whole picture on this route: an
    approach line says only the reminder, the line at the maneuver only what is needed now."""
    fam = family(man, last_leg)
    band = dist_band(dist_m)
    if kind == CONTINUE:
        return line([pick("cont.verb", variant, fem), seg_cont_dist(band)])
    segs = [] if kind == FINAL else [seg_lead(band, variant)]
    if told:
        segs += action(fam, "now" if kind == FINAL else "remind", variant, fem)
    else:
        segs += action(fam, "final" if kind == FINAL else "approach", variant, fem)
    if then_included(kind, man, then_man, then_gap_m, last_leg):
        segs += [pick("then", variant, fem)] + action(family(then_man, last_leg), "approach", variant, fem)
    return line(segs)


class Told:
    """What the driver has already been told on THIS route — so no maneuver's detail is said twice.

    One per speaker (the phone screen, the car). `sync(route_key)` before every line: a new route (a
    reroute, a switch to an alternative) forgets everything — the situation changed, so the first
    line about each maneuver gives the whole picture again. A maneuver is (leg index, step index)."""

    def __init__(self):
        self.route = None
        self.said = set()

    def sync(self, route_key):
        if route_key != self.route:
            self.route = route_key
            self.said = set()

    def has(self, leg, step):
        return (int(leg), int(step)) in self.said

    def mark(self, leg, step):
        self.said.add((int(leg), int(step)))


def route_line(maneuvers, step_idx, instr_idx, instr_count, dist_m, fem=False, last_leg=True, variant=None,
               told=None, leg=0, can_say=None):
    """The line for voice instruction `instr_idx` (of `instr_count`) on step `step_idx`.

    maneuvers[k] = (Maneuver at the START of step k, length of step k). The instruction announces
    maneuvers[step_idx + 1]; `dist_m` is its distanceAlongGeometry. Returns "" when the leg has no
    next maneuver (the host then keeps its fallback). With `told` (a Told, already synced to the
    route) the line is staged — the whole picture once, then reminders — and what it said is recorded.
    `can_say(line)` (the host's clip bank): a short line the bank cannot play yet is spoken as the whole
    picture — never in another voice."""
    if step_idx + 1 >= len(maneuvers):
        return ""
    man = maneuvers[step_idx + 1][0]
    then_man, gap = None, None
    if step_idx + 2 < len(maneuvers):
        then_man, gap = maneuvers[step_idx + 2][0], maneuvers[step_idx + 1][1]
    kind = instruction_kind(instr_idx, instr_count, dist_m)
    v = (step_idx + instr_idx) if variant is None else variant
    already = told is not None and told.has(leg, step_idx + 1)
    out = compose(kind, man, dist_m, then_man, gap, v, fem, last_leg, already)
    if already and can_say is not None and not can_say(out):
        out = compose(kind, man, dist_m, then_man, gap, v, fem, last_leg, False)
    if told is not None and kind != CONTINUE:
        told.mark(leg, step_idx + 1)
        if then_included(kind, man, then_man, gap, last_leg):
            told.mark(leg, step_idx + 2)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
# ENGLISH — the provider's words the first time, a brief line once told (the same stages)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def en_dist(band):
    """200 → "200 meters" · 1000 → "1 kilometer" · 1500 → "1.5 kilometers" · 12000 → "12 kilometers"."""
    band = int(band)
    if band < 1000:
        return EN["meters"].replace("{N}", str(band))
    if band == 1000:
        return EN["kilometer"]
    whole, rest = band // 1000, band % 1000
    n = str(whole) if rest == 0 else "%d.5" % whole
    return EN["kilometers"].replace("{N}", n)


def en_action(fam, form):
    """A family's English words; form = 0 (remind: follows «In 200 meters, ») | 1 (now: the whole line)."""
    k = fam[0]
    if k in ("turn", "sharp", "bend", "curve", "fork", "onramp", "offramp", "merge", "endofroad", "arrive", "waypoint"):
        return EN["act"]["%s.%s" % (k, fam[1])][form]
    if k in ("straight", "uturn"):
        return EN["act"][k][form]
    if k == "rbleave":
        return EN["act"]["rb.leave"][form]
    if k == "rbturn":
        return en_action(fam[2], form)
    if k == "rb":
        ex, d = fam[2], fam[3]
        if ex >= 1:
            return EN["act"]["rb.exit"][form].replace("{O}", EN["ordinal"][ex])
        return EN["act"]["rb.dir." + (d or "straight")][form]
    raise ValueError(fam)


def brief_en(kind, man, dist_m, then_man=None, then_gap_m=None, last_leg=True):
    """The English line for a maneuver the driver has already been told about: «In 200 meters, turn
    right.» on approach, «Turn right.» at the maneuver (plus « Then …» when the next one follows close).
    "" for a CONTINUE line — the provider's own «Continue for 2 kilometers» names no maneuver."""
    if kind == CONTINUE:
        return ""
    fam = family(man, last_leg)
    if kind == FINAL:
        out = en_action(fam, 1)
    else:
        out = EN["lead"].replace("{D}", en_dist(dist_band(dist_m))) + en_action(fam, 0) + "."
    if then_included(kind, man, then_man, then_gap_m, last_leg):
        out += EN["then"].replace("{A}", en_action(family(then_man, last_leg), 0))
    return out


def route_line_en(maneuvers, step_idx, instr_idx, instr_count, dist_m, last_leg=True, told=None, leg=0,
                  provider_then=False):
    """The English counterpart of route_line: "" = speak the provider's own announcement (the first
    line about a maneuver, a CONTINUE line, nothing to place it on); else the brief line. Records what
    was told — `provider_then` = the provider's own line carries a «Then …» clause."""
    if step_idx + 1 >= len(maneuvers):
        return ""
    man = maneuvers[step_idx + 1][0]
    then_man, gap = None, None
    if step_idx + 2 < len(maneuvers):
        then_man, gap = maneuvers[step_idx + 2][0], maneuvers[step_idx + 1][1]
    kind = instruction_kind(instr_idx, instr_count, dist_m)
    if told is None or kind == CONTINUE:
        return ""
    out = brief_en(kind, man, dist_m, then_man, gap, last_leg) if told.has(leg, step_idx + 1) else ""
    told.mark(leg, step_idx + 1)
    if then_included(kind, man, then_man, gap, last_leg) if out else provider_then:
        told.mark(leg, step_idx + 2)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
# THE APP'S OWN EVENTS
# ════════════════════════════════════════════════════════════════════════════════════════════════
def ev_bridge_take(tunnel, side, dist_m=None, pre_turn="", variant=0, fem=False):
    """side: "right" | "left" | "" ; pre_turn: "right" | "left" | "" (a turn first, then the deck)."""
    obj = "tunnel" if tunnel else "bridge"
    segs = [seg_lead(dist_band(dist_m), variant)] if dist_m is not None else []
    if pre_turn:
        segs.append(pick("ev.bridge.pre." + pre_turn, variant, fem))
    segs.append(pick("ev.bridge.take.%s.%s" % (obj, side or "none"), variant, fem))
    return line(segs)


def ev_bridge_dont(tunnel, side, dist_m=None, variant=0, fem=False):
    obj = "tunnel" if tunnel else "bridge"
    segs = [seg_lead(dist_band(dist_m), variant)] if dist_m is not None else []
    segs.append(pick("ev.bridge.dont.%s.%s" % (obj, side or "ahead"), variant, fem))
    return line(segs)


def ev_bridge_optional(tunnel, variant=0, fem=False):
    return line([pick("ev.bridge.optional." + ("tunnel" if tunnel else "bridge"), variant, fem)])


def ev_fork(side, dist_m=None, variant=0, fem=False, told=False):
    """ForkSense's two calls. `told` = its early call was spoken: the final call says only «دلوقتي، خليك …»."""
    s = side if side in ("right", "left") else "straight"
    if dist_m is None or dist_m <= FINAL_MAX_M:
        return line([pick("act.fork.%s.%s" % (s, "now" if told else "final"), variant, fem)])
    return line([seg_lead(dist_band(dist_m), variant), pick("act.fork.%s.approach" % s, variant, fem)])


def fork_en(side, dist_m=None, told=False):
    """ForkSense's final call in English once its early call was spoken; "" = keep ForkSense's own line."""
    if not told or not (dist_m is None or dist_m <= FINAL_MAX_M):
        return ""
    s = side if side in ("right", "left") else "straight"
    return EN["act"]["fork." + s][1]


CAMERA_KINDS = ("speed", "mobile", "redlight", "average")


def ev_camera(kind, dist_m, limit_kmh=None, variant=0, fem=False):
    k = kind if kind in CAMERA_KINDS else "speed"
    segs = [seg_lead(dist_band(dist_m), variant), pick("ev.cam." + k, variant, fem)]
    if k == "redlight":
        segs.append(pick("ev.cam.redlight.tail", variant, fem))
    elif limit_kmh is not None and limit_kmh > 0:
        segs.append(seg_limit(speed_band(limit_kmh)))
    else:
        segs.append(pick("ev.cam.nolimit", variant, fem))
    return line(segs)


REPORT_KINDS = ("camera", "accident", "police", "hazard", "traffic", "closed", "incident")


def ev_report(kind, dist_m, limit_kmh=None, variant=0, fem=False):
    k = kind if kind in REPORT_KINDS else "incident"
    segs = [seg_lead(dist_band(dist_m), variant), pick("ev.report." + k, variant, fem)]
    if k == "camera":
        segs.append(seg_limit(speed_band(limit_kmh)) if (limit_kmh is not None and limit_kmh > 0)
                    else pick("ev.cam.nolimit", variant, fem))
    return line(segs)


def ev_simple(key, variant=0, fem=False):
    """wrongway · offroute · offroute.missed · offline.coverage_exit · gps.lost · traffic.switched ·
    alt.switched · stop.reached · mark."""
    return line([pick("ev." + key, variant, fem)])


def ev_faster_route(save_min, variant=0, fem=False):
    return line([pick("ev.traffic.faster.f" if fem else "ev.traffic.faster", variant, fem), seg_save(save_min, fem)])


TRAFFIC_KINDS = ("offer.faster", "offer.jam", "offer.stuck", "offer.closed", "jam", "stuck", "closed",
                 "accident", "incident")


def seg_delay(minutes, fem=False):
    return tpl("traffic.delay.f" if fem else "traffic.delay",
               T=minutes_words(minutes_band(min(float(minutes), SAVE_MAX_MIN))))


def ev_traffic(kind, dist_m=None, minutes=None, variant=0, fem=False):
    """TrafficAdvisor's lines (docs/traffic-alerts-2026-10.md). offer.*: minutes = what it saves (none for
    a closure) and a question; jam: dist_m to the jam, minutes = its delay; stuck: dist_m = what is left of
    it; closed / accident / incident: dist_m to it."""
    k = kind if kind in TRAFFIC_KINDS else "jam"
    faster = "ev.traffic.faster.f" if fem else "ev.traffic.faster"
    ask = pick("ev.traffic.ask", variant, fem)
    lead = [seg_lead(dist_band(dist_m), variant)] if dist_m is not None else []
    if k == "offer.faster":
        return line([pick(faster, 0, fem), seg_save(minutes if minutes is not None else 1.0, fem), ask])
    if k == "offer.jam":
        return line([pick(faster, 1, fem), seg_save(minutes if minutes is not None else 1.0, fem), ask])
    if k == "offer.stuck":
        return line([pick("ev.traffic.stuck", variant, fem), pick(faster, 0, fem),
                     seg_save(minutes if minutes is not None else 1.0, fem), ask])
    if k == "offer.closed":
        return line([pick("ev.traffic.closed", variant, fem), pick("ev.traffic.other", variant, fem), ask])
    if k == "jam":
        segs = lead + [pick("ev.traffic.jam", variant, fem)]
        segs.append(seg_delay(minutes, fem) if minutes is not None else pick("ev.report.hazard", variant, fem))
        return line(segs)
    if k == "stuck":
        segs = [pick("ev.traffic.stuck", variant, fem)]
        if dist_m is not None:
            segs.append(tpl("traffic.left", D=dist_words(dist_band(dist_m))))
        elif minutes is not None:
            segs.append(seg_delay(minutes, fem))
        else:
            segs.append(pick("ev.report.hazard", variant, fem))
        return line(segs)
    if k == "closed":
        return line(lead + [pick("ev.traffic.closed.ahead", variant, fem)])
    if k == "accident":
        return line(lead + [pick("ev.report.accident", variant, fem)])
    return line(lead + [pick("ev.report.hazard", variant, fem)])


def ev_arrival(side, last_leg=True, variant=0, fem=False, extra_key=None):
    s = side if side in ("right", "left", "straight") else "none"
    segs = [pick("act.%s.%s.final" % ("arrive" if last_leg else "waypoint", s), variant, fem)]
    if extra_key:
        segs.append(pick(extra_key, variant, fem))
    return line(segs)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# QUIET MOMENTS — the personality. Never at a maneuver, a camera, a bridge or a missed turn.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def start_line(hour, weekday, hijri_month, hijri_day, trip_min=None, variant=0, fem=False, personality=True):
    """weekday: 0 = Monday … 4 = Friday (ISO-1). Hijri from the device's Umm al-Qura calendar.
    Personality off: the plain time-of-day greeting and nothing after it (the app's old rule —
    صباح الخير until noon, مساء الخير after; مساء النور is the REPLY, never the opening)."""
    if not personality:
        return line([pick("greet.morning" if 5 <= hour < 12 else "greet.evening", variant, fem)])
    if hijri_month == 9:
        greet = "qm.start.ramadan"
    elif (hijri_month == 10 and hijri_day <= 3) or (hijri_month == 12 and 10 <= hijri_day <= 13):
        greet = "qm.start.eid"
    elif 0 <= hour < 5:
        greet = "qm.start.latenight"
    elif weekday == 4 and 5 <= hour < 12:
        greet = "qm.start.friday"
    elif 5 <= hour < 12:
        greet = "qm.start.morning"
    elif 12 <= hour < 17:
        greet = "qm.start.afternoon"
    else:
        greet = "qm.start.evening"
    segs = [pick(greet, variant, fem)]
    if greet != "qm.start.latenight":
        segs.append(pick("qm.start.go", variant, fem))
    if trip_min is not None and trip_min >= 3:
        segs.append(seg_eta(trip_min))
    elif len(segs) > 1:
        segs[-1] = segs[-1].rstrip("،")
    return line(segs)


def arrival_extra_key(hour, trip_min):
    if trip_min is not None and trip_min >= 90:
        return "qm.arrive.long"
    if hour >= 22 or hour < 5:
        return "qm.arrive.night"
    return "qm.arrive"


def quiet_line(category, variant=0, fem=False):
    """category: long · jam · iftar."""
    return line([pick("qm." + category, variant, fem)])
