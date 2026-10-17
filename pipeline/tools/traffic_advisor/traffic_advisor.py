# -*- coding: utf-8 -*-
"""TrafficAdvisor — when the drive speaks about traffic, and when it offers another route (2026-10-11).

REFERENCE copy. Mirrors android nav/TrafficAdvisor.kt and ios/Runner/TrafficAdvisor.swift (constants,
inventory and the scenario fixtures under tools/traffic_advisor/fixtures/ are held by
tool/traffic_advisor_parity.py). Pure: the hosts (the phone on both platforms, Android Auto's and
CarPlay's car-owned trips) hand it what they already have and act on its answer. docs/traffic-alerts-2026-10.md.

WHAT IT READS (nothing it reads costs a request):
  • the route's REFRESHED traffic (RouteTraffic runs: the congested stretch ahead, or the one the car is in),
  • the provider's incidents and closures on the route ahead (the same refresh carries them),
  • the alternatives the navigator already holds (the SDK's continuous alternatives, with its own time
    delta from the start of the primary), plus at most REQUEST_MAX_PER_TRIP routes the host asked for
    because THIS model said so (`want_request`) — never a poll,
  • where the next maneuver is, when the last one and the last spoken line were, whether a card is up.

WHAT IT SAYS — one decision per call, offers first:
  OFFER    a route that saves ≥ max(OFFER_MIN_S, OFFER_FRAC × remaining) and is at most
           max(OFFER_MAX_EXTRA_M, OFFER_MAX_EXTRA_FRAC × remaining) longer; the reason it gives is
           the jam ahead, the jam the car is in, a closure ahead (a route asked for after the closure,
           no saving required), or just "faster". Spoken once, shown with Take / Stay; silence = Stay.
  CLOSED / ACCIDENT / INCIDENT   a provider closure / major incident on the route ahead — once each.
  STUCK    the car has crawled ≥ STUCK_MIN_S inside the jam: how much of it is left — once per jam.
  JAM      a jam of ≥ JAM_MIN_DELAY_S starts within JAM_LOOKAHEAD_M — once per jam (not if the car is in it).

WHAT IT NEVER DOES:
  • speak within QUIET_MANEUVER_M / QUIET_MANEUVER_S of the next maneuver, QUIET_AFTER_MANEUVER_S after
    one, QUIET_AFTER_SPEECH_S after any line, or while another card is up (it waits; nothing queues);
  • offer a route the driver turned down, or another one unless it saves DECLINE_MARGIN_S more;
  • offer the route the driver just left (LEFT_HOLD_S), or anything within SWITCH_HOLD_S of a switch unless
    it saves twice the threshold (no flip-flop); offer more often than OFFER_COOLDOWN_S;
  • offer the route the arbiter overruled (a lane cut-through — StreetSense's decision stands);
  • speak in the first SETTLE_S of a trip, or more than INFO_MAX_PER_TRIP information lines.
"""
import math

# ── constants (IDENTICAL in TrafficAdvisor.kt / TrafficAdvisor.swift) ───────────────────────────
SETTLE_S = 120.0
OFFER_MIN_S = 240.0
OFFER_FRAC = 0.10
OFFER_MAX_EXTRA_M = 3000.0
OFFER_MAX_EXTRA_FRAC = 0.25
OFFER_COOLDOWN_S = 300.0
OFFER_TTL_S = 20.0
DECLINE_MARGIN_S = 180.0
SWITCH_HOLD_S = 600.0
LEFT_HOLD_S = 1200.0
QUIET_MANEUVER_M = 400.0
QUIET_MANEUVER_S = 25.0
QUIET_AFTER_MANEUVER_S = 8.0
QUIET_AFTER_SPEECH_S = 6.0
JAM_LOOKAHEAD_M = 4000.0
JAM_MIN_DELAY_S = 180.0
JAM_MIN_LEN_M = 300.0
JAM_SAME_M = 800.0
JAM_SAME_S = 1800.0
STUCK_KMH = 15.0
STUCK_MIN_S = 45.0
STUCK_MIN_DELAY_S = 120.0
HAZARD_LOOKAHEAD_M = 3000.0
CLOSED_LOOKAHEAD_M = 8000.0
INFO_COOLDOWN_S = 240.0
INFO_MAX_PER_TRIP = 8
REQUEST_MAX_PER_TRIP = 3
REQUEST_COOLDOWN_S = 600.0
CLOSED_REQUEST_COOLDOWN_S = 120.0
JAM_GAP_M = 200.0
LOW_MAX = 39               # RouteTraffic.K.LOW_MAX: a numeric value above it is congestion
EARTH_R = 6371000.0
DEG = math.pi / 180.0

KINDS = ("NONE", "OFFER", "CLOSED", "ACCIDENT", "INCIDENT", "STUCK", "JAM")
REASONS = ("faster", "jam", "stuck", "closed")


class Candidate(object):
    """A route the host could install. saving_s: + = faster than staying (the navigator's own delta);
    extra_m: + = longer; left_route: it is the route the driver just switched away from; overruled: the
    arbiter rejected it (a lane cut-through); requested: it came from a request THIS model asked for;
    after_closure: that request was made after the closure ahead was known."""
    __slots__ = ("id", "saving_s", "extra_m", "left_route", "overruled", "requested", "after_closure")

    def __init__(self, id, saving_s, extra_m, left_route=False, overruled=False, requested=False, after_closure=False):
        self.id, self.saving_s, self.extra_m = str(id), float(saving_s), float(extra_m)
        self.left_route, self.overruled = bool(left_route), bool(overruled)
        self.requested, self.after_closure = bool(requested), bool(after_closure)


class Hazard(object):
    """A provider incident on the route ahead. kind: "closed" | "accident" | "incident"."""
    __slots__ = ("key", "kind", "dist_m")

    def __init__(self, key, kind, dist_m):
        self.key, self.kind, self.dist_m = str(key), str(kind), float(dist_m)


class Jam(object):
    """The first significant congested run ahead of the car, or the one it is in. lat/lng: where it
    starts (its identity across refreshes and reroutes); dist_m: to its start (0 when inside);
    length_m: what is left of it; delay_s: the extra time it costs from here."""
    __slots__ = ("lat", "lng", "dist_m", "length_m", "delay_s", "inside")

    def __init__(self, lat, lng, dist_m, length_m, delay_s, inside):
        self.lat, self.lng, self.dist_m = float(lat), float(lng), float(dist_m)
        self.length_m, self.delay_s, self.inside = float(length_m), float(delay_s), bool(inside)


class Context(object):
    __slots__ = ("t", "trip_s", "remaining_s", "remaining_m", "speed_kmh", "jam", "hazards", "candidates",
                 "worth_checking", "next_maneuver_m", "since_maneuver_s", "since_speech_s", "card_showing")

    def __init__(self, t, trip_s, remaining_s, remaining_m, speed_kmh, jam=None, hazards=(), candidates=(),
                 worth_checking=False, next_maneuver_m=1e9, since_maneuver_s=1e9, since_speech_s=1e9,
                 card_showing=False):
        self.t, self.trip_s = float(t), float(trip_s)
        self.remaining_s, self.remaining_m, self.speed_kmh = float(remaining_s), float(remaining_m), float(speed_kmh)
        self.jam = jam
        self.hazards = list(hazards)
        self.candidates = list(candidates)
        self.worth_checking = bool(worth_checking)
        self.next_maneuver_m, self.since_maneuver_s = float(next_maneuver_m), float(since_maneuver_s)
        self.since_speech_s, self.card_showing = float(since_speech_s), bool(card_showing)


class Decision(object):
    __slots__ = ("kind", "reason", "candidate", "minutes", "dist_m", "key", "why")

    def __init__(self, kind, why, reason="", candidate="", minutes=0.0, dist_m=0.0, key=""):
        self.kind, self.why, self.reason, self.candidate = kind, why, reason, candidate
        self.minutes, self.dist_m, self.key = float(minutes), float(dist_m), key

    def row(self):
        return [self.kind, self.reason, self.candidate, round(self.minutes, 2), round(self.dist_m, 1), self.key, self.why]


def hav(a_lat, a_lng, b_lat, b_lng):
    p1, p2 = a_lat * DEG, b_lat * DEG
    dp, dl = p2 - p1, (b_lng - a_lng) * DEG
    s1, s2 = math.sin(dp / 2.0), math.sin(dl / 2.0)
    h = s1 * s1 + math.cos(p1) * math.cos(p2) * s2 * s2
    return 2.0 * EARTH_R * math.asin(min(1.0, math.sqrt(h)))


def jam_ahead(rows, from_index):
    """The jam the advisor reads, from the route's REFRESHED traffic rows (RouteTraffic.segments: band,
    numeric, metres, seconds) and the car's segment: congested segments (numeric > LOW_MAX) merged across
    gaps up to JAM_GAP_M; the first one costing ≥ STUCK_MIN_DELAY_S that contains the car or starts within
    JAM_LOOKAHEAD_M. -> (start_m along the route, dist_m to it, length_m left, delay_s from here, inside)
    or None. The delay is RouteTraffic's: duration × numeric/100."""
    car = 0.0
    for i in range(0, min(max(0, from_index), len(rows))):
        car += rows[i][2]
    pos = car
    start, end, delay, gap = -1.0, 0.0, 0.0, 0.0
    for i in range(max(0, from_index), len(rows)):
        _band, num, dist, dur = rows[i]
        if start < 0.0 and pos - car > JAM_LOOKAHEAD_M:
            break
        if num is not None and num > LOW_MAX:
            if start < 0.0:
                start, delay = pos, 0.0
            gap = 0.0
            end = pos + dist
            delay += dur * (min(100, int(num)) / 100.0)
        elif start >= 0.0:
            gap += dist
            if gap > JAM_GAP_M:
                if delay >= STUCK_MIN_DELAY_S:
                    break
                start, delay, gap = -1.0, 0.0, 0.0
        pos += dist
    if start < 0.0 or delay < STUCK_MIN_DELAY_S:
        return None
    return (start, max(0.0, start - car), end - max(start, car), delay, start <= car)


def hazard_kind(itype, closed, impact):
    """A provider incident (Directions `incidents[]`: type, closed, impact) as the advisor reads it:
    "closed" | "accident" | "incident" | "" (not worth a line — congestion is the jam's, minor is noise)."""
    if closed or itype == "road_closure":
        return "closed"
    if itype == "accident":
        return "" if impact == "low" else "accident"
    if itype in ("road_hazard", "disabled_vehicle", "construction", "lane_restriction", "weather") \
            and impact in ("critical", "major"):
        return "incident"
    return ""


def offer_min_s(remaining_s):
    return max(OFFER_MIN_S, OFFER_FRAC * remaining_s)


def extra_cap_m(remaining_m):
    return max(OFFER_MAX_EXTRA_M, OFFER_MAX_EXTRA_FRAC * remaining_m)


class Advisor(object):
    """One per trip (the odometer of what was said). decide() is pure over the context and this memory;
    commit() records what the host actually presented; answer() records the driver's choice."""

    def __init__(self):
        self.last_offer_t = -1e9
        self.offered = []            # candidate ids offered (an offer is made once)
        self.declined = []           # [(id, saving_s)]
        self.last_switch_t = -1e9
        self.last_info_t = -1e9
        self.info_count = 0
        self.jams = []               # [(lat, lng, t)] announced jams (ahead or inside)
        self.hazards_said = []       # hazard keys announced
        self.requests = 0
        self.last_request_t = -1e9
        self.closed_requested = []   # closure keys a request was made for
        self.slow_since_t = -1.0     # since when the car is under STUCK_KMH (< 0: it is not)

    # ── the car's crawl, tracked across calls ───────────────────────────────────────────────────
    def observe(self, t, speed_kmh):
        if speed_kmh < STUCK_KMH:
            if self.slow_since_t < 0.0:
                self.slow_since_t = t
        else:
            self.slow_since_t = -1.0

    def _quiet(self, c):
        speed_mps = c.speed_kmh / 3.6
        if c.next_maneuver_m < max(QUIET_MANEUVER_M, speed_mps * QUIET_MANEUVER_S):
            return "quiet:maneuver-ahead"
        if c.since_maneuver_s < QUIET_AFTER_MANEUVER_S:
            return "quiet:maneuver-just-done"
        if c.since_speech_s < QUIET_AFTER_SPEECH_S:
            return "quiet:speech"
        if c.card_showing:
            return "quiet:card"
        return ""

    def _jam_known(self, j, t):
        for (lat, lng, at) in self.jams:
            if t - at <= JAM_SAME_S and hav(lat, lng, j.lat, j.lng) <= JAM_SAME_M:
                return True
        return False

    def _closure_ahead(self, c):
        for h in c.hazards:
            if h.kind == "closed" and 0.0 <= h.dist_m <= CLOSED_LOOKAHEAD_M:
                return h
        return None

    def _best(self, c, closed):
        """The candidate worth offering now, or None. closed: a closure is ahead (only a route asked
        for after it qualifies, and no saving is required)."""
        if c.t - self.last_offer_t < OFFER_COOLDOWN_S and closed is None:
            return None
        need = offer_min_s(c.remaining_s)
        if c.t - self.last_switch_t < SWITCH_HOLD_S:
            need = 2.0 * need
        best_declined = -1e9
        for (_id, s) in self.declined:
            if s > best_declined:
                best_declined = s
        if self.declined:
            need = max(need, best_declined + DECLINE_MARGIN_S)
        best = None
        for k in c.candidates:
            if k.overruled or k.id in self.offered:
                continue
            if k.left_route and c.t - self.last_switch_t < LEFT_HOLD_S:
                continue
            if any(k.id == d for (d, _s) in self.declined):
                continue
            if closed is not None:
                if not (k.requested and k.after_closure):
                    continue
            else:
                if k.saving_s < need or k.extra_m > extra_cap_m(c.remaining_m):
                    continue
            if best is None or k.saving_s > best.saving_s:
                best = k
        return best

    def decide(self, c):
        """→ the one thing to present now (or NONE with the reason it waits)."""
        self.observe(c.t, c.speed_kmh)
        if c.trip_s < SETTLE_S:
            return Decision("NONE", "settling")
        q = self._quiet(c)
        if q:
            return Decision("NONE", q)
        closed = self._closure_ahead(c)
        j = c.jam
        stuck = (j is not None and j.inside and self.slow_since_t >= 0.0
                 and c.t - self.slow_since_t >= STUCK_MIN_S and j.delay_s >= STUCK_MIN_DELAY_S)
        jam_ahead = (j is not None and not j.inside and j.dist_m <= JAM_LOOKAHEAD_M
                     and j.delay_s >= JAM_MIN_DELAY_S and j.length_m >= JAM_MIN_LEN_M)

        # 1. offers
        k = self._best(c, closed)
        if k is not None:
            if closed is not None:
                return Decision("OFFER", "offer:closed", "closed", k.id, 0.0, closed.dist_m, closed.key)
            reason = "stuck" if stuck else ("jam" if jam_ahead else "faster")
            return Decision("OFFER", "offer:" + reason, reason, k.id, k.saving_s / 60.0,
                            j.dist_m if (j is not None and reason != "faster") else 0.0)

        # 2. information — once each, spaced, capped
        if self.info_count >= INFO_MAX_PER_TRIP:
            return Decision("NONE", "info-cap")
        if closed is not None and closed.key not in self.hazards_said:
            return Decision("CLOSED", "closed-ahead", "", "", 0.0, closed.dist_m, closed.key)
        if c.t - self.last_info_t < INFO_COOLDOWN_S:
            return Decision("NONE", "info-cooldown")
        if stuck and not self._jam_known(j, c.t):
            return Decision("STUCK", "stuck", "", "", j.delay_s / 60.0, j.length_m, "")
        if jam_ahead and not self._jam_known(j, c.t):
            return Decision("JAM", "jam-ahead", "", "", j.delay_s / 60.0, j.dist_m, "")
        for h in c.hazards:
            if h.kind in ("accident", "incident") and 0.0 <= h.dist_m <= HAZARD_LOOKAHEAD_M \
                    and h.key not in self.hazards_said:
                return Decision("ACCIDENT" if h.kind == "accident" else "INCIDENT", "hazard-ahead",
                                "", "", 0.0, h.dist_m, h.key)
        return Decision("NONE", "nothing")

    def want_request(self, c, offer_pending):
        """Should the host ask for ONE route now (silent; it comes back as a `requested` candidate)?"""
        if c.trip_s < SETTLE_S or offer_pending:
            return False, "no:settling-or-pending"
        if self.requests >= REQUEST_MAX_PER_TRIP:
            return False, "no:cap"
        closed = self._closure_ahead(c)
        if closed is not None and closed.key not in self.closed_requested:
            if c.t - self.last_request_t >= CLOSED_REQUEST_COOLDOWN_S:
                return True, "yes:closed"
            return False, "no:cooldown"
        if c.t - self.last_request_t < REQUEST_COOLDOWN_S:
            return False, "no:cooldown"
        if self._best(c, None) is not None:
            return False, "no:have-candidate"
        j = c.jam
        stuck = (j is not None and j.inside and self.slow_since_t >= 0.0
                 and c.t - self.slow_since_t >= STUCK_MIN_S and j.delay_s >= STUCK_MIN_DELAY_S)
        if c.worth_checking or stuck:
            return True, "yes:" + ("stuck" if stuck else "delay-ahead")
        return False, "no:nothing-to-avoid"

    # ── what the host did ─────────────────────────────────────────────────────────────────────────
    def requested(self, c):
        self.requests += 1
        self.last_request_t = c.t
        closed = self._closure_ahead(c)
        if closed is not None:
            self.closed_requested.append(closed.key)

    def commit(self, d, c):
        """The host presented `d` (spoke it / showed it)."""
        if d.kind == "OFFER":
            self.last_offer_t = c.t
            self.offered.append(d.candidate)
            if d.reason == "closed" and d.key:
                self.hazards_said.append(d.key)
            if d.reason in ("jam", "stuck") and c.jam is not None:
                self.jams.append((c.jam.lat, c.jam.lng, c.t))
            return
        if d.kind == "NONE":
            return
        self.info_count += 1
        self.last_info_t = c.t
        if d.kind in ("CLOSED", "ACCIDENT", "INCIDENT"):
            self.hazards_said.append(d.key)
        elif d.kind in ("STUCK", "JAM") and c.jam is not None:
            self.jams.append((c.jam.lat, c.jam.lng, c.t))

    def answer(self, candidate_id, saving_s, accepted, t):
        """The driver took the offer (accepted) or turned it down — Stay, or the card ran out."""
        if accepted:
            self.last_switch_t = t
        else:
            self.declined.append((candidate_id, saving_s))

    def switched(self, t):
        """The route changed for another reason the driver chose (a chip tap, a line tap)."""
        self.last_switch_t = t


# ── the words (English; the Egyptian lines are MasriVoice.traffic) ──────────────────────────────────
def half_up(x):
    return int(math.floor(x + 0.5))


def en_minutes(m):
    n = max(1, half_up(m))
    return "1 minute" if n == 1 else "%d minutes" % n


def en_dist(m):
    if m < 950.0:
        v = max(100, half_up(m / 100.0) * 100)
        return "%d meters" % v
    half = half_up(m / 500.0)               # half-kilometres
    if half % 2 == 0:
        k = half // 2
        return "1 kilometer" if k == 1 else "%d kilometers" % k
    return "%d.5 kilometers" % (half // 2)


def english(d):
    """The English line for a decision (the app's own words; the car and the phone say the same)."""
    if d.kind == "OFFER":
        if d.reason == "closed":
            return "The road ahead is closed. There's another route. Want to take it?"
        save = en_minutes(d.minutes)
        if d.reason == "jam":
            return "Heavy traffic ahead. There's a faster route that saves about %s. Want to take it?" % save
        if d.reason == "stuck":
            return "You're in heavy traffic. A faster route saves about %s. Want to take it?" % save
        return "Found a faster route that saves about %s. Want to take it?" % save
    if d.kind == "CLOSED":
        return "The road is closed in %s." % en_dist(d.dist_m)
    if d.kind == "ACCIDENT":
        return "Accident reported in %s. Slow down." % en_dist(d.dist_m)
    if d.kind == "INCIDENT":
        return "Road incident in %s. Take care." % en_dist(d.dist_m)
    if d.kind == "STUCK":
        return "You're in heavy traffic. It continues for about %s." % en_dist(d.dist_m)
    if d.kind == "JAM":
        return "Heavy traffic in %s, about %s of delay." % (en_dist(d.dist_m), en_minutes(d.minutes))
    return ""


# ── the card (the app language; the phone and the car show the same words) ─────────────────────
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def ar_num(text):
    return "".join(AR_DIGITS[int(c)] if c.isdigit() else ("٫" if c == "." else c) for c in text)


def card_minutes(m, arabic):
    n = max(1, half_up(m))
    if not arabic:
        return "%d min" % n
    if n == 1:
        return "دقيقة"
    if n == 2:
        return "دقيقتين"
    return ar_num(str(n)) + (" دقايق" if n <= 10 else " دقيقة")


def card_distance(m, arabic):
    if m < 950.0:
        v = "%d" % max(100, half_up(m / 100.0) * 100)
        return ar_num(v) + " م" if arabic else v + " m"
    tenth = half_up(m / 100.0)
    v = "%d" % (tenth // 10) if tenth % 10 == 0 else "%d.%d" % (tenth // 10, tenth % 10)
    return ar_num(v) + " كم" if arabic else v + " km"


def card_title(d, arabic):
    if d.kind == "OFFER":
        if d.reason == "closed":
            return "الطريق قدّام مقفول" if arabic else "Road closed ahead"
        return "طريق أسرع" if arabic else "Faster route"
    return {"CLOSED": ("الطريق مقفول", "Road closed"), "ACCIDENT": ("حادثة قدّام", "Accident ahead"),
            "INCIDENT": ("مشكلة في الطريق", "Road incident ahead"), "STUCK": ("إنت في زحمة", "Heavy traffic"),
            "JAM": ("زحمة جامدة قدّام", "Heavy traffic ahead")}.get(d.kind, ("", ""))[0 if arabic else 1]


def card_detail(d, arabic):
    if d.kind == "OFFER":
        if d.reason == "closed":
            return "فيه طريق تاني" if arabic else "Another route is ready"
        return ("هيوفّر " if arabic else "Saves ") + card_minutes(d.minutes, arabic)
    if d.kind in ("CLOSED", "ACCIDENT", "INCIDENT"):
        return ("بعد " if arabic else "In ") + card_distance(d.dist_m, arabic)
    if d.kind == "STUCK":
        return ("فاضل %s · %s" if arabic else "%s left · %s") % (card_distance(d.dist_m, arabic), card_minutes(d.minutes, arabic))
    if d.kind == "JAM":
        return ("بعد %s · +%s" if arabic else "In %s · +%s") % (card_distance(d.dist_m, arabic), card_minutes(d.minutes, arabic))
    return ""


def action_take(arabic):
    return "خده" if arabic else "Take it"


def action_stay(arabic):
    return "خليني على طريقي" if arabic else "Stay"


def masri_kind(d):
    """The MasriVoice.traffic kind for a decision (the Egyptian line)."""
    if d.kind == "OFFER":
        return "offer." + d.reason
    return {"CLOSED": "closed", "ACCIDENT": "accident", "INCIDENT": "incident",
            "STUCK": "stuck", "JAM": "jam"}.get(d.kind, "")


def masri(d, variant=0, fem=False):
    """The Egyptian line for a decision (MasriVoice.traffic)."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.masri_voice import masri_voice as mv
    if d.kind == "NONE":
        return ""
    if d.kind == "OFFER":
        return mv.ev_traffic(masri_kind(d), None, None if d.reason == "closed" else d.minutes, variant, fem)
    if d.kind == "STUCK":
        return mv.ev_traffic("stuck", d.dist_m, None, variant, fem)
    if d.kind == "JAM":
        return mv.ev_traffic("jam", d.dist_m, d.minutes, variant, fem)
    return mv.ev_traffic(masri_kind(d), d.dist_m, None, variant, fem)
