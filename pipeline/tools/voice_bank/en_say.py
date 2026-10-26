# -*- coding: utf-8 -*-
"""English SPOKEN-FORM overrides for the Azure bank — the English twin of the Arabic SAY table (pronounce.py).

A segment's NAME is fnv64 of the segment text the app splits («onto 30th June Axis,»); that text, and so the key, never
changes. What the VOICE is asked to say can: Phoebe reads "30th June" as a date ("June thirtieth") and "1 kilometer"
as "in kilometer", and swallows or garbles some transliterated names. `EN_SAY` maps a segment to up to
`build.MAX_TAKES` spoken variants, BEST FIRST — take n renders variant n (a segment with no entry is rendered exactly
as before, and the bank's `prompt_id` does not move: only these segments differ).

A variant is {"ssml": <SSML body, already markup — plain words are fine>, "plain": <the words as a listener hears them,
what the judge compares the read-back with>, "rate": <optional prosody rate>}. `plain` defaults to the SSML body without
its tags. The judge (build.read_back_stt) compares the recogniser's transcript with `plain`, with the same rules as
for every road segment: the type word heard, the name by consonant skeleton.

The lexical rule for a name that is DATED in the street's own name (30 June, 6 October): Cairo says them as the name,
"Thirtieth of June Axis", "Sixth of October Bridge" — never as the date the digits suggest.

`override_id(seg)` fingerprints a segment's variants: build.Build.apply_en_say archives the failed takes of a segment
whose id changed and renders it afresh (a changed table is re-judged, never silently mixed with the old takes).
"""
import hashlib
import json
import re


def _punct(table):
    """Every entry for the segment written with a comma AND with a full stop (the app splits both forms)."""
    out = {}
    for seg, vs in table.items():
        for p in (",", "."):
            out[seg + p] = [dict(v, ssml=v["ssml"] + p, **({"plain": v["plain"] + p} if v.get("plain") else {}))
                            for v in vs]
    return out


_BRK = '<break time="%dms"/>'

# The 2026-10-18 variant study (docs/voice-bank-publish-2026-10.md §8.1): 3-5 variants of each of the 8 spoken forms
# were rendered and read back; the first variant of each is the study's best, the rest the next-best (the retakes).
EN_SAY = {
    # "1 kilometer" — the digit made Phoebe say "in kilometer"; the word is read ("one" heard on every variant)
    "In 1 kilometer,": [{"ssml": "In one kilometer,"},
                        {"ssml": "In " + _BRK % 90 + "one kilometer,"},
                        {"ssml": 'In <emphasis level="moderate">one</emphasis> kilometer,'}],
}
EN_SAY.update(_punct({
    "1 kilometer": [{"ssml": "One kilometer"},
                    {"ssml": _BRK % 80 + "One kilometer"},
                    {"ssml": '<emphasis level="moderate">One</emphasis> kilometer'}],
    # roads named after a date are spoken as the NAME, never as the date their digits suggest
    "onto 30th June Axis": [{"ssml": "onto Thirtieth of June Axis"},
                            {"ssml": "onto the Thirtieth of June Axis"},
                            {"ssml": "onto Thirtieth of June " + _BRK % 60 + "Axis"}],
    "onto 6th October Bridge": [{"ssml": "onto Sixth of October Bridge"},
                                {"ssml": "onto the Sixth of October Bridge"},
                                {"ssml": "onto Sixth of October " + _BRK % 60 + "Bridge"}],
    # المخيم الدائم — "El Mokhayam El Daaem" (Phoebe garbled «Al Mokhim Al Da'em» on every take)
    "onto Al Mokhim Al Da'em Street": [
        {"ssml": "onto El Mokhayam El Daaem Street"},
        {"ssml": "onto " + _BRK % 100 + "El Mokhayam El Daaem " + _BRK % 100 + "Street"},
        {"ssml": "onto El Mo-khay-yam El Da-em Street"}],
    # جسر السويس — "Gisr El Suez": Egyptian g; «Gesr» read as "jasrel" (the Arabic article is said)
    "onto Gesr Suez Axis": [
        {"ssml": "onto Gisser El Suez Axis"},
        {"ssml": "onto " + _BRK % 100 + "Gisr El Suez " + _BRK % 80 + "Axis"},
        {"ssml": "onto Jisr El Suez Axis"}],
}))
# the recogniser glued "axis" to the name (al ganzoriakis): a light break before the type word
EN_SAY["onto Al Doctor Kamal Al Ganzori Axis,"] = [
    {"ssml": "onto Al Doctor Kamal Al Ganzori " + _BRK % 120 + "Axis,"},
    {"ssml": "onto El Doctor Kamal El Ganzouri " + _BRK % 120 + "Axis,"},
    {"ssml": "onto Doctor Kamal El Ganzouri " + _BRK % 120 + "Axis,"}]

# REDO — published clips that passed the bank's judge but that NO listener of the full-bank verification hears right
# (tools/voice_bank/verify_bank.py --final, 2026-10-26: Whisper base + small, Azure's cached read-back, Gemini 3.5 —
# docs/english-voice-2026-10.md §4). Their variants below were picked on the desk the same day (the voice rendered,
# both Whisper models hearing the words); a REDO segment is rendered afresh EVEN THOUGH it has a passing take, and that
# take stays the published one until a new take passes (build.Build.apply_en_say / published_take).
_BEAR = '<phoneme alphabet="ipa" ph="bɛɹ">Bear</phoneme>'
REDO = {
    "bear left": "heard «they're left» by every listener",
    "bear right.": "heard «they're right» by every listener",
    "Bear right.": "heard «they're right» by every listener",
    "Okay, switched to the faster route.": "heard «switch to» — the -ed lost in «switched to»",
    "Then take the sixth exit.": "heard «sixth section» / «six sexit» — «exit» lost",
    "onto Al Dekhela Desert Street.": "heard «until dicla» / «untold liquid treats»",
    "onto El-Dabaa Axis,": "heard «the other axis» / «on 12 axis»",
}
EN_SAY.update({
    "bear left": [
        {"ssml": '<prosody rate="-12%">Bear</prosody> left', "plain": "bear left"},
        {"ssml": _BEAR + " left", "plain": "bear left"},
        {"ssml": '<prosody rate="-12%">' + _BEAR + "</prosody> left", "plain": "bear left"}],
    "Okay, switched to the faster route.": [
        {"ssml": "Okay, switched " + _BRK % 80 + "to the faster route."},
        {"ssml": "Okay, switched " + _BRK % 110 + "to the faster route."},
        {"ssml": "Okay, switched " + _BRK % 60 + "to the faster route."}],
    "Then take the sixth exit.": [
        {"ssml": 'Then take the <prosody rate="-10%">sixth exit</prosody>.', "plain": "Then take the sixth exit."},
        {"ssml": "Then take the sixth " + _BRK % 80 + "exit."},
        {"ssml": 'Then take the <prosody rate="-16%">sixth exit</prosody>.', "plain": "Then take the sixth exit."}],
    "onto Al Dekhela Desert Street.": [
        {"ssml": "onto " + _BRK % 100 + "El Dekheila Desert Street."},
        {"ssml": "onto " + _BRK % 100 + "El De-khay-la Desert Street."},
        {"ssml": "onto " + _BRK % 100 + "Al Dekhela " + _BRK % 60 + "Desert Street."}],
    "onto El-Dabaa Axis,": [
        {"ssml": "onto " + _BRK % 100 + "El Dab-ah " + _BRK % 80 + "Axis,"},
        {"ssml": "onto " + _BRK % 100 + "Ed Dabaa " + _BRK % 120 + "Axis,"},
        {"ssml": "onto " + _BRK % 100 + "El Dabaa " + _BRK % 120 + "Axis,"}],
})
for _seg in ("bear right.", "Bear right."):
    EN_SAY[_seg] = [
        {"ssml": '<prosody rate="-12%">' + _BEAR + "</prosody> right.", "plain": "bear right."},
        {"ssml": '<prosody rate="-12%">Bear</prosody> right.', "plain": "bear right."},
        {"ssml": _BEAR + " right.", "plain": "bear right."}]

_TAG = re.compile(r"<[^>]+>")


def variants(seg):
    return EN_SAY.get(seg) or []


def variant(seg, take):
    """The spoken variant for take number `take` (1-based) or None (the segment is rendered as written)."""
    v = variants(seg)
    return v[(take - 1) % len(v)] if v else None


def plain_of(v):
    """The words a listener hears for a variant: its `plain`, else the SSML body without tags."""
    if v.get("plain"):
        return v["plain"]
    return " ".join(_TAG.sub(" ", v["ssml"]).split())


def override_id(seg):
    v = variants(seg)
    if not v:
        return None
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
