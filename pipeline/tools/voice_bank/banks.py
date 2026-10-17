# -*- coding: utf-8 -*-
"""The two published navigation banks — the owner's choice after the audition (2026-10-17). ONE place that says which
voice, which settings and which build each language is; tools/voice_bank/daily.py renders and judges them,
tools/voice_bank/publish.py ships them (docs/voice-bank-publish-2026-10.md).

  ar  gemini-3.8-flash-tts, prebuilt voice Despina — EXACTLY the audition's settings for her
      (tools/voice_audition/audition.py): the director's-notes prompt `prompt_ar` (one segment) / `prompt_ar_batch`
      (K segments, one line each, "a clear one-second pause after every line"), no generationConfig beyond the voice
      (the API's default temperature — the audition sets none), the text through tools/masri_voice/pronounce.py SAY
      (a retake reads the next spelling), the aligned cut (`split_on_silence` with the letters of each line), the
      bank's processing (`processed` = build_bank.process: 25 / 40 ms lead / tail, fades, speech RMS -20 dBFS, peak
      under -1.5 dBFS) and format (MPEG-2 Layer III, 24 kHz mono 32 kbps). 25 segments a request (build.GEMINI_BATCH).
      Azure Arabic was rejected by the owner.
  en  Azure en-US-PhoebeMultilingualNeural — the audition's Phoebe settings: none (audition.AZ_EN_PICK gives her {}:
      no rate, pitch or style), one segment per request (English carries no markup to share), the same processing and
      format.

`prompt_id` names the settings a bank's clips were made with; daily.py refuses to add clips to a build whose settings
changed under it (one bank = one voice = one prompt).
"""
import hashlib
import json
import types

from tools.masri_voice import build_bank
from tools.masri_voice import pronounce
from tools.voice_audition import audition as A
from tools.voice_bank import build as B
from tools.voice_bank import inventory as INV

AR_VOICE = "Despina"
EN_VOICE = "en-US-PhoebeMultilingualNeural"
assert (EN_VOICE, {}) in A.AZ_EN_PICK, "Phoebe's audition settings changed — re-read audition.AZ_EN_PICK"

BANKS = {
    "ar": {"build": "ar-gemini-despina", "provider": "gemini", "label": "%s/%s" % (A.MAIN_MODEL, AR_VOICE)},
    "en": {"build": "en-azure-phoebe", "provider": "azure", "label": "azure/%s" % EN_VOICE},
}
FORMAT = "MPEG-2 Layer III, 24 kHz mono, 32 kbps CBR, bare frames (no ID3 / Xing)"
PROCESS = {"lead_s": build_bank.LEAD_S, "tail_s": build_bank.TAIL_S, "fade_in_s": build_bank.FADE_IN_S,
           "fade_out_s": build_bank.FADE_OUT_S, "rms_db": build_bank.TARGET_RMS_DB, "peak_db": build_bank.PEAK_CEIL_DB}


def args_for(lang, no_phones=False, judge_model=None, max_chars=None, judge=None):
    """The namespace build.Build reads, for one bank."""
    if lang == "ar":
        return types.SimpleNamespace(
            provider="gemini", build=BANKS["ar"]["build"], voice_ar=AR_VOICE, voice_en=AR_VOICE, ar_rate="",
            ar_pitch="", en_rate="", en_pitch="", en_style="", batch_ar=B.GEMINI_BATCH, batch_en=B.GEMINI_BATCH,
            sep="nl", ar_spell="say", lang="ar", first=None, max_chars=max_chars, no_phones=no_phones,
            judge_model=judge_model, again=False, judge=judge)
    return types.SimpleNamespace(
        provider="azure", build=BANKS["en"]["build"], voice_ar=A.AZ_BANK_AR, voice_en=EN_VOICE, ar_rate="",
        ar_pitch="", en_rate="", en_pitch="", en_style="", batch_ar=B.AZ_BANK_BATCH, batch_en=1, sep="nl",
        ar_spell="pausal", lang="en", first=None, max_chars=max_chars, no_phones=True, judge_model=judge_model,
        again=False, judge=judge)


def prompt_id(lang):
    """A short id of every setting that shapes a clip: model / voice, the prompt (or SSML) templates, the spellings,
    the processing and the format. Any change = another id = another build."""
    if lang == "ar":
        spec = {"model": A.MAIN_MODEL, "voice": AR_VOICE, "prompt": A.prompt_ar("{TEXT}"),
                "batch_prompt": A.prompt_ar_batch("{TEXT}", 0), "say": pronounce.SAY,
                "cut": "audition.split_on_silence(weights=letters_of)", "process": PROCESS, "format": FORMAT}
        tag = "g38f-despina"
    else:
        from tools.voice_audition import azure as AZR
        spec = {"voice": EN_VOICE, "ssml": AZR.ssml("{TEXT}", EN_VOICE, "en"), "fmt": AZR.HQ_FMT,
                "process": PROCESS, "format": FORMAT}
        tag = "az-phoebe"
    h = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return "%s-%s" % (tag, h[:10])


def open_build(lang, **kw):
    b = B.Build(args_for(lang, **kw))
    meta = b.s.setdefault("bank", {"lang": lang, "label": BANKS[lang]["label"], "prompt_id": prompt_id(lang)})
    if meta["prompt_id"] != prompt_id(lang):
        raise RuntimeError("build %s was made with settings %s; the code now says %s — start another build (a new id "
                           "in banks.BANKS) instead of mixing two settings in one bank"
                           % (b.id, meta["prompt_id"], prompt_id(lang)))
    b.save()
    return b


def order():
    """The ranked inventory in build order: every composer segment before any road name, then most spoken first."""
    o, lines, trips = INV.ranked()
    return sorted(o, key=B.prio), lines, trips
