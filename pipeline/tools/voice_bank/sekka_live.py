# -*- coding: utf-8 -*-
"""Sekka's LIVE voice on Gemini Despina — measured at ZERO extra requests (the owner, 2026-10-17).

The audition's one-shot Windows task ("RouteMind Sekka Despina test", 2026-10-18 07:01 UTC) would have spent up to 3
of the bank's 10 free daily TTS requests; it is deleted. Instead the six Arabic Sekka replies (audition.SEKKA_REPLIES,
the audition's text: pronounce.say, the batch prompt) ride at the END of the first regular Arabic bank request of the
next daily run (build.Build.tts_round `extras`): the same model, voice and prompt the audition's "(b) Gemini Despina"
row used, cut by the same aligned cut (or the transcript cut). What it measures:

  • quality — each reply's piece read back by the Azure STT (ar-EG; the words, the masculine addressee) and heard by
    the blind phone listener (ج / ق / ث / ذ, garbled consonants) — the bank's own quota-free judges;
  • time to first audio — the request is STREAMED (gemini_tts), so its first audio chunk's arrival is measured
    (a live Sekka sentence is one such request; a long request does not delay its first chunk).

    python -m tools.voice_bank.sekka_live status          # what is measured (no network)
    python -m tools.voice_bank.sekka_live import          # desk: the work release's sekka/ → the audition page's
                                                          # "Sekka (live)" tab (takes + verdicts), page rebuilt
State: $VOICE_BANK_OUT/sekka/ (sekka.json + one WAV per reply) — carried between Actions runs by work.py.
"""
import argparse
import datetime
import io
import json
import os
import sys

from tools.voice_audition import audition as A

OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
DIR = os.path.join(OUT, "sekka")
STATE = os.path.join(DIR, "sekka.json")
VOICE = A.SEKKA_GEMINI_VOICE


def lines():
    return [ln for ln in A.sekka_lines() if ln["lang"] == "ar"]


def load():
    return json.load(io.open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}


def save(st):
    os.makedirs(DIR, exist_ok=True)
    tmp = STATE + ".tmp"
    io.open(tmp, "w", encoding="utf-8").write(json.dumps(st, ensure_ascii=False, indent=1))
    os.replace(tmp, STATE)


def pending():
    """True while no bank request has carried the replies yet."""
    return not load().get("pieces")


def extras():
    """[(id, the reply as written, the text the voice reads)] — the audition's own text for each reply."""
    return [(ln["id"], ln["plain"], A.say_text(ln)) for ln in lines()]


def record(res):
    """build.make's result for the request that carried the replies → the pieces + the request's numbers."""
    st = load()
    os.makedirs(DIR, exist_ok=True)
    st["request"] = {k: res["req"].get(k) for k in ("at", "n", "extras", "audio_s", "latency_s", "first_audio_s",
                                                    "tokens_out", "cut_ok", "recovered", "key")}
    st["meta"] = {k: res["meta"].get(k) for k in ("first_audio_s", "latency_s", "audio_s", "usage", "key")}
    st["model"], st["voice"] = A.MAIN_MODEL, VOICE
    st["recorded"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pieces = st.setdefault("pieces", {})
    for lid, pc in res["extra_pieces"]:
        if pc is None:
            pieces[lid] = {"missing": True}
            continue
        p = os.path.join(DIR, lid + ".wav")
        open(p, "wb").write(pc)
        x, rate = A.acoustics.pcm(pc)
        pieces[lid] = {"file": lid + ".wav", "audio_s": round(len(x) / float(rate), 2), "verdict": None}
    for lid, _, _ in extras():
        pieces.setdefault(lid, {"missing": True})
    save(st)
    A.log("  Sekka live: %d replies recorded from the bank request (first audio after %s s)" % (
        sum(1 for v in pieces.values() if not v.get("missing")), st["request"].get("first_audio_s")))


def judge(need_phones=True):
    """The STT read-back + the phone listener of every recorded reply. → replies still waiting."""
    from tools.voice_bank import stt as STT
    st = load()
    if not st.get("pieces"):
        return 0
    byid = {ln["id"]: ln for ln in lines()}
    left = 0
    for lid, pc in sorted(st["pieces"].items()):
        if pc.get("missing") or pc.get("verdict") is not None:
            continue
        ln = byid[lid]
        wav = A.processed(open(os.path.join(DIR, pc["file"]), "rb").read())
        ph = A.phones_verdict(ln["plain"], wav) if need_phones else None
        if need_phones and ph is None:
            left += 1
            continue
        try:
            v = STT.verdict(wav, "ar")
        except (STT.Exhausted, STT.SttError):
            left += 1
            continue
        chk = A.text_check(ln, v["transcript"])
        pc["verdict"] = {"transcript": v["transcript"], "display": v.get("display"), "check": chk, "phones": ph,
                         "pass": bool(chk["text_ok"] and not (ph or {}).get("hard")),
                         "_judge_model": v["_judge_model"]}
    save(st)
    return left


def summary():
    st = load()
    ps = st.get("pieces") or {}
    got = [p for p in ps.values() if not p.get("missing")]
    judged = [p for p in got if p.get("verdict")]
    return {"recorded": st.get("recorded"), "request": st.get("request"), "replies": len(lines()),
            "pieces": len(got), "judged": len(judged),
            "words_right": sum(1 for p in judged if p["verdict"]["check"]["text_ok"]),
            "phone_flags": [f for p in judged for f in ((p["verdict"].get("phones") or {}).get("hard") or [])],
            "passed": sum(1 for p in judged if p["verdict"]["pass"])}


def import_to_audition():
    """Desk: the work release's sekka/ (pulled into $VOICE_BANK_OUT by work.py when missing) → the audition's takes
    (group "sekka-gemini", Despina, MAIN_MODEL — the "(b) Gemini Despina" row) with the STT verdicts, page rebuilt."""
    if not os.path.exists(STATE):
        import tempfile
        from tools.voice_bank import work as W
        with tempfile.TemporaryDirectory() as d:
            W.pull(d)
            src = os.path.join(d, "sekka")
            if not os.path.isdir(src):
                raise SystemExit("no Sekka measurement in the work release yet (the next daily run makes it)")
            import shutil
            shutil.copytree(src, DIR, dirs_exist_ok=True)
    st = load()
    if not st.get("pieces"):
        raise SystemExit("no Sekka measurement yet")
    A.LIM = A.Limits()
    os.makedirs(A.CACHE, exist_ok=True)
    R = A.Run()
    byid = {ln["id"]: ln for ln in lines()}
    key = (st.get("meta") or {}).get("key") or "bank"
    n = 0
    for i, (lid, pc) in enumerate(sorted(st["pieces"].items())):
        if pc.get("missing"):
            continue
        ln = byid[lid]
        wav = open(os.path.join(DIR, pc["file"]), "rb").read()
        meta = dict(st.get("meta") or {}, model=A.MAIN_MODEL, voice=VOICE, take=1, provider="gemini",
                    kind="bank request (extras)", batched=(st.get("request") or {}).get("n"),
                    audio_s=pc.get("audio_s"), bank_request=True)
        rec = R.record(ln, VOICE, A.MAIN_MODEL, "sekka-gemini", 1, A.say_text(ln), key, wav, meta,
                       sub="_bank%d" % i)
        v = pc.get("verdict")
        if v:
            rec["judge"] = {"transcript": v["transcript"], "_judge_model": v["_judge_model"], "_batch": 1,
                            "notes": "Azure STT read-back (the bank's quota-free judge) — no accent scores"}
            rec["check"] = v["check"]
            rec["pass"] = v["pass"]
            rec["phones"] = v.get("phones") or rec.get("phones")
        n += 1
    R.state["sekka_gemini_live"] = summary()
    R.state.pop("sekka_gemini_error", None)
    R.save()
    A.write_page(R)
    print("imported %d replies into the audition page (%s)" % (n, os.path.join(A.OUT, "index.html")))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "import"])
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
    else:
        import_to_audition()


if __name__ == "__main__":
    main()
