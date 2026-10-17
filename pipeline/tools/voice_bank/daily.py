# -*- coding: utf-8 -*-
"""The navigation voice bank — the DAILY run, detached (the Windows task "RouteMind voice bank" that
tools/voice_bank/schedule_task.ps1 registers; docs/voice-bank-publish-2026-10.md).

    pythonw -m tools.voice_bank.daily --log G:/grad_all/.p5work/voice_bank/daily.log      # what the task runs
    python -m tools.voice_bank.daily [--lang ar,en] [--no-publish] [--no-phones] [--max-requests N]
    python -m tools.voice_bank.daily --status                                             # no network

ONE RUN, in this order — every step resumable, nothing asked twice, each stops cleanly on its quota:
  1. judge   — Arabic clips still without a verdict (yesterday's), most important first
  2. render  — ARABIC on the free Gemini TTS (gemini-3.8-flash-tts / Despina, 10 requests a day, 25 segments each):
               first takes and batched retakes in one queue, composer segments before road names, most spoken
               first; stops on the first 429 / RESOURCE_EXHAUSTED (the day's quota) and saves where it stopped
  3. judge   — the Arabic clips just made (the blind judge within ITS free quota + the blind phone listener)
  4. render  — ENGLISH on Azure F0 (en-US-PhoebeMultilingualNeural, one segment a request, paced 20 a minute): every
               segment not rendered yet, inside the month's desk ceiling (build.BANK_MONTH_CEILING) — the rest of
               the F0 month is Sekka's
  5. judge   — English clips without a verdict (several days of the free judge: the same daily run picks them up)
  6. settle  — a clip that failed every take it can have waits for a human listen (never published)
  7. publish — the new QA-passed clips of each language → ONE pack → uploaded, then manifest.json LAST
               (tools/voice_bank/publish.py; the contract is there)
A step that fails for any other reason (the network, a bug) is retried twice, 10 minutes apart, then the run moves on
to the next step (a failed render never blocks the publish of what already passed). Exit 0 = every step ended
cleanly (done or out of quota); 1 = a step failed after its retries.

ONE KEY, ONE PROJECT: the Gemini key is read at runtime from C:/Users/ibrah/routemind-secrets/gemini_key.txt
(audition.KEY_FILE) — never a second key or project to multiply the free quota. Azure: azure.py's desk files.
Work files: $VOICE_BANK_OUT (G:/grad_all/.p5work/voice_bank); audio never enters the repo.
"""
import argparse
import datetime
import io
import json
import os
import subprocess
import sys
import time
import traceback

OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
STATUS = os.path.join(OUT, "daily_status.json")
LOCK = os.path.join(OUT, "daily.lock")
RETRIES = 2
RETRY_WAIT_S = 600


def _no_console_windows():
    """Under pythonw every ffmpeg / gh child would open a console window on the desktop: none."""
    if os.name != "nt":
        return
    base = subprocess.Popen

    class _Quiet(base):
        def __init__(self, *a, **k):
            k["creationflags"] = k.get("creationflags", 0) | 0x08000000      # CREATE_NO_WINDOW
            super().__init__(*a, **k)

    subprocess.Popen = _Quiet


def _lock():
    """An exclusive lock on LOCK for the life of this process (released by the OS if it dies)."""
    os.makedirs(OUT, exist_ok=True)
    fh = open(LOCK, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def log(*a):
    from tools.voice_audition import audition as A
    A.log("[daily %s]" % datetime.datetime.now().strftime("%H:%M:%S"), *a)


def step(rec, name, fn):
    """Run one step with retries (an exception, or a round that stopped on "error: …" — the network, a 5xx storm);
    record its outcome in `rec`. A quota stop is an outcome, never retried."""
    for attempt in range(RETRIES + 1):
        t0 = time.time()
        try:
            res = fn()
            if isinstance(res, str) and res.startswith("error") and attempt < RETRIES:
                log("%s → %s — again in %d min" % (name, res, RETRY_WAIT_S // 60))
                time.sleep(RETRY_WAIT_S)
                continue
            rec.append({"step": name, "result": res, "s": round(time.time() - t0)})
            log("%s → %s" % (name, res))
            return res
        except Exception as e:
            from tools.voice_audition import audition as A
            log("%s FAILED (attempt %d): %s" % (name, attempt + 1, A.scrub(traceback.format_exc())[-1500:]))
            if attempt < RETRIES:
                time.sleep(RETRY_WAIT_S)
            else:
                rec.append({"step": name, "result": "FAILED: %s" % A.scrub(e)[:200], "s": round(time.time() - t0)})
                return None


def progress(order, lines):
    """Per language: segment statuses, what is published, and the corpus lines the published bank plays whole."""
    from tools.voice_bank import banks as BK
    from tools.voice_bank import publish as P
    from tools.masri_voice import masri_voice as mv
    pub = P.load_state()
    out = {}
    for lang in BK.BANKS:
        b = BK.open_build(lang)
        st = {}
        for x in order:
            if x["lang"] != lang:
                continue
            r = b.s["segs"].get(lang + "|" + x["seg"])
            k = r["status"] if r else "pending"
            st[k] = st.get(k, 0) + 1
        have = set(pub["banks"][lang]["clips"])
        segs = [x for x in order if x["lang"] == lang]
        n_lines = [segs_ for lg, segs_ in lines if lg == lang]
        whole = sum(1 for ss in n_lines if all(mv.fnv64(s) + ".mp3" in have for s in ss))
        reqs = b.s["requests"]
        out[lang] = {"build": b.id, "statuses": st, "segments": len(segs),
                     "published": sum(1 for x in segs if mv.fnv64(x["seg"]) + ".mp3" in have),
                     "composer_published": sum(1 for x in segs if x["cat"] != "road"
                                               and mv.fnv64(x["seg"]) + ".mp3" in have),
                     "corpus_lines_playable": "%d / %d" % (whole, len(n_lines)),
                     "requests": len(reqs), "chars": sum(q.get("chars", 0) for q in reqs),
                     "batch": int(b.cfg.get("batch_ar") or 0) if lang == "ar" else 1,
                     "calibration": (b.s.get("calibration") or {}).get("decision") if lang == "ar" else None,
                     "packs": [p["file"] for p in pub["banks"][lang]["packs"]]}
        first = [r["takes"][0] for r in b.s["segs"].values() if r["lang"] == lang and r["takes"]
                 and r["takes"][0].get("verdict") is not None]
        if first:
            out[lang]["first_take_pass_rate"] = round(sum(1 for t in first if t.get("pass")) / float(len(first)), 3)
    return out


def days_left(prog, order):
    """Arabic: Gemini requests still needed at the measured first-take pass rate (the plan's 1.3 takes before one is
    measured), 10 requests a day."""
    from tools.voice_bank import build as B
    ar = prog.get("ar") or {}
    st = ar.get("statuses") or {}
    todo = st.get("pending", 0) + st.get("fail", 0) + st.get("made", 0)
    p = ar.get("first_take_pass_rate")
    # expected takes per segment with at most MAX_TAKES: 1 + (1-p) + (1-p)^2 …
    per = 1.3 if p is None else sum((1.0 - p) ** k for k in range(B.MAX_TAKES))
    batch = ar.get("batch") or B.GEMINI_BATCH
    reqs = todo * per / float(batch)
    return {"segments_left": todo, "first_take_pass_rate": p, "takes_per_segment": round(per, 2), "batch": batch,
            "requests_left": round(reqs, 1), "days_left_at_10_requests": round(reqs / 10.0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="append stdout + stderr here (the task runs pythonw: no console)")
    ap.add_argument("--lang", default="ar,en")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-phones", action="store_true", help="skip the blind phone listener (NOT for the bank)")
    ap.add_argument("--max-requests", type=int, default=None, help="render at most N requests per language (tests)")
    ap.add_argument("--en-max-chars", type=int, default=None, help="Azure characters this run (tests)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--gh", default=None)
    ap.add_argument("--status", action="store_true", help="print the progress (no network) and exit")
    a = ap.parse_args(argv)
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        sys.stdout = sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)
    elif sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8")
    _no_console_windows()
    if a.ffmpeg:
        os.environ["FFMPEG"] = a.ffmpeg
    if a.gh:
        os.environ["GH"] = a.gh
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # the phone listener's model is on the desk already
    from tools.voice_audition import audition as A
    from tools.voice_bank import banks as BK
    from tools.voice_bank import publish as P

    order, lines, trips = BK.order()
    if a.status:
        prog = progress(order, lines)
        print(json.dumps({"progress": prog, "arabic_days": days_left(prog, order)}, ensure_ascii=False, indent=1))
        return 0
    fh = _lock()
    if fh is None:
        log("another daily run holds %s — this one exits" % LOCK)
        return 0
    langs = tuple(a.lang.split(","))
    started = datetime.datetime.now(datetime.timezone.utc)
    log("== start (pid %d) — %s, inventory %d segments, phrasebook %s" % (os.getpid(), ",".join(langs), len(order),
                                                                          BK.B.phrasebook_hash()))
    A.LIM = A.Limits()
    if a.no_phones:
        A.PHONES["on"] = False
    rec = []
    builds = {}

    def open_b(lang):
        if lang not in builds:
            builds[lang] = BK.open_build(lang, no_phones=a.no_phones if lang == "ar" else True,
                                         max_chars=a.en_max_chars if lang == "en" else None)
        return builds[lang]

    if "ar" in langs:
        from tools.voice_bank import sekka_live as SK
        step(rec, "ar judge (earlier clips)", lambda: open_b("ar").judge(order=order, langs=("ar",)))
        # the first Gemini day: the batch size is MEASURED before it is adopted (build.Build.calibrate)
        step(rec, "ar calibrate (batch size)", lambda: open_b("ar").calibrate(order))

        def render_ar():
            b = open_b("ar")
            if not (b.s.get("calibration") or {}).get("decision"):
                return "waiting for the calibration"
            # Sekka's live test rides at the end of the first request (no request of its own: sekka_live.py)
            ex = SK.extras() if SK.pending() else None
            return b.tts_round(order, ("ar",), max_requests=a.max_requests, extras=ex, on_extras=SK.record)

        step(rec, "ar render (Gemini)", render_ar)
        step(rec, "ar judge (new clips)", lambda: open_b("ar").judge(order=order, langs=("ar",)))
        step(rec, "sekka live (judge)", lambda: "%d replies wait" % SK.judge(need_phones=not a.no_phones))
        step(rec, "ar settle", lambda: "%d wait for a human listen" % open_b("ar").settle())
    if "en" in langs:
        # Azure has no daily request quota and the STT judge none either: render → judge → re-render the failed
        # clips (build.EN_RETAKE_RATES) → judge, until nothing is left to retake — all in this run
        from tools.voice_bank import build as B
        for rnd in range(1 + len(B.EN_RETAKE_RATES)):
            tag = "" if rnd == 0 else " (retakes %d)" % rnd
            step(rec, "en render (Azure)" + tag,
                 lambda: open_b("en").tts_round(order, ("en",), max_requests=a.max_requests))
            res = step(rec, "en judge" + tag, lambda: open_b("en").judge(order=order, langs=("en",)))
            if not (isinstance(res, str) and res == "done") or not any(
                    t > 1 for _, t in open_b("en").queue(order, ("en",))):
                break
        step(rec, "en settle", lambda: "%d wait for a human listen" % open_b("en").settle())
    if not a.no_publish:
        step(rec, "publish", lambda: P.run(langs=langs, order=order))
    prog = None
    try:
        prog = progress(order, lines)
    except Exception as e:
        log("progress: %s" % A.scrub(e)[:200])
    summary = {"started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "ended": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "steps": rec, "progress": prog, "arabic_days": days_left(prog, order) if prog else None,
               "manifest": P.BASE_URL + "manifest.json"}
    hist = []
    if os.path.exists(STATUS):
        try:
            hist = json.load(io.open(STATUS, encoding="utf-8")).get("runs", [])
        except ValueError:
            hist = []
    io.open(STATUS, "w", encoding="utf-8").write(json.dumps({"last": summary, "runs": (hist + [summary])[-60:]},
                                                            ensure_ascii=False, indent=1))
    log("== end: %s" % json.dumps({"steps": [(s["step"], s["result"]) for s in rec],
                                   "published": {k: v.get("published") for k, v in (prog or {}).items()}},
                                  ensure_ascii=False))
    failed = any(str(s["result"]).startswith(("FAILED", "error")) for s in rec)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
