# -*- coding: utf-8 -*-
"""The voice bank's WORK STATE between GitHub Actions runs — never in git (docs/voice-bank-publish-2026-10.md §7).

    python -m tools.voice_bank.work pull [--dir OUT]          # the work release's assets → OUT (the run's first step)
    python -m tools.voice_bank.work push [--dir OUT]          # OUT → the work release (the run's last step, always)
    python -m tools.voice_bank.work seed --dir <desk OUT>     # the first push, from the desk (creates the release)
    python -m tools.voice_bank.work status                    # what the work release holds

WHERE: a PRERELEASE tagged `work` on the public repo ibrahim212-wq/routemind-voice — not the reader's `bank` release
(that one stays exactly the app's contract: manifest.json + packs). A prerelease is never `releases/latest`.

WHAT (deterministic tars — the same files give the same bytes, so an unchanged asset is not uploaded again):
  • state.tar.gz — every build's state.json (statuses, takes, verdicts, the request log), the publisher's record
    (publish/state.json + manifest.json), the STT and Azure ledgers, the Gemini quota record (limits.json), the STT
    read-back cache, the phone listener's cache, daily_status.json, the Sekka live measurement (sekka/), and
    work.json (what each asset's sha256 is).
  • clips-<build>.tar — that build's rendered clips (MP3, the bank's format) — published or not; the next pack and a
    re-pack at publish.MAX_PACKS read them. The judge's .heard.mp3 copies are left out (made again on demand).
  • context_lead.mp3 — the bundled bank's «بعد تلتمية متر،» (assets/voice/ar), the lead every Arabic clip is heard
    behind; the app repo's audio never enters the public repo's git.
Nothing secret is in any of them (every ledger / state is scrubbed of keys when written; there is no key on disk in
OUT). `gh` authenticates with $GH_TOKEN (the workflow's GITHUB_TOKEN, contents: write) or its own login.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

from tools.voice_bank import banks as BK

REPO = "ibrahim212-wq/routemind-voice"
TAG = "work"
OUT = os.environ.get("VOICE_BANK_OUT", "G:/grad_all/.p5work/voice_bank")
GH = os.environ.get("GH") or shutil.which("gh") or "gh"
NOTES = ("Work in progress of the voice bank build (state, verdicts, rendered clips) — NOT for the app: the app reads "
         "the `bank` release only. Written by the voice-bank workflow (pipeline/tools/voice_bank/work.py).")
CLIP_EXT = (".mp3",)


def log(*a):
    print("[work]", *a, flush=True)


def gh(*args, check=True, timeout=1800):
    env = dict(os.environ, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1", NO_COLOR="1")
    r = subprocess.run([GH] + [str(x) for x in args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit("[work] gh %s failed (%d): %s" % (" ".join(str(x) for x in args[:3]), r.returncode,
                                                            (r.stderr or r.stdout or "").strip()[-400:]))
    return r


def aud_dir(out):
    return os.environ.get("VOICE_AUDITION_OUT") or os.path.join(out, "audition")


def state_members(out):
    """(archive name, path) of every state file present. Desk layout: the audition's files live in
    G:/grad_all/.p5work/voice_audition — the archive always stores them under audition/."""
    m = []
    for lang, meta in BK.BANKS.items():
        m.append(("%s/state.json" % meta["build"], os.path.join(out, meta["build"], "state.json")))
    for rel in ("publish/state.json", "publish/manifest.json", "stt_ledger.json", "daily_status.json"):
        m.append((rel, os.path.join(out, rel)))
    for sub in ("stt_cache", "sekka"):
        base = os.path.join(out, sub)
        for dp, ds, fs in os.walk(base):
            ds.sort()
            for f in sorted(fs):
                p = os.path.join(dp, f)
                m.append((os.path.relpath(p, out).replace(os.sep, "/"), p))
    aud = aud_dir(out)
    for rel in ("limits.json", "azure/ledger.json"):
        m.append(("audition/" + rel, os.path.join(aud, rel)))
    cache = os.path.join(aud, "cache")
    if os.path.isdir(cache):
        for f in sorted(os.listdir(cache)):
            if f.startswith("phones_") and f.endswith(".json"):
                m.append(("audition/cache/" + f, os.path.join(cache, f)))
    return [(a, p) for a, p in m if os.path.isfile(p)]


def clip_members(out, build):
    base = os.path.join(out, build, "clips")
    m = []
    for dp, ds, fs in os.walk(base):
        ds.sort()
        for f in sorted(fs):
            if f.endswith(CLIP_EXT) and not f.endswith(".heard.mp3"):
                p = os.path.join(dp, f)
                m.append((os.path.relpath(p, out).replace(os.sep, "/"), p))
    return m


def write_tar(path, members, gz):
    """A deterministic archive: sorted names, mtime 0, owner 0, mode 644 (gzip without a timestamp)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as t:
        for name, p in sorted(members):
            data = open(p, "rb").read()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            t.addfile(ti, io.BytesIO(data))
    data = raw.getvalue()
    if gz:
        import gzip
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=6) as g:
            g.write(data)
        data = buf.getvalue()
    open(path, "wb").write(data)
    return hashlib.sha256(data).hexdigest()


def release_exists():
    return gh("release", "view", TAG, "-R", REPO, "--json", "tagName", check=False).returncode == 0


def ensure_release():
    if not release_exists():
        gh("release", "create", TAG, "-R", REPO, "--prerelease", "--title", "voice bank — work in progress (not for "
           "the app)", "--notes", NOTES, "--latest=false")
        log("prerelease %s created" % TAG)


def remote_work_json():
    with tempfile.TemporaryDirectory() as d:
        r = gh("release", "download", TAG, "-R", REPO, "-p", "work.json", "-D", d, check=False)
        p = os.path.join(d, "work.json")
        if r.returncode != 0 or not os.path.exists(p):
            return {"assets": {}}
        return json.load(io.open(p, encoding="utf-8"))


def push(out, lead=None):
    ensure_release()
    have = remote_work_json().get("assets", {})
    stage = tempfile.mkdtemp(prefix="vb-work-")
    try:
        assets = {}
        files = []
        for build in sorted(m["build"] for m in BK.BANKS.values()):
            name = "clips-%s.tar" % build
            p = os.path.join(stage, name)
            assets[name] = write_tar(p, clip_members(out, build), gz=False)
            if have.get(name) != assets[name]:
                files.append(p)
        if lead:
            p = os.path.join(stage, "context_lead.mp3")
            shutil.copyfile(lead, p)
            assets["context_lead.mp3"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if have.get("context_lead.mp3") != assets["context_lead.mp3"]:
                files.append(p)
        elif "context_lead.mp3" in have:
            assets["context_lead.mp3"] = have["context_lead.mp3"]
        p = os.path.join(stage, "state.tar.gz")
        assets["state.tar.gz"] = write_tar(p, state_members(out), gz=True)
        files.append(p)                                    # the state goes every time (it carries the run's record)
        wj = os.path.join(stage, "work.json")
        import datetime
        io.open(wj, "w", encoding="utf-8").write(json.dumps({
            "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "assets": assets}, indent=1))
        # clips + lead first, then the state, then work.json LAST (a pull that finds a work.json finds its files)
        for f in files + [wj]:
            gh("release", "upload", TAG, f, "--clobber", "-R", REPO)
            log("uploaded %s (%.1f KB)" % (os.path.basename(f), os.path.getsize(f) / 1024.0))
        log("work state pushed: %s" % ", ".join("%s %s" % (k, v[:10]) for k, v in sorted(assets.items())))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def pull(out, lead_dst=None):
    os.makedirs(out, exist_ok=True)
    if not release_exists():
        # never a silent fresh start: it would render (and bill) the whole bank again
        raise SystemExit("[work] no `work` release on %s — seed it from the desk first: "
                         "python -m tools.voice_bank.work seed" % REPO)
    with tempfile.TemporaryDirectory() as d:
        gh("release", "download", TAG, "-R", REPO, "-D", d, "--clobber")
        aud = aud_dir(out)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.endswith((".tar", ".tar.gz")):
                with tarfile.open(p, "r:*") as t:
                    for ti in t.getmembers():
                        if not ti.isfile() or ti.name.startswith(("/", "..")) or "/../" in ti.name:
                            continue
                        rel = ti.name
                        dst = (os.path.join(aud, rel[len("audition/"):]) if rel.startswith("audition/")
                               else os.path.join(out, rel))
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as fh:
                            fh.write(t.extractfile(ti).read())
                log("unpacked %s" % f)
            elif f == "context_lead.mp3" and lead_dst:
                os.makedirs(os.path.dirname(lead_dst), exist_ok=True)
                shutil.copyfile(p, lead_dst)
                log("lead clip → %s" % lead_dst)


def lead_path():
    from tools.masri_voice import build_bank, masri_voice as mv
    return os.path.join(mv.ROOT, "assets", "voice", "ar", mv.fnv64(build_bank.CONTEXT_LEAD) + ".mp3")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push", "seed", "status"])
    ap.add_argument("--dir", default=OUT)
    a = ap.parse_args()
    if a.cmd == "pull":
        pull(a.dir, lead_dst=lead_path())
    elif a.cmd == "push":
        push(a.dir)
    elif a.cmd == "seed":
        push(a.dir, lead=lead_path())
    else:
        print(json.dumps(remote_work_json(), indent=1))


if __name__ == "__main__":
    main()
