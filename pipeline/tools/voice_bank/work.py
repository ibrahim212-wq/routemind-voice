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


PART_BYTES = 2_500_000   # an asset larger than this goes up in parts: the desk's uplink stalls on one 10 MB body
                         # (HTTP 408 "upload body timed out due to inactivity", 2026-10-17); 1-3 MB go up in ~10 s


def _sha(v):
    return v.get("sha256") if isinstance(v, dict) else v


def _stage_asset(stage, name, data):
    """Write `name` (whole, or as name.partNNN when larger than PART_BYTES) → (listing entry, [paths])."""
    sha = hashlib.sha256(data).hexdigest()
    if len(data) <= PART_BYTES:
        p = os.path.join(stage, name)
        open(p, "wb").write(data)
        return {"sha256": sha, "bytes": len(data)}, [p]
    parts, paths = [], []
    for i in range(0, len(data), PART_BYTES):
        pn = "%s.part%03d" % (name, i // PART_BYTES)
        p = os.path.join(stage, pn)
        open(p, "wb").write(data[i:i + PART_BYTES])
        parts.append(pn)
        paths.append(p)
    return {"sha256": sha, "bytes": len(data), "parts": parts}, paths


def push(out, lead=None):
    import datetime
    ensure_release()
    have = remote_work_json().get("assets", {})
    stage = tempfile.mkdtemp(prefix="vb-work-")
    try:
        assets, files = {}, []
        named = []
        for build in sorted(m["build"] for m in BK.BANKS.values()):
            tmp = os.path.join(stage, "_t")
            write_tar(tmp, clip_members(out, build), gz=False)
            named.append(("clips-%s.tar" % build, open(tmp, "rb").read(), False))
        if lead:
            named.append(("context_lead.mp3", open(lead, "rb").read(), False))
        tmp = os.path.join(stage, "_t")
        write_tar(tmp, state_members(out), gz=True)
        named.append(("state.tar.gz", open(tmp, "rb").read(), True))   # the state goes every time (the run's record)
        os.remove(tmp)
        for name, data, always in named:
            entry, paths = _stage_asset(stage, name, data)
            assets[name] = entry
            if always or _sha(have.get(name)) != entry["sha256"]:
                files += paths
            elif isinstance(have.get(name), dict):
                assets[name] = have[name]                   # unchanged: keep the uploaded listing (its part names)
        if not lead and "context_lead.mp3" in have:
            assets["context_lead.mp3"] = have["context_lead.mp3"]
        wj = os.path.join(stage, "work.json")
        io.open(wj, "w", encoding="utf-8").write(json.dumps({
            "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "assets": assets}, indent=1))
        # clips + lead first, then the state, then work.json LAST (a pull that finds a work.json finds its files)
        for f in files + [wj]:
            for attempt in range(3):
                r = gh("release", "upload", TAG, f, "--clobber", "-R", REPO, check=attempt == 2)
                if r.returncode == 0:
                    break
                log("upload of %s failed (%s) — again" % (os.path.basename(f), (r.stderr or "").strip()[-120:]))
            log("uploaded %s (%.1f KB)" % (os.path.basename(f), os.path.getsize(f) / 1024.0))
        # assets no listing names any more (old parts, a test file) go
        keep = {"work.json"} | {n for n, e in assets.items() if not isinstance(e, dict) or not e.get("parts")}
        keep |= {pn for e in assets.values() if isinstance(e, dict) for pn in (e.get("parts") or [])}
        r = gh("release", "view", TAG, "-R", REPO, "--json", "assets", check=False)
        if r.returncode == 0:
            for a_ in json.loads(r.stdout).get("assets", []):
                if a_["name"] not in keep:
                    gh("release", "delete-asset", TAG, a_["name"], "-R", REPO, "-y", check=False)
                    log("removed the stale asset %s" % a_["name"])
        log("work state pushed: %s" % ", ".join("%s %s" % (k, _sha(v)[:10]) for k, v in sorted(assets.items())))
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
        wj = os.path.join(d, "work.json")
        listing = json.load(io.open(wj, encoding="utf-8")).get("assets", {}) if os.path.exists(wj) else {}
        aud = aud_dir(out)
        for name, entry in sorted(listing.items()):
            parts = entry.get("parts") if isinstance(entry, dict) else None
            data = (b"".join(open(os.path.join(d, pn), "rb").read() for pn in parts) if parts
                    else open(os.path.join(d, name), "rb").read())
            if hashlib.sha256(data).hexdigest() != _sha(entry):
                raise SystemExit("[work] %s does not match work.json (a push was interrupted?)" % name)
            if name.endswith((".tar", ".tar.gz")):
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                    for ti in t.getmembers():
                        if not ti.isfile() or ti.name.startswith(("/", "..")) or "/../" in ti.name:
                            continue
                        rel = ti.name
                        dst = (os.path.join(aud, rel[len("audition/"):]) if rel.startswith("audition/")
                               else os.path.join(out, rel))
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as fh:
                            fh.write(t.extractfile(ti).read())
                log("unpacked %s (%.1f KB)" % (name, len(data) / 1024.0))
            elif name == "context_lead.mp3" and lead_dst and not os.path.exists(lead_dst):    # the desk has it
                os.makedirs(os.path.dirname(lead_dst), exist_ok=True)
                open(lead_dst, "wb").write(data)
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
