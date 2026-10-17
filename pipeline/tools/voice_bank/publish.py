# -*- coding: utf-8 -*-
"""The navigation voice bank on GitHub — the publisher (the app downloads the bank at runtime).
docs/voice-bank-publish-2026-10.md has the contract, the run and the schedule.

    python -m tools.voice_bank.publish init                 # the public repo + README + the rolling release `bank`
                                                            # + its first manifest.json (no packs yet)
    python -m tools.voice_bank.publish run [--lang ar,en]   # the new QA-passed clips → ONE pack per language →
                                  [--dry-run]               # uploaded, then manifest.json LAST
    python -m tools.voice_bank.publish status               # the local record and the live manifest
    python -m tools.voice_bank.publish verify               # download the live manifest + every pack, check them

THE CONTRACT (the app's reader is built to it — never change a field without the reader):
  • Public repo ibrahim212-wq/routemind-voice, ONE rolling release tagged `bank` (not a prerelease); every file at
    https://github.com/ibrahim212-wq/routemind-voice/releases/download/bank/<file>.
  • manifest.json = {"format": 1, "updated": "<ISO UTC>", "banks": {"ar": {"voice": "gemini-3.8-flash-tts/Despina",
    "prompt_id": "<settings id>", "phrasebook": "<sha256 hex of nav-spec/masri_voice.json>", "segments": <n covered>,
    "packs": [{"file": "ar-0001-<sha8>.zip", "sha256": "<hex>", "bytes": <n>, "segments": <k>}, ...]},
    "en": {... "voice": "azure/en-US-PhoebeMultilingualNeural" ...}}}.
  • A pack = a zip (STORED — MP3 does not compress, and a stored zip is trivial to read on iOS) of
    `<fnv64(segment text)>.mp3` at its root — masri_voice.fnv64 of the exact segment text the app composes, the same
    function and naming as assets/voice/ar; English: the same hash over inventory.py's English segment text. Every
    clip is bare MPEG-2 Layer III frames, 24 kHz mono 32 kbps CBR (no ID3, no Xing/Info frame — mp3.strip), the
    bank's loudness: the app joins clips frame by frame. `validate` refuses anything else.
  • Packs are immutable (the name carries the first 8 hex of the zip's sha256); a later pack's clip overrides an
    earlier one with the same name. A pack holds only the clips that are new or changed since the last pack
    (a retake that replaced a published clip rides in the next pack).
  • Upload order per batch: the pack first, then the new manifest LAST (`gh release upload bank <file> --clobber`), so
    a reader never sees a manifest naming a pack that is not there.
  • `gh` uses its own login (the keyring); no token is read, printed or stored here.

ONLY QA-PASSED CLIPS: a segment ships its latest passing take (build.Build.published_take) — never a take that failed
or waits for a human listen. Local record (never in the repo): $VOICE_BANK_OUT/publish/ (state.json, packs/,
manifest.json).
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from tools.masri_voice import masri_voice as mv
from tools.masri_voice import mp3 as mp3mod
from tools.voice_bank import banks as BK
from tools.voice_bank import build as B

REPO = "ibrahim212-wq/routemind-voice"
TAG = "bank"
BASE_URL = "https://github.com/%s/releases/download/%s/" % (REPO, TAG)
PUB = os.path.join(B.OUT, "publish")
GH = os.environ.get("GH") or shutil.which("gh") or "gh"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)      # every entry: a fixed date, so one set of clips = one zip = one sha256

README = """# RouteMind voice bank

Pre-rendered navigation voice clips for **RouteMind**, a navigation app for Cairo & Giza (a graduation project).
The app downloads these clips after install and plays every guidance line from them, offline — no speech
synthesis at navigation time. There is no code here: only the release.

## What is in the release

One rolling release, tag [`bank`](https://github.com/ibrahim212-wq/routemind-voice/releases/tag/bank):

- `manifest.json` — format 1: per language (`ar`, `en`) the voice, the settings id, the phrasebook hash, how many
  segments are covered, and the list of packs with their sha256 and size. It is always uploaded LAST.
- `ar-NNNN-<sha8>.zip`, `en-NNNN-<sha8>.zip` — packs (stored zip) of `<fnv64 of the segment text>.mp3` clips: bare
  MPEG-2 Layer III frames, 24 kHz mono, 32 kbps. Packs never change; a later pack's clip replaces an earlier clip
  with the same name.

Every clip passed the bank's checks before it was published: a blind speech recogniser (Azure speech-to-text) read
it back (direction words and exit numbers heard, Arabic inside a line; the addressee's gender), its pace and length
were checked, and Arabic clips were also heard by a blind phone recogniser (the Cairene ج and ق). Clips that did not
pass are not here.

## Voices and credits

**These are synthetic, AI-generated voices — not recordings of a person.**

- **Arabic (Egyptian, Cairo):** generated with the Google Gemini API, model `gemini-3.8-flash-tts`, prebuilt voice
  *Despina*. Use is subject to the Gemini API Additional Terms of Service.
- **English:** generated with Microsoft Azure AI Speech, neural voice `en-US-PhoebeMultilingualNeural`. Use is
  subject to the Microsoft Product Terms and the Azure AI Speech code of conduct for text to speech.
- **Road names** spoken in some clips come from OpenStreetMap: (c) OpenStreetMap contributors, available under the
  Open Database License (ODbL) 1.0 — https://www.openstreetmap.org/copyright.
- The words (the phrasebook) are RouteMind's own.

The clips are published for the RouteMind app. No licence is granted beyond what the providers' terms above allow.

## How it is built

`.github/workflows/voice-bank.yml` runs every day on GitHub-hosted minutes from `pipeline/` — a generated copy of
the build code in the RouteMind app repository (never edited here). Every clip is read back by Azure speech-to-text
(Arabic also by a blind phone recogniser) before it is packed. The work in progress (state, verdicts, rendered clips)
is kept in the `work` prerelease — not for the app; the app reads the `bank` release only. `status/runs.jsonl` has one
line per run.
"""

RELEASE_NOTES = ("The RouteMind navigation voice bank: manifest.json (uploaded last) + immutable packs of clips. "
                 "See the README. Synthetic voices: Gemini 3.8 Flash TTS (Despina) for Arabic, Azure "
                 "en-US-PhoebeMultilingualNeural for English.")


class PublishError(Exception):
    pass


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


# ── gh ────────────────────────────────────────────────────────────────────────────────────────────────────
def gh(*args, check=True, timeout=600):
    env = dict(os.environ, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1", NO_COLOR="1")
    r = subprocess.run([GH] + [str(a) for a in args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout)
    if check and r.returncode != 0:
        raise PublishError("gh %s failed (%d): %s" % (" ".join(str(a) for a in args[:3]), r.returncode,
                                                      (r.stderr or r.stdout or "").strip()[-400:]))
    return r


def repo_exists():
    return gh("repo", "view", REPO, "--json", "name", check=False).returncode == 0


def release_exists():
    return gh("release", "view", TAG, "-R", REPO, "--json", "tagName", check=False).returncode == 0


def init():
    """The repo (public, a README, no code) and the rolling release `bank` with its first manifest."""
    if not repo_exists():
        log("creating the public repo %s" % REPO)
        gh("repo", "create", REPO, "--public", "--description",
           "RouteMind's pre-rendered navigation voice clips (Arabic + English) - the app downloads them after install")
    r = gh("api", "repos/%s/contents/README.md" % REPO, check=False)
    sha = json.loads(r.stdout).get("sha") if r.returncode == 0 else None
    want = base64.b64encode(README.encode("utf-8")).decode("ascii")
    have = json.loads(r.stdout).get("content", "").replace("\n", "") if r.returncode == 0 else None
    if have != want:
        args = ["api", "-X", "PUT", "repos/%s/contents/README.md" % REPO, "-f",
                "message=README: what the voice bank is, its format, the voices and their terms", "-f",
                "content=" + want]
        if sha:
            args += ["-f", "sha=" + sha]
        gh(*args)
        log("README written")
    if not release_exists():
        gh("release", "create", TAG, "-R", REPO, "--title", "RouteMind voice bank", "--notes", RELEASE_NOTES,
           "--latest")
        log("release %s created" % TAG)
    st = load_state()
    if not st.get("manifest_uploaded"):
        upload_manifest(st)


# ── the local record ─────────────────────────────────────────────────────────────────────────────────────
def state_path():
    return os.path.join(PUB, "state.json")


def load_state():
    p = state_path()
    if os.path.exists(p):
        return json.load(io.open(p, encoding="utf-8"))
    return {"banks": {lang: {"clips": {}, "packs": []} for lang in BK.BANKS}, "manifest_uploaded": None}


def save_state(st):
    os.makedirs(PUB, exist_ok=True)
    B.atomic_write(state_path(), json.dumps(st, ensure_ascii=False, indent=1))


def manifest(st):
    out = {"format": 1, "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "banks": {}}
    for lang, meta in BK.BANKS.items():
        s = st["banks"][lang]
        out["banks"][lang] = {
            "voice": meta["label"], "prompt_id": s.get("prompt_id") or BK.prompt_id(lang),
            "phrasebook": s.get("phrasebook") or B.phrasebook_hash(full=True), "segments": len(s["clips"]),
            "packs": [{k: p[k] for k in ("file", "sha256", "bytes", "segments")} for p in s["packs"]]}
    return out


def upload(path):
    gh("release", "upload", TAG, path, "--clobber", "-R", REPO, timeout=1800)


def upload_manifest(st):
    man = manifest(st)
    os.makedirs(PUB, exist_ok=True)
    p = os.path.join(PUB, "manifest.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    upload(p)
    st["manifest_uploaded"] = man["updated"]
    st["manifest_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    save_state(st)
    log("manifest.json uploaded (%s)" % ", ".join("%s: %d segments in %d packs" % (
        lang, b["segments"], len(b["packs"])) for lang, b in man["banks"].items()))
    return man


# ── clips → a pack ───────────────────────────────────────────────────────────────────────────────────────
def bank_frames(data):
    """A clip as the bundled bank stores it: bare frames (mp3.strip) with the header's copyright / original flags
    clear, as Media Foundation wrote assets/voice/ar (ff f3 44 c0; LAME sets 'original': … c4). No CRC is used
    (protection bit set), so the flags change nothing else."""
    b = bytearray(mp3mod.strip(data))
    for off, _, _, _ in mp3mod.frames(bytes(b)):
        b[off + 3] &= 0xF3
    return bytes(b)


def validate(data, name):
    """Bare MPEG-2 Layer III frames, 24 kHz, 32 kbps, mono — the bundled bank's format — or PublishError."""
    fr = mp3mod.frames(data)
    if not fr:
        raise PublishError("%s: no MPEG audio frames" % name)
    if sum(n for _, n, _, _ in fr) != len(data) or fr[0][0] != 0:
        raise PublishError("%s: not bare frames (a tag or a header frame is left)" % name)
    for off, n, spf, rate in fr:
        b = data[off:off + 4]
        ver = (b[1] >> 3) & 3
        _, _, _, br = mp3mod.frame_info(data, off)
        if ver != 2 or rate != 24000 or br != 32000 or ((b[3] >> 6) & 3) != 3 or spf != 576:
            raise PublishError("%s: a frame is not MPEG-2 L3 24 kHz 32 kbps mono (ver %d, %d Hz, %d bps)"
                               % (name, ver, rate, br))
        if not b[1] & 1 or b[3] & 0x0C:
            raise PublishError("%s: a frame carries a CRC or the copyright / original flags" % name)
    return len(fr)


def new_clips(lang, st, order):
    """{file name: (bytes, segment)} of every QA-passed clip of the bank that is not published as it is now."""
    b = BK.open_build(lang)
    have = st["banks"][lang]["clips"]
    out = {}
    for x in order:
        if x["lang"] != lang:
            continue
        r = b.s["segs"].get(lang + "|" + x["seg"])
        t = b.published_take(r) if r else None
        if not t:
            continue
        name = mv.fnv64(x["seg"]) + ".mp3"
        if name in out:
            raise PublishError("two segments hash to %s" % name)
        data = bank_frames(open(os.path.join(b.dir, t["clip"]), "rb").read())
        validate(data, name)
        sha = hashlib.sha256(data).hexdigest()
        if have.get(name, {}).get("sha256") != sha:
            out[name] = (data, x["seg"], sha, t["take"])
    return out, b


def make_pack(lang, clips, number):
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(clips):
            zi = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, clips[name][0])
    data = zb.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    fname = "%s-%04d-%s.zip" % (lang, number, sha[:8])
    os.makedirs(os.path.join(PUB, "packs"), exist_ok=True)
    path = os.path.join(PUB, "packs", fname)
    open(path, "wb").write(data)
    return {"file": fname, "sha256": sha, "bytes": len(data), "segments": len(clips), "path": path}


MAX_PACKS = 150     # the reader ignores a manifest with > 200 packs (docs/voice-bank-download-2026-10.md §1): past
                    # this many, the language is re-packed into ONE pack of every published clip (a replacing list)


def run(langs=("ar", "en"), dry=False, order=None):
    """→ [{lang, file, segments, bytes}] published by this call (dry: what would be)."""
    if order is None:
        order = BK.order()[0]
    st = load_state()
    done = []
    for lang in langs:
        s = st["banks"][lang]
        whole = len(s["packs"]) >= MAX_PACKS
        clips, b = new_clips(lang, {"banks": {lang: {"clips": {}}}} if whole else st, order)
        if not clips:
            log("%s: no new QA-passed clip (%d published)" % (lang, len(s["clips"])))
            continue
        number = max([int(p["file"].split("-")[1]) for p in s["packs"]] + [0]) + 1
        pack = make_pack(lang, clips, number)
        if whole:
            log("%s: %d packs — re-packed into one pack of all %d clips" % (lang, len(s["packs"]), len(clips)))
        log("%s: pack %s — %d clips, %.1f KB%s" % (lang, pack["file"], pack["segments"], pack["bytes"] / 1024.0,
                                                   " (dry run, not uploaded)" if dry else ""))
        if dry:
            done.append({"lang": lang, "file": pack["file"], "segments": pack["segments"], "bytes": pack["bytes"],
                         "dry": True})
            continue
        upload(pack["path"])                      # the pack first …
        if whole:
            s["packs"], s["clips"] = [], {}
        s["packs"].append({k: pack[k] for k in ("file", "sha256", "bytes", "segments")})
        s["packs"][-1]["at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for name, (_, seg, sha, take) in clips.items():
            s["clips"][name] = {"sha256": sha, "seg": seg, "take": take, "pack": pack["file"]}
        s["prompt_id"] = b.s["bank"]["prompt_id"]
        s["phrasebook"] = B.phrasebook_hash(full=True)
        st["manifest_uploaded"] = None            # the manifest owes this pack
        save_state(st)
        done.append({"lang": lang, "file": pack["file"], "segments": pack["segments"], "bytes": pack["bytes"]})
    if not dry and (done or not st.get("manifest_uploaded")):
        upload_manifest(st)                       # … the manifest LAST
        check_live(st)
    return done


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "routemind-voice-publisher", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def check_live(st):
    """The public URL serves the manifest just uploaded (GitHub's download redirect, no login)."""
    try:
        live = json.loads(fetch(BASE_URL + "manifest.json").decode("utf-8"))
    except Exception as e:                        # a slow CDN is not a failed publish: the next run checks again
        log("  live manifest not readable yet: %s" % str(e)[:160])
        return None
    ok = live.get("updated") == st.get("manifest_uploaded")
    log("  live manifest %s (%s)" % ("matches" if ok else "is still the previous one", live.get("updated")))
    return live


def verify():
    """Every pack of the live manifest: present, its sha256 and size, every clip bare frames in the bank's format."""
    live = json.loads(fetch(BASE_URL + "manifest.json").decode("utf-8"))
    rep = {"updated": live.get("updated"), "banks": {}}
    for lang, b in live["banks"].items():
        names = set()
        for p in b["packs"]:
            data = fetch(BASE_URL + p["file"], timeout=300)
            if hashlib.sha256(data).hexdigest() != p["sha256"] or len(data) != p["bytes"]:
                raise PublishError("%s: sha256 / size differ from the manifest" % p["file"])
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = z.infolist()
                if len(infos) != p["segments"]:
                    raise PublishError("%s: %d clips, the manifest says %d" % (p["file"], len(infos), p["segments"]))
                for zi in infos:
                    validate(z.read(zi), zi.filename)
                    names.add(zi.filename)
        if len(names) != b["segments"]:
            raise PublishError("%s: %d distinct clips in the packs, the manifest says %d" % (lang, len(names),
                                                                                           b["segments"]))
        rep["banks"][lang] = {"packs": len(b["packs"]), "segments": len(names), "voice": b["voice"]}
    return rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run", "status", "verify"])
    ap.add_argument("--lang", default="ar,en")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "init":
        init()
    elif a.cmd == "run":
        print(json.dumps(run(tuple(a.lang.split(",")), dry=a.dry_run), indent=1))
    elif a.cmd == "status":
        st = load_state()
        print(json.dumps({"local": {lang: {"segments": len(s["clips"]), "packs": [p["file"] for p in s["packs"]]}
                                    for lang, s in st["banks"].items()},
                          "manifest_uploaded": st.get("manifest_uploaded")}, indent=1))
        try:
            live = json.loads(fetch(BASE_URL + "manifest.json").decode("utf-8"))
            print(json.dumps({"live": {lang: {"segments": b["segments"], "packs": len(b["packs"])}
                                       for lang, b in live["banks"].items()}, "updated": live["updated"]}, indent=1))
        except Exception as e:
            print("live manifest: %s" % e)
    else:
        print(json.dumps(verify(), indent=1))


if __name__ == "__main__":
    main()
