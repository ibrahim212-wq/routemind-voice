"""The voice-bank pipeline mirror — ONE source of truth (this app repository), run on free GitHub-hosted minutes in
the PUBLIC repo ibrahim212-wq/routemind-voice (docs/voice-bank-publish-2026-10.md §7), the way
tools/place_index/mirror.py mirrors the place-index pipeline into routemind-places.

The public repo's `pipeline/` is a GENERATED, byte-exact copy of

    tools/__init__.py   tools/voice_bank/**   tools/voice_audition/{__init__,audition,azure,bank_plan}.py
    tools/masri_voice/{__init__,acoustics,build_bank,inventory,masri_voice,mp3,phones,pronounce,route_check,tts}.py
    tools/traffic_advisor/traffic_advisor.py   nav-spec/masri_voice.json (the phrasebook)

plus two generated files: `pipeline/data/inventory.json` (the RANKED inventory as data — segment texts, categories,
ranks and the corpus lines as segment lists; the route corpora and the road list it is computed from stay here, no
route geometry is ever published) and `pipeline/SOURCE.json` (the app commit + every file's sha256). The workflow
`tools/voice_bank/ci/voice-bank.yml` becomes `.github/workflows/voice-bank.yml` there and publish.README its README.md.
Nothing in the public repo is edited by hand; audio never enters it (the lead clip and the clips live in the `work`
prerelease: tools/voice_bank/work.py).

    python tools/voice_bank/mirror.py check  [--ref HEAD]     # the public copy == this checkout? (exit 1 on drift)
    python tools/voice_bank/mirror.py push   [--ref HEAD]     # one commit (GitHub API) with every changed file

Without --ref, the WORKING TREE (the files as they are on disk); with --ref, the committed files of that commit. The
inventory is always computed from the working tree's code + data (it needs the desk's corpora). Token (never written
anywhere): $GH_TOKEN, else `gh auth token`; the workflow file needs the `workflow` scope. Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = "ibrahim212-wq/routemind-voice"
APP_REPO = "ibrahim212-wq/maps-mobile-application"
PREFIX = "pipeline/"
WORKFLOW_SRC = "tools/voice_bank/ci/voice-bank.yml"
WORKFLOW_DST = ".github/workflows/voice-bank.yml"
INVENTORY = "data/inventory.json"
SINGLE_FILES = (
    "tools/__init__.py", "nav-spec/masri_voice.json", "tools/traffic_advisor/traffic_advisor.py",
    "tools/voice_audition/__init__.py", "tools/voice_audition/audition.py", "tools/voice_audition/azure.py",
    "tools/voice_audition/bank_plan.py",
    *("tools/masri_voice/%s.py" % m for m in ("__init__", "acoustics", "build_bank", "inventory", "masri_voice", "mp3",
                                              "phones", "pronounce", "route_check", "tts")))
TREE = "tools/voice_bank"
SKIP = (".pyc", ".pyo", ".ps1")
GENERATED_README = """# pipeline/ — a generated copy. Do not edit here.

A byte-exact copy of the navigation voice-bank pipeline from the RouteMind app repository (`cairoWay/tools/voice_bank`
and the modules it uses, the phrasebook `nav-spec/masri_voice.json`), plus `data/inventory.json` (the ranked list of
segments to render, generated there) and `SOURCE.json` (the app commit and every file's sha256). It is replaced by
`python tools/voice_bank/mirror.py push` in the app repository after every change.

The daily workflow (`.github/workflows/voice-bank.yml`) runs `python -m tools.voice_bank.daily` from here: Arabic on
the free Gemini API (`gemini-3.8-flash-tts`, voice Despina) until the day's quota, English on Azure Speech
(`en-US-PhoebeMultilingualNeural`), every clip read back by Azure speech-to-text (and, Arabic, a blind phone
recogniser), and the passing clips published to the `bank` release. Its work in progress lives in the `work`
prerelease (`tools/voice_bank/work.py`), never in git.
"""


def log(*a):
    print("[mirror]", *a, flush=True)


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        b = f.read()
    return b.replace(b"\r\n", b"\n")


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", APP, *args], capture_output=True, check=True).stdout


def files(ref: str | None) -> dict[str, bytes]:
    out = {}
    if ref:
        names = _git("ls-tree", "-r", "--name-only", ref, "--", *SINGLE_FILES, TREE).decode().splitlines()
        for rel in (n.strip() for n in names):
            if rel and "/__pycache__/" not in rel and not rel.endswith(SKIP):
                out[rel] = _git("show", f"{ref}:./{rel}").replace(b"\r\n", b"\n")
        return dict(sorted(out.items()))
    for rel in SINGLE_FILES:
        out[rel] = _read(os.path.join(APP, rel))
    for dirpath, dirs, names in os.walk(os.path.join(APP, TREE)):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for n in sorted(names):
            if n.endswith(SKIP) or n.startswith("."):
                continue
            p = os.path.join(dirpath, n)
            out[os.path.relpath(p, APP).replace(os.sep, "/")] = _read(p)
    return dict(sorted(out.items()))


def inventory_json() -> bytes:
    """The ranked inventory, computed here (the corpora are the desk's) — `inventory.freeze`."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "inventory.json")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("VOICE_BANK_INVENTORY", None)
        subprocess.run([sys.executable, "-c", "import sys; from tools.voice_bank import inventory as I; "
                        "I.freeze(sys.argv[1])", p], cwd=APP, env=env, check=True)
        return _read(p)


def readme() -> bytes:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", "import sys; from tools.voice_bank import publish as P; "
                        "sys.stdout.buffer.write(P.README.encode('utf-8'))"], cwd=APP, env=env, check=True,
                       capture_output=True)
    return r.stdout


def app_commit(ref: str | None) -> str | None:
    try:
        sha = _git("rev-parse", ref or "HEAD").decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if ref is None and _git("status", "--porcelain", "--", *SINGLE_FILES, TREE).strip():
        sha += "+worktree"
    return sha


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wanted(ref: str | None) -> tuple[dict[str, bytes], dict[str, bytes]]:
    fs = files(ref)
    pipe = {PREFIX + k: v for k, v in fs.items()}
    pipe[PREFIX + INVENTORY] = inventory_json()
    doc = {"app_repo": APP_REPO, "app_commit": app_commit(ref),
           "note": "generated by tools/voice_bank/mirror.py - edit in the app repository",
           "files": {k[len(PREFIX):]: sha256(v) for k, v in sorted(pipe.items())}}
    pipe[PREFIX + "SOURCE.json"] = (json.dumps(doc, indent=1) + "\n").encode()
    pipe[PREFIX + "README.md"] = GENERATED_README.encode()
    extra = {WORKFLOW_DST: fs[WORKFLOW_SRC], "README.md": readme()}
    return pipe, extra


# ── GitHub API ───────────────────────────────────────────────────────────────────────────────────────
def token() -> str | None:
    t = os.environ.get("GH_TOKEN")
    if t:
        return t
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def api(method: str, path: str, tok: str | None, body=None):
    hdr = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
           "User-Agent": "routemind-voice-mirror"}
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request("https://api.github.com" + path, data=data, method=method, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"message": raw[:300].decode(errors="replace")}


def git_blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def remote_tree(repo: str, tok: str | None):
    code, ref = api("GET", f"/repos/{repo}/git/ref/heads/main", tok)
    if code != 200:
        raise SystemExit(f"[mirror] {repo} main not readable ({code}: {ref})")
    head = ref["object"]["sha"]
    _, commit = api("GET", f"/repos/{repo}/git/commits/{head}", tok)
    _, tree = api("GET", f"/repos/{repo}/git/trees/{commit['tree']['sha']}?recursive=1", tok)
    return head, commit["tree"]["sha"], {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}


def diff(repo: str, tok: str | None, ref: str | None):
    head, tree_sha, have = remote_tree(repo, tok)
    pipe, extra = wanted(ref)
    changed = {p: b for p, b in {**pipe, **extra}.items()
               if have.get(p) != git_blob_sha(b) and not p.endswith("SOURCE.json")}
    gone = [p for p in have if p.startswith(PREFIX) and p not in pipe]
    return head, tree_sha, pipe, changed, gone


def check(repo: str, ref: str | None) -> int:
    _, _, _, changed, gone = diff(repo, token(), ref)
    if not changed and not gone:
        log(f"OK — {repo}/{PREFIX}, its workflow and README equal {'the working tree' if ref is None else ref}")
        return 0
    for p in sorted(changed):
        log("differs:", p)
    for p in gone:
        log("only in the mirror:", p)
    log("DRIFT — run: python tools/voice_bank/mirror.py push")
    return 1


def push(repo: str, ref: str | None, message: str | None) -> int:
    tok = token()
    if not tok:
        raise SystemExit("[mirror] no token: set GH_TOKEN or log in with `gh auth login`")
    head, tree_sha, pipe, changed, gone = diff(repo, tok, ref)
    if not changed and not gone:
        log("nothing to push — the mirror is current")
        return 0
    changed[PREFIX + "SOURCE.json"] = pipe[PREFIX + "SOURCE.json"]
    entries = []
    for p, b in sorted(changed.items()):
        code, blob = api("POST", f"/repos/{repo}/git/blobs", tok,
                         {"content": base64.b64encode(b).decode(), "encoding": "base64"})
        if code != 201:
            raise SystemExit(f"[mirror] blob {p}: {code} {blob}")
        entries.append({"path": p, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    entries += [{"path": p, "mode": "100644", "type": "blob", "sha": None} for p in gone]
    code, tree = api("POST", f"/repos/{repo}/git/trees", tok, {"base_tree": tree_sha, "tree": entries})
    if code != 201:
        raise SystemExit(f"[mirror] tree: {code} {tree}")
    sha = app_commit(ref) or "?"
    msg = message or f"pipeline: mirror of {APP_REPO}@{sha[:12]}{'+worktree' if sha.endswith('+worktree') else ''}"
    code, com = api("POST", f"/repos/{repo}/git/commits", tok, {"message": msg, "tree": tree["sha"], "parents": [head]})
    if code != 201:
        raise SystemExit(f"[mirror] commit: {code} {com}")
    code, res = api("PATCH", f"/repos/{repo}/git/refs/heads/main", tok, {"sha": com["sha"]})
    if code != 200:
        raise SystemExit(f"[mirror] update ref: {code} {res}")
    log(f"pushed {len(changed)} file(s), removed {len(gone)} → {repo}@{res['object']['sha'][:12]}")
    return 0


def main(argv=None):
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "push"])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ref", default=None, help="a commit (default: the working tree)")
    ap.add_argument("--message")
    a = ap.parse_args(argv)
    return check(a.repo, a.ref) if a.cmd == "check" else push(a.repo, a.ref, a.message)


if __name__ == "__main__":
    sys.exit(main())
