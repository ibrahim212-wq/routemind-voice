# -*- coding: utf-8 -*-
"""The live bank, read EXACTLY as the app reads it — a Python transcription of the reader's decisions
(android/.../nav/VoiceBankRemote.kt `parseManifest` + `Store.install`, ⇄ ios/Runner/Nav/VoiceBankRemote.swift; the
constants and patterns are READ from the Kotlin file, so this check cannot drift from it).

    python -m tools.voice_bank.reader_check [--base URL] [--all-packs]

  1. the manifest from the public URL (GitHub's 302 to the asset host followed, as the app's HTTP client does), under
     MAX_MANIFEST_BYTES; parseManifest: format == FORMAT, every pack's file / sha256 / bytes / segments valid, no
     file or sha listed twice, ≤ MAX_PACKS per language; a language without packs is absent;
  2. each pack (the newest per language, or --all-packs): the download capped at its listed `bytes`, sha256 equal to
     the listing; install: only `<16 hex>.mp3` entries, each ≤ MAX_CLIP_BYTES, the total within 4 × bytes + 1 MB, and
     a clip kept only when it begins on an MPEG audio frame after VoiceBank.strip (here: tools/masri_voice/mp3) —
     plus the publisher's own stricter format check (publish.validate: bare MPEG-2 L3 24 kHz mono 32 kbps);
  3. the clips the app would hold play the inventory: every kept clip name is fnv64 of an inventory segment of that
     language, and (English) a few real lines split into banked segments (VoiceBankRemote.splitEnglish's third copy,
     tool/voice_bank_remote_parity.split_english).
Exit 1 on any failure. Nothing is written.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

from tools.masri_voice import masri_voice as mv
from tools.masri_voice import mp3 as mp3mod

ROOT = mv.ROOT
KT = os.path.join(ROOT, "android", "app", "src", "main", "java", "com", "routemind", "app", "nav", "VoiceBankRemote.kt")
fails = []


def check(ok, what):
    print(("OK    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        fails.append(what)
    return ok


def reader_constants():
    src = io.open(KT, encoding="utf-8").read()

    def const(name):
        m = re.search(r'const val %s = ("[^"]*"|[^\n]+)' % name, src)
        e = m.group(1)
        if e.startswith('"'):
            return e.strip('"')
        e = e.split("//")[0].strip().replace("_", "").replace("L", "")
        return eval(e)  # noqa: S307 — digits and * only (the Kotlin literal)

    def rx(name):
        m = re.search(r'private val %s = Regex\("(.+?)"\)\n' % name, src)
        return re.compile(m.group(1).replace("\\\\", "\\"))

    return {"FORMAT": const("FORMAT"), "DEFAULT_BASE": const("DEFAULT_BASE"), "MANIFEST": const("MANIFEST"),
            "MAX_MANIFEST_BYTES": const("MAX_MANIFEST_BYTES"), "MAX_BANK_BYTES": const("MAX_BANK_BYTES"),
            "MAX_CLIP_BYTES": const("MAX_CLIP_BYTES"), "MAX_PACKS": const("MAX_PACKS"),
            "PACK_FILE": rx("PACK_FILE"), "SHA256": rx("SHA256"), "CLIP": rx("CLIP")}


def fetch(url, cap):
    req = urllib.request.Request(url, headers={"User-Agent": "routemind-reader-check", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=300) as r:          # urllib follows GitHub's 302 like the app does
        data = r.read(cap + 1)
    return data if len(data) <= cap else None


def parse_manifest(text, C):
    """VoiceBankRemote.parseManifest, line for line → {lang: bank} or None (the whole manifest is ignored)."""
    try:
        o = json.loads(text)
    except ValueError:
        return None
    if not isinstance(o, dict) or not isinstance(o.get("format"), (int, float)) or float(o["format"]) != C["FORMAT"]:
        return None
    if not isinstance(o.get("banks"), dict):
        return None
    banks = {}
    for lang in ("ar", "en"):
        raw = o["banks"].get(lang)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            return None
        lst = raw.get("packs")
        if not isinstance(lst, list) or len(lst) > C["MAX_PACKS"]:
            return None
        packs, files, shas = [], set(), set()
        for p in lst:
            p = p if isinstance(p, dict) else {}
            f, sha, b = p.get("file"), p.get("sha256"), p.get("bytes")
            sha = sha.lower() if isinstance(sha, str) else None
            segs = p.get("segments") if isinstance(p.get("segments"), (int, float)) else 0
            if (not isinstance(f, str) or not C["PACK_FILE"].match(f) or sha is None or not C["SHA256"].match(sha)
                    or not isinstance(b, (int, float)) or b < 1 or b > C["MAX_BANK_BYTES"] or b != int(b) or segs < 0
                    or f in files or sha in shas):
                return None
            files.add(f)
            shas.add(sha)
            packs.append({"file": f, "sha256": sha, "bytes": int(b), "segments": int(segs)})
        if packs:
            banks[lang] = dict(raw, packs=packs)
    return banks


def install(data, pack, C):
    """Store.install's checks on the downloaded bytes → {clip name: bytes} kept, or None (the pack fails)."""
    if len(data) > pack["bytes"] or hashlib.sha256(data).hexdigest() != pack["sha256"]:
        return None
    kept, total, cap = {}, 0, 4 * pack["bytes"] + 1024 * 1024
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for zi in z.infolist():
            if zi.is_dir():
                continue
            name = zi.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if not C["CLIP"].match(name):
                continue
            b = z.read(zi)
            if len(b) > C["MAX_CLIP_BYTES"]:
                return None
            total += len(b)
            if total > cap:
                return None
            fr = mp3mod.frames(mp3mod.strip(b)) if b else []
            if not fr or fr[0][0] != 0:                 # VoiceBank.frameInfo(strip(b), 0) == null → skipped
                continue
            kept[name] = b
    return kept


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--all-packs", action="store_true")
    a = ap.parse_args()
    C = reader_constants()
    base = a.base or C["DEFAULT_BASE"]
    check(C["DEFAULT_BASE"].endswith("/releases/download/bank/"), "the reader's base URL is the bank release (%s)"
          % C["DEFAULT_BASE"])
    raw = fetch(base + C["MANIFEST"], C["MAX_MANIFEST_BYTES"])
    if not check(raw is not None, "manifest.json fetched within %d bytes" % C["MAX_MANIFEST_BYTES"]):
        return 1
    banks = parse_manifest(raw.decode("utf-8"), C)
    if not check(banks is not None, "parseManifest accepts the live manifest"):
        return 1
    from tools.voice_bank import publish as P
    from tools.voice_bank import banks as BK
    order = BK.order()[0]
    inv = {lang: {mv.fnv64(x["seg"]) + ".mp3": x["seg"] for x in order if x["lang"] == lang} for lang in ("ar", "en")}
    for lang, b in sorted(banks.items()):
        total = sum(p["bytes"] for p in b["packs"])
        check(total <= C["MAX_BANK_BYTES"], "%s: %d packs, %.1f MB listed (≤ %d MB the reader fetches)" % (
            lang, len(b["packs"]), total / 1e6, C["MAX_BANK_BYTES"] // (1024 * 1024)))
        held = {}
        for p in (b["packs"] if a.all_packs else b["packs"][-1:]):
            data = fetch(base + p["file"], p["bytes"])
            if not check(data is not None, "%s: %s downloaded within its listed %d bytes" % (lang, p["file"],
                                                                                           p["bytes"])):
                continue
            kept = install(data, p, C)
            if not check(kept is not None, "%s: %s — sha256 %s… and every entry pass the reader's install" % (
                    lang, p["file"], p["sha256"][:12])):
                continue
            check(len(kept) == p["segments"], "%s: %s keeps %d clips (the listing says %d)" % (
                lang, p["file"], len(kept), p["segments"]))
            bad = []
            for name, clip in kept.items():
                try:
                    P.validate(clip, name)
                except P.PublishError as e:
                    bad.append(str(e))
            check(not bad, "%s: every clip is bare MPEG-2 Layer III 24 kHz mono 32 kbps (header %s)%s" % (
                lang, next(iter(kept.values()))[:4].hex() if kept else "-", (" — " + bad[0]) if bad else ""))
            unknown = [n for n in kept if n not in inv[lang]]
            check(not unknown, "%s: every clip name is fnv64 of a %s inventory segment (%d unknown)" % (
                lang, lang, len(unknown)))
            held.update(kept)
        if lang == "en" and held:
            sys.path.insert(0, os.path.join(ROOT, "tool"))
            import voice_bank_remote_parity as VR
            names = set(held)

            def has(seg):
                return mv.fnv64(seg) + ".mp3" in names
            for line in ("In 200 meters, turn right.", "Turn right.", "In 500 meters, keep left.",
                         "Finding a new route."):
                parts = VR.split_english(line, has)
                print("      %-30s → %s" % (line, " | ".join(parts) if parts else "not covered by this pack"))
    print("\n%s" % ("ALL OK" if not fails else "%d FAILED" % len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
