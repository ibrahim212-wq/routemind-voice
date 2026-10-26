# -*- coding: utf-8 -*-
"""The names the LIVE bank publishes, as a fixture the readers replay — docs/voice-bank-download-2026-10.md §7.

    python -m tools.voice_bank.published_keys [--base URL] [--write]

Downloads the live manifest and every pack exactly as `reader_check` does (the reader's own constants and install
checks), maps each published clip name back to the segment text of the inventory (`banks.order`, the publisher's
ranked inventory: a name is `fnv64(segment)`), and prints how many clips each language publishes by category. With
--write it stores `tools/voice_bank/fixtures/published_keys.tsv` — one row `lang<TAB>text<TAB>name` per published
clip, sorted — which:
  • `android .../VoiceBankRemoteTest.publishedClipNamesAreTheReadersKeys` replays through `VoiceBankRemote.clipName`
    (every row: the Kotlin key function equals the name the bank holds, and an English row reaches its own clip as a
    one-segment line);
  • `tool/voice_bank_remote_parity.py` §6 checks with the Python reference (`masri_voice.fnv64`);
  • `ios/RunnerTests/VoiceBankRemoteTests.testPublishedClipNamesAreTheReadersKeys` replays a spread of the rows.
Audio never enters the repo: the fixture is text and names. A clip whose name no inventory segment explains is listed
and the run exits 1 (it could never be asked for). Re-run after every publish that adds a pack.
"""
import argparse
import io
import os
import sys

from tools.masri_voice import masri_voice as mv
from tools.voice_bank import banks as BK
from tools.voice_bank import reader_check as RC

FIXTURE = os.path.join(mv.ROOT, "tools", "voice_bank", "fixtures", "published_keys.tsv")


def published(base=None):
    """({lang: {clip name: segment text}}, manifest updated, [unexplained names])."""
    C = RC.reader_constants()
    base = base or C["DEFAULT_BASE"]
    raw = RC.fetch(base + C["MANIFEST"], C["MAX_MANIFEST_BYTES"])
    if raw is None:
        raise SystemExit("manifest.json not fetched")
    banks = RC.parse_manifest(raw.decode("utf-8"), C)
    if banks is None:
        raise SystemExit("the live manifest is not one the reader accepts")
    import json
    updated = json.loads(raw.decode("utf-8")).get("updated", "")
    order = BK.order()[0]
    inv = {lang: {mv.fnv64(x["seg"]) + ".mp3": x for x in order if x["lang"] == lang} for lang in ("ar", "en")}
    held, unknown = {}, []
    for lang, b in sorted(banks.items()):
        held[lang] = {}
        for p in b["packs"]:
            data = RC.fetch(base + p["file"], p["bytes"])
            kept = RC.install(data, p, C) if data is not None else None
            if kept is None:
                raise SystemExit("%s %s: the reader would not install it" % (lang, p["file"]))
            for name in kept:
                if name in inv[lang]:
                    held[lang][name] = inv[lang][name]
                else:
                    unknown.append((lang, name))
    return held, updated, unknown


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    held, updated, unknown = published(a.base)
    print("manifest updated %s" % updated)
    total = 0
    for lang, d in sorted(held.items()):
        cats = {}
        for x in d.values():
            cats[x["cat"]] = cats.get(x["cat"], 0) + 1
        total += len(d)
        print("  %s: %d clips  %s" % (lang, len(d), ", ".join("%s %d" % kv for kv in sorted(cats.items()))))
    print("  total %d clips; named roads: %s" % (total, ", ".join(
        "%s %d" % (lang, sum(1 for x in d.values() if x["cat"] == "road")) for lang, d in sorted(held.items()))))
    if unknown:
        print("UNEXPLAINED (no inventory segment has this name): %s" % unknown[:10])
    if a.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        rows = sorted((lang, x["seg"], name[:-4]) for lang, d in held.items() for name, x in d.items())
        with io.open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            for lang, text, name in rows:
                f.write("%s\t%s\t%s\n" % (lang, text, name))
        print("wrote %s (%d rows)" % (os.path.relpath(FIXTURE, mv.ROOT), len(rows)))
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
