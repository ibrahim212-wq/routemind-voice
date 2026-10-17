# RouteMind voice bank

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

Every clip passed the bank's checks before it was published: a blind listening model read it back (direction words
and exit numbers heard inside a line, the addressee's gender, the Cairene accent), its pace and length were checked,
and Arabic clips were also heard by a blind phone recogniser. Clips that did not pass are not here.

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
