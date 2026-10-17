# pipeline/ — a generated copy. Do not edit here.

A byte-exact copy of the navigation voice-bank pipeline from the RouteMind app repository (`cairoWay/tools/voice_bank`
and the modules it uses, the phrasebook `nav-spec/masri_voice.json`), plus `data/inventory.json` (the ranked list of
segments to render, generated there) and `SOURCE.json` (the app commit and every file's sha256). It is replaced by
`python tools/voice_bank/mirror.py push` in the app repository after every change.

The daily workflow (`.github/workflows/voice-bank.yml`) runs `python -m tools.voice_bank.daily` from here: Arabic on
the free Gemini API (`gemini-3.8-flash-tts`, voice Despina) until the day's quota, English on Azure Speech
(`en-US-PhoebeMultilingualNeural`), every clip read back by Azure speech-to-text (and, Arabic, a blind phone
recogniser), and the passing clips published to the `bank` release. Its work in progress lives in the `work`
prerelease (`tools/voice_bank/work.py`), never in git.
