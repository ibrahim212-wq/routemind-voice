# -*- coding: utf-8 -*-
"""A BLIND phone recogniser for the bank — what the audio says, as IPA, never told the words.

The transcribe-back QA (build_bank: Gemini ASR) hears WORDS: a transcriber that knows Arabic writes
«طريق» whether the voice said ṭarīq, ṭarīg or ṭarīʔ, and «فطار» / «فطور» collapse the same way. So it
passed every line Khaled heard as wrong. This listener is different in kind:

  facebook/wav2vec2-xlsr-53-espeak-cv-ft — a CTC phone recogniser trained on CommonVoice in many
  languages, labelled in espeak IPA. It outputs PHONES from the acoustics alone: no word model, no
  Arabic spelling to fall back on, no expected text. It can separate a glottal stop from q / g, and a
  from i / u, when the audio is clear.

What it CANNOT do, stated plainly: its Arabic training labels came from espeak's MSA phonemiser, so it
leans toward MSA-looking output on unclear audio; short unstressed vowels are its weakest phones; it
does not judge prosody, naturalness or "does this sound Egyptian". It is a second, independent listener
for specific targets (pronounce.py), not a verdict on the voice.

    python -m tools.masri_voice.phones <wav> ...      # print the IPA of each file
"""
import json
import os
import sys

import numpy as np

MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
os.environ.setdefault("HF_HOME", "G:/grad_all/.hf-cache")

_M = {}


def _load():
    if not _M:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2ForCTC
        path = snapshot_download(MODEL, allow_patterns=["*.json", "*.bin"])
        vocab = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
        _M["id2tok"] = {v: k for k, v in vocab.items()}
        _M["model"] = Wav2Vec2ForCTC.from_pretrained(path).eval()
        _M["torch"] = torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    return _M


def _resample(x, rate, to=16000):
    if rate == to:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(rate, to)
    return resample_poly(x, to // g, rate // g).astype(np.float32)


def ipa(samples, rate):
    """IPA phones (space-separated) the recogniser hears in float samples at `rate`."""
    m = _load()
    torch = m["torch"]
    x = _resample(np.asarray(samples, dtype=np.float32), rate)
    x = (x - x.mean()) / (x.std() + 1e-7)            # the feature extractor's normalisation
    with torch.no_grad():
        logits = m["model"](torch.from_numpy(x)[None]).logits[0]
    ids = logits.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev:
            t = m["id2tok"].get(i, "")
            if t not in ("<pad>", "<s>", "</s>", "<unk>", "|", ""):
                out.append(t)
        prev = i
    return " ".join(out)


def ipa_wav(wav_bytes):
    from tools.masri_voice import acoustics
    x, rate = acoustics.pcm(wav_bytes)
    return ipa(x, rate)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for f in sys.argv[1:]:
        print(os.path.basename(f), "|", ipa_wav(open(f, "rb").read()))
