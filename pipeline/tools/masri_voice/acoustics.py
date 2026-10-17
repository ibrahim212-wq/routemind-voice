# -*- coding: utf-8 -*-
"""Objective acoustics of a TTS take — what a clip bank needs from its voice, measured.

A bank is ~800 separate synthesis calls that must sound like ONE speaker: the same pitch, the same
pace, the same loudness, no stray breaths. So besides "does it sound Egyptian" (a listener's call —
the LLM judges proved position-biased and non-discriminating, see docs/egyptian-voice-2026-09.md),
a voice is scored on how CONSISTENT it is across takes and lines:
  • F0 median per clip (autocorrelation pitch on voiced frames) → spread across clips (Hz, cents)
  • speaking rate (letters of text per second of speech) → spread across clips
  • loudness (RMS dBFS of the voiced part) → spread across clips
  • dead air: leading / trailing silence, and any clip that is mostly silence or clips at 0 dBFS
Pure numpy on 24 kHz LINEAR16, no codecs.
"""
import io
import math
import wave

import numpy as np


def pcm(wav_bytes):
    """(samples float32 in [-1, 1], rate) from WAV bytes (Cloud TTS LINEAR16 carries a RIFF header)."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def frames(x, rate, win_s=0.04, hop_s=0.01):
    n, h = int(rate * win_s), int(rate * hop_s)
    if len(x) < n:
        return np.zeros((0, n), dtype=np.float32)
    idx = np.arange(0, len(x) - n, h)
    return np.stack([x[i:i + n] for i in idx])


def speech_bounds(x, rate, thresh_db=-40.0):
    """First / last sample of speech: 10 ms RMS above (peak RMS + thresh_db)."""
    f = frames(x, rate, 0.01, 0.01)
    if len(f) == 0:
        return 0, len(x)
    rms = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    on = np.where(db > db.max() + thresh_db)[0]
    if len(on) == 0:
        return 0, len(x)
    h = int(rate * 0.01)
    return int(on[0] * h), int(min(len(x), (on[-1] + 1) * h))


def f0_track(x, rate, fmin=70.0, fmax=400.0):
    f = frames(x, rate)
    if len(f) == 0:
        return np.array([])
    rms = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    voiced = rms > rms.max() * 0.12
    lo, hi = int(rate / fmax), int(rate / fmin)
    out = []
    win = np.hanning(f.shape[1]).astype(np.float32)
    for fr, v in zip(f, voiced):
        if not v:
            continue
        fr = (fr - fr.mean()) * win
        ac = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if ac[k] / ac[0] < 0.35:        # weak periodicity: unvoiced / noisy frame
            continue
        out.append(rate / k)
    return np.array(out)


def clip_stats(wav_bytes, text):
    x, rate = pcm(wav_bytes)
    a, b = speech_bounds(x, rate)
    sp = x[a:b]
    f0 = f0_track(sp, rate)
    letters = sum(1 for c in text if c.isalpha())
    dur = max(1e-3, (b - a) / rate)
    rms = float(np.sqrt((sp ** 2).mean() + 1e-12))
    return {
        "dur_s": round(len(x) / rate, 3),
        "speech_s": round(dur, 3),
        "lead_s": round(a / rate, 3),
        "trail_s": round((len(x) - b) / rate, 3),
        "f0_med": round(float(np.median(f0)), 1) if len(f0) else None,
        "f0_iqr": round(float(np.percentile(f0, 75) - np.percentile(f0, 25)), 1) if len(f0) else None,
        "rate_lps": round(letters / dur, 2),
        "rms_db": round(20 * math.log10(rms), 2),
        "peak": round(float(np.abs(x).max()), 3),
    }


def spread(stats):
    """Across-clip consistency of one voice."""
    f0 = [s["f0_med"] for s in stats if s["f0_med"]]
    rate = [s["rate_lps"] for s in stats]
    loud = [s["rms_db"] for s in stats]
    cents = [1200 * math.log2(v / np.median(f0)) for v in f0] if f0 else []
    return {
        "n": len(stats),
        "f0_median_hz": round(float(np.median(f0)), 1) if f0 else None,
        "f0_spread_cents": round(float(np.std(cents)), 1) if cents else None,
        "expressiveness_hz": round(float(np.median([s["f0_iqr"] for s in stats if s["f0_iqr"]])), 1) if f0 else None,
        "rate_lps": round(float(np.median(rate)), 2),
        "rate_spread_pct": round(100 * float(np.std(rate) / np.median(rate)), 1),
        "loud_spread_db": round(float(np.std(loud)), 2),
        "lead_s_max": round(max(s["lead_s"] for s in stats), 2),
        "trail_s_max": round(max(s["trail_s"] for s in stats), 2),
        "clipped": sum(1 for s in stats if s["peak"] >= 0.999),
    }
