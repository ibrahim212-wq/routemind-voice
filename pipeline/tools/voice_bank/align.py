# -*- coding: utf-8 -*-
"""The TRANSCRIPT CUT — a batched Gemini take cut by what was SAID, when the cut by pauses cannot be trusted.

The bank asks for K segments in one request ("a clear one-second pause after every line") and cuts the answer with
audition.split_on_silence + the aligned cut (each piece matched to its line's letters). That holds while the voice
keeps to the script; it does not when the model ADDS lines of its own (Despina's 17-line audition request came back
105 s long for 73 s of text) or drops one — every piece after the slip is then another line's audio, and the old
answer was to fail the request (its K segments unheard) and ask again in two halves: two of the day's ten requests.

Here the take is cut into ISLANDS (speech runs between pauses ≥ ISLAND_GAP_S), each island is read back by the Azure
speech-to-text (tools/voice_bank/stt.py — no Gemini quota), and a dynamic programme aligns the K lines to the islands
in order: a line takes 1..MAX_ISLANDS consecutive islands (cost = the bank's Arabic CER of their transcript against
the line), an island may be left out (speech the script does not have: SKIP_ISLAND) and a line may be missing
(SKIP_LINE). A line whose best match reads worse than MATCH_MAX gets no piece — it is rendered again, never guessed.
The pieces then pass every gate a piece of a clean take passes (pace, the in-line read-back, the phone listener).

`suspicious(wav, k)` is the cheap trigger: the number of long pauses (≥ LONG_PAUSE_S) against the k − 1 the prompt
asks for. Measured on the audition's takes: Vindemiatrix 17 lines → 17 long pauses, the 12-segment batch → 11 (both
clean); Despina 17 lines → 28 (the added lines), her 15-line retake → 5 (pauses of ~0.5 s).
"""
import io
import math

import numpy as np

from tools.masri_voice import acoustics
from tools.masri_voice import build_bank
from tools.voice_audition import audition as A

ISLAND_GAP_S = 0.30
LONG_PAUSE_S = 0.60
MAX_ISLANDS = 4
# Costs: leaving out an island the script does not have is CHEAP (the model ad-libs whole sentences between lines —
# Despina's 17-line audition request: ~20 islands of her own), a line that is not there costs a whole line, and a
# match worse than MATCH_MAX is never forced (it costs more than leaving the line out: the line is rendered again).
SKIP_ISLAND = 0.2
SKIP_LINE = 1.0
MATCH_MAX = 0.45
BAD_MATCH = SKIP_LINE + 0.1
PAD_S = 0.08


def _frames(wav_b):
    x, rate = acoustics.pcm(wav_b)
    hop = int(rate * 0.01)
    n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return x, rate, hop, db < db.max() - 40.0


def pauses(wav_b):
    """[(start_frame, end_frame)] of the silences inside the speech (10 ms frames)."""
    x, rate, hop, silent = _frames(wav_b)
    voiced = np.where(~silent)[0]
    if len(voiced) == 0:
        return [], 0, 0, x, rate, hop
    first, last = int(voiced[0]), int(voiced[-1])
    out, i = [], first
    while i <= last:
        if silent[i]:
            j = i
            while j <= last and silent[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out, first, last + 1, x, rate, hop


def suspicious(wav_b, k):
    """True when the take's long pauses do not match the k - 1 the batch prompt asks for."""
    if k < 3:
        return False
    ps = pauses(wav_b)[0]
    n_long = sum(1 for a, b in ps if (b - a) * 0.01 >= LONG_PAUSE_S)
    return abs(n_long - (k - 1)) > max(2, int(math.ceil(0.15 * (k - 1))))


def islands(wav_b):
    """[(start_sample, end_sample)] speech runs separated by pauses ≥ ISLAND_GAP_S, with the gap midpoints as the
    boundaries a piece is cut at."""
    ps, first, end, x, rate, hop = pauses(wav_b)
    gaps = [(a, b) for a, b in ps if (b - a) * 0.01 >= ISLAND_GAP_S]
    cuts = [first] + [v for a, b in gaps for v in (a, b)] + [end]
    runs = [(cuts[i] * hop, cuts[i + 1] * hop) for i in range(0, len(cuts), 2)]
    mids = [0] + [((a + b) // 2) * hop for a, b in gaps] + [len(x)]
    return runs, mids, x, rate


def transcript_cut(wav_b, lines, lang="ar"):
    """→ (pieces: [wav bytes | None] one per line, report). `lines` = the PLAIN text of each line (what a listener
    writes, not the SAY respelling). Raises stt.Exhausted / stt.SttError when the read-back cannot run."""
    from tools.voice_bank import stt as STT
    runs, mids, x, rate = islands(wav_b)
    pad = int(rate * PAD_S)
    heard = []
    for a, b in runs:
        seg = x[max(0, a - pad):min(len(x), b + pad)]
        if len(seg) / float(rate) > STT.MAX_AUDIO_S:
            raise STT.SttError("an island of %.0f s (no pauses): cannot read it back" % (len(seg) / float(rate)))
        r = STT.recognize(build_bank.wav_bytes(seg, rate), lang)
        heard.append((r.get("lexical") or r.get("display") or "") if r.get("status") == "Success" else "")
    k, m = len(lines), len(runs)
    INF = float("inf")
    cost = {}

    def match(i, j, n):
        key = (i, j, n)
        if key not in cost:
            h = " ".join(t for t in heard[j:j + n] if t)
            c = min(1.0, A.cer_ar(lines[i], h)) if h else 1.0
            cost[key] = c if c <= MATCH_MAX else BAD_MATCH
        return cost[key]

    dp = [[INF] * (m + 1) for _ in range(k + 1)]
    back = [[None] * (m + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for i in range(k + 1):
        for j in range(m + 1):
            v = dp[i][j]
            if v == INF:
                continue
            if j < m and v + SKIP_ISLAND < dp[i][j + 1]:            # speech the script does not have
                dp[i][j + 1], back[i][j + 1] = v + SKIP_ISLAND, ("skip_island", i, j)
            if i < k and v + SKIP_LINE < dp[i + 1][j]:              # a line the take does not have
                dp[i + 1][j], back[i + 1][j] = v + SKIP_LINE, ("skip_line", i, j)
            if i < k:
                for n in range(1, MAX_ISLANDS + 1):
                    if j + n > m:
                        break
                    c = v + match(i, j, n)
                    if c < dp[i + 1][j + n]:
                        dp[i + 1][j + n], back[i + 1][j + n] = c, ("match", i, j, n)
    pieces, rep = [None] * k, {"islands": m, "lines": k, "matched": 0, "added_islands": 0, "missing": 0,
                               "cer": [None] * k, "heard": [None] * k}
    i, j = k, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        if step[0] == "skip_island":
            rep["added_islands"] += 1
            i, j = step[1], step[2]
        elif step[0] == "skip_line":
            rep["missing"] += 1
            i, j = step[1], step[2]
        else:
            _, li, j0, n = step
            c = match(li, j0, n)
            h = " ".join(t for t in heard[j0:j0 + n] if t)
            rep["cer"][li] = round(min(1.0, A.cer_ar(lines[li], h)) if h else 1.0, 3)
            rep["heard"][li] = " ".join(t for t in heard[j0:j0 + n] if t)
            if c <= MATCH_MAX:
                a, b = mids[j0], mids[j0 + n]
                pieces[li] = build_bank.wav_bytes(x[a:b], rate)
                rep["matched"] += 1
            i, j = li, j0
    return pieces, rep
