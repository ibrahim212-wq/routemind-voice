# -*- coding: utf-8 -*-
"""MP3 at the FRAME level — the same operations the phone does when it composes a spoken line.

The clip bank ships every clip as bare MPEG Layer III frames (no ID3, no Xing/Info/VBRI header
frame, constant bitrate, one format for the whole bank). A spoken line is then the byte
concatenation of its clips with SILENT FRAMES between them — no decoder, no encoder, no
resampling on the phone, and the players the app already has (MediaPlayer / AVAudioPlayer) play
the result as one clip. VoiceBank.kt / VoiceBank.swift transcribe `concat` and `silent_frame`.

Why the header frames must go: a Xing/Info frame carries the frame count of ITS file; a player
that reads it would stop after the first clip of a concatenation.
"""

BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],   # MPEG-1 L3
            2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]}       # MPEG-2/2.5 L3
RATES = {3: {0: 44100, 1: 48000, 2: 32000}, 2: {0: 22050, 1: 24000, 2: 16000}, 0: {0: 11025, 1: 12000, 2: 8000}}


def frame_info(b, i):
    """(length, samples, rate, bitrate) of the Layer III frame at b[i], or None."""
    if i + 4 > len(b) or b[i] != 0xFF or (b[i + 1] & 0xE0) != 0xE0:
        return None
    ver = (b[i + 1] >> 3) & 3
    layer = (b[i + 1] >> 1) & 3
    bri = (b[i + 2] >> 4) & 15
    sri = (b[i + 2] >> 2) & 3
    pad = (b[i + 2] >> 1) & 1
    if ver == 1 or layer != 1 or bri in (0, 15) or sri == 3:
        return None
    br = BITRATES[1 if ver == 3 else 2][bri] * 1000
    rate = RATES[ver][sri]
    spf = 1152 if ver == 3 else 576
    length = (144 if ver == 3 else 72) * br // rate + pad
    return length, spf, rate, br


def _is_header_frame(b, i, length):
    body = b[i + 4:i + min(length, 64)]
    return b"Xing" in body or b"Info" in body or b"VBRI" in body


def frames(b):
    """[(offset, length, samples, rate)] of the audio frames, skipping tags and header frames."""
    out = []
    i = 0
    if b[:3] == b"ID3":
        size = (b[6] & 0x7F) << 21 | (b[7] & 0x7F) << 14 | (b[8] & 0x7F) << 7 | (b[9] & 0x7F)
        i = 10 + size + (10 if b[5] & 0x10 else 0)
    end = len(b) - (128 if len(b) >= 128 and b[-128:-125] == b"TAG" else 0)
    while i + 4 <= end:
        fi = frame_info(b, i)
        if not fi or i + fi[0] > end:
            i += 1
            continue
        length, spf, rate, _ = fi
        if not _is_header_frame(b, i, length):
            out.append((i, length, spf, rate))
        i += length
    return out


def strip(b):
    """Bare frames only."""
    return b"".join(b[o:o + n] for o, n, _, _ in frames(b))


def duration_s(b):
    return sum(spf / rate for _, _, spf, rate in frames(b))


def silent_frame(header4):
    """A frame that decodes to silence: the clip's own header (padding cleared) with zeroed
    side info and main data — main_data_begin 0, part2_3_length 0, global_gain 0."""
    h = bytearray(header4)
    h[2] &= ~0x02 & 0xFF
    length, _, _, _ = frame_info(bytes(h), 0)
    return bytes(h) + bytes(length - 4)


def silence(header4, ms):
    """Whole silent frames covering at least `ms` milliseconds."""
    length, spf, rate, _ = frame_info(bytes(bytearray(header4[:2]) + bytes([header4[2] & ~0x02 & 0xFF, header4[3]])), 0)
    n = max(0, -(-int(ms) * rate // (1000 * spf)))
    return silent_frame(header4) * n


def concat(clips, gap_ms=160, gaps=None):
    """Join bare-frame clips with silence between them. `gaps[k]` is the pause after clip k."""
    clips = [strip(c) for c in clips if c]
    if not clips:
        return b""
    header = clips[0][:4]
    out = bytearray()
    for k, c in enumerate(clips):
        out += c
        if k < len(clips) - 1:
            out += silence(header, gaps[k] if gaps else gap_ms)
    return bytes(out)
