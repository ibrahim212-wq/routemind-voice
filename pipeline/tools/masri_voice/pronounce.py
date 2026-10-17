# -*- coding: utf-8 -*-
"""How the bank's voice must SAY the phrasebook — the synthesis spelling and the pronunciation checks.

Two things live here, both build-time only (the app never sees either):

1. SAY — the text the voice READS for a segment. The segment itself (what the driver may see, what the
   clip is named after, what a fallback engine reads) stays normal Arabic; only the synthesis input is
   respelled. Zephyr reads «ق» as the MSA / Gulf q even when the style prompt asks for the Cairene
   glottal stop («الطريق» → ṭarīq, «النفق» → nafaq — Khaled heard it, the phone listener measured it),
   and it guesses MSA vowels on words it does not know («فوّت», «تلت», «المغرب», «أسرع»). So ق is
   written as the hamza Egyptians say (the way Egyptians write it in chat: «أدّام», «طريء»), and the
   words whose Egyptian vowels differ from MSA carry full diacritics. A word may list SEVERAL spellings:
   the build tries them in turn when the listener rejects a take.

2. TARGETS — what the blind phone listener (phones.py) must and must not hear. Phones are joined without
   spaces, so a target is a regex over e.g. "tiltsaːʕa". HARD targets gate a take; SOFT ones only rank
   takes. Every rule is scoped to segments containing the word.

What this can hear (and what it cannot) is stated in docs/egyptian-voice-2026-09.md §2.
"""
import re

# ── 1. the synthesis spelling ───────────────────────────────────────────────────────────────────────
# ق → ء / أ / ئ (the Cairene glottal stop), plus Egyptian vowels where MSA would be guessed.
SAY = {
    # ق — the road
    "الطريق": ["الطَّرِيء", "الطريء"],
    "طريق": ["طَرِيء", "طريء"],
    "طريقك": ["طَرِيئَك", "طريئك"],
    "طريقِك": ["طَرِيئِك", "طريئِك"],
    "النفق": ["النَّفَء", "النفء"],
    # ق — time
    "دقايق": ["دَآيِء", "دءايء"],            # «دَأايِء» came back «دقات»
    "دقيقة": ["دِئيئَة", "دئيئة"],
    "دقيقتين": ["دِئيئْتين", "دئيئتين"],
    "دلوقتي": ["دِلْوَءْتي", "دلوئتي"],
    "الوقت": ["الوَءْت", "الوئت"],
    "قبل": ["أَبْل", "ءبل"],
    # ق — place / stop
    "الوقفة": ["الوَءْفَة", "الوأفة"],
    "قدّام": ["أُدّام", "ادّام"],
    "قدّامك": ["أُدّامَك", "ادّامك"],
    "قدّامِك": ["أُدّامِك", "ادّامِك"],
    "فوق": ["فوء", "فوأ"],
    "قاعدين": ["أاعْدين", "آعدين"],
    "اقف": ["اُءَف", "أُءَف"],
    "اقفي": ["اُءْفي", "أُءْفي"],
    "مقفول": ["مَأْفول", "مءفول"],
    # ق — verbs
    "هيبقى": ["هَيِبْأَى", "هيبءى"],
    "وهتبقى": ["وِهَتِبْأَى", "وهتبءى"],
    "قرّب": ["أَرَّب", "ءرّب"],
    "قرّبنا": ["أَرَّبْنا", "ءرّبنا"],
    "قرّبت": ["أَرَّبْت", "ءرّبت"],
    "قرّبتي": ["أَرَّبْتي", "ءرّبتي"],
    "قد": ["أَدّ", "ءد"],
    "أقدر": ["أَءْدَر", "أءدر"],
    "تقدر": ["تِءْدَر", "تئدر"],
    "تقدري": ["تِءْدَري", "تئدري"],
    "هقولك": ["هَأولَك", "هءولك"],
    "هقولِك": ["هَأولِك", "هءولِك"],
    "أقولك": ["أَأولَك", "أءولك"],
    "أقولِك": ["أَأولِك", "أءولِك"],
    "لقيتلك": ["لِئيتْلَك", "لئيتلك"],
    "لقيتلِك": ["لِئيتْلِك", "لئيتلِك"],
    "ولقيتلك": ["وِلِئيتْلَك", "ولئيتلك"],
    "ولقيتلِك": ["وِلِئيتْلِك", "ولئيتلِك"],
    "هنلحق": ["هَنِلْحَء", "هنلحء"],
    "الباقي": ["الباأي", "الباءي"],
    "الأقل": ["الأَءَلّ", "الأءل"],
    # Egyptian vowels the voice got wrong (Khaled, 2026-09-24) — and a ث said as s
    "فوّت": ["فَوِّت", "فَوِّتّ"],
    "فوّتي": ["فَوِّتي", "فَوِّتّي"],
    "أسرع": ["أَسْرَع", "أسْرَع"],
    "تلت": ["تِلْت", "تِلت"],
    "وتلت": ["وِتِلْت", "وتِلت"],
    "المغرب": ["المَغْرِب", "المغْرِب"],
    "ملف": ["مَلَفّ", "مَلَف"],
    "الملف": ["المَلَفّ", "المَلَف"],
    # the U-turn (2026-10-11): Egyptian vowels on «لِفّ / وِارْجَع», the woman's form spelled out
    "لفّي": ["لِفّي", "لفّي"],
    "وارجعي": ["وِارْجَعي", "وارجعي"],
    "وارجع": ["وِارْجَع", "وارجع"],
    "لفّة": ["لَفّة", "لفّة"],
    "وترجعي": ["وِتِرْجَعي", "وترجعي"],
    "وترجع": ["وِتِرْجَع", "وترجع"],
    "ثانية": ["سانْيَة", "سانية"],
    "الدخلة": ["الدَّخْلَة", "الدخلة"],
}

_PUNCT = "،.!؟"


def _split(tok):
    """'طريقك،' → ('طريقك', '،')"""
    i = len(tok)
    while i > 0 and tok[i - 1] in _PUNCT:
        i -= 1
    return tok[:i], tok[i:]


def variants(segment):
    """How many synthesis spellings this segment has (the longest word list)."""
    n = 1
    for tok in segment.split():
        w, _ = _split(tok)
        if w in SAY:
            n = max(n, len(SAY[w]))
    return n


def say(segment, variant=0):
    """The text the voice reads for `segment` (spelling `variant` of every listed word, clamped)."""
    out = []
    for tok in segment.split():
        w, p = _split(tok)
        if w in SAY:
            opts = SAY[w]
            w = opts[min(variant, len(opts) - 1)]
        out.append(w + p)
    return " ".join(out)


def all_says(segment):
    return {say(segment, v) for v in range(variants(segment))}


# ── 2. what the blind listener must / must not hear ─────────────────────────────────────────────────
V_A = "[aæɐɑeɛə]"          # an a-quality vowel as the recogniser writes it (Cairene a is fronted)
V_I = "[iɪeɛəy]"
V_U = "[uʊoɔ]"

# word -> (hard, [(kind, regex, message)]) ; kind "need" = must match somewhere, "ban" = must not
TARGETS = {
    "فوّت": (True, [("need", r"f" + V_A + r"?w", "«فَوِّت»: the w of fawwit"),
                     ("ban", r"f" + V_U + r"t", "«فوّت» said as «فُت»")]),
    "فوّتي": (True, [("need", r"f" + V_A + r"?w", "«فَوِّتي»: the w of fawwiti")]),
    "أسرع": (True, [("need", r"[sz]ː?[rɾɹ]", "«أسرع»: the s-r cluster")]),
    "تلت": (True, [("need", r"t" + V_I + r"l", "«تِلت»: tilt"),
                    ("ban", r"t" + V_U + r"lt", "«تلت» said as «تُلت»")]),
    "وتلت": (True, [("need", r"t" + V_I + r"l", "«وتِلت»: tilt")]),
    "المغرب": (True, [("need", r"m" + V_A + r"[ɣgɡxʁq]", "«المَغْرِب»: ma-"),
                       ("ban", r"m" + V_U + r"[ɣgɡxʁq]", "«المغرب» said «مُغرب»"),
                       ("ban", r"[ɣgɡxʁq]r[aɑɐæ]b", "«المغرب» said «مغرَب» (Khaled: -rib)")]),
    # the ban is anchored on the f: «اعمل ملف» is «ʕamil malaff» — the -mil of اعمل is not the word
    "ملف": (True, [("need", r"m" + V_A + r"l" + V_A + r"?f", "«مَلَفّ»: malaff"),
                    ("ban", r"m[iɪ]l" + V_A + r"?f", "«مِلَف» (a file)")]),
    "الملف": (True, [("need", r"m" + V_A + r"l" + V_A + r"?f", "«المَلَفّ»: malaff"),
                      ("ban", r"m[iɪ]l" + V_A + r"?f", "«المِلَف» (a file)")]),
    # the hundreds are Egyptian u-vowels (rubʕumiyya, not MSA arbaʕumiʔa) — soft: measured, ranked
    "تلتمية": (False, [("need", r"t" + V_U + r"lt", "«تُلتُمية»")]),
    "ربعمية": (False, [("need", r"r" + V_U + r"b", "«رُبعُمية»")]),
    "خمسمية": (False, [("need", r"[xh]" + V_U + r"m", "«خُمسُمية»")]),
    "ستمية": (False, [("need", r"s" + V_U + r"t", "«سُتُّمية»")]),
    "سبعمية": (False, [("need", r"s" + V_U + r"b", "«سُبعُمية»")]),
    "تمنمية": (False, [("need", r"t" + V_U + r"m", "«تُمنُمية»")]),
    "تسعمية": (False, [("need", r"t" + V_U + r"s", "«تُسعُمية»")]),
}


def words(segment):
    return [_split(t)[0] for t in segment.split()]


def check(segment, ipa_str):
    """→ (hard problems, soft problems) for what the blind listener heard in this segment's clip."""
    toks = ipa_str.split()
    joined = "".join(toks)
    hard, soft = [], []
    # ق is a glottal stop in Cairo: never q (or the uvular ɢ)
    n_q = sum(1 for t in toks if t in ("q", "ɢ", "qʰ"))
    if "ق" in segment and n_q > segment.count("غ"):      # the recogniser writes a ɣ as q now and then
        hard.append("ق heard as q")
    # ج is a hard g in Cairo: never dʒ / ʒ
    if "ج" in segment and any(t in ("dʒ", "ʒ", "d͡ʒ") for t in toks):
        hard.append("ج heard as dʒ")
    # ث / ذ are t / d (or s / z) in Cairo: never θ / ð
    if ("ث" in segment or "ذ" in segment) and any(t in ("θ", "ð") for t in toks):
        hard.append("ث/ذ heard as θ/ð")
    # a word whose strong consonants did not come through (garbled / another word): gates the take
    # (the build then records more takes and keeps the best); see garbled_words for what it cannot see
    for w, c, st in garbles(segment, ipa_str):
        hard.append("«%s» garbled (consonants %.1f/%d)" % (w, c, st))
    ws = set(words(segment))
    for w, (is_hard, rules) in TARGETS.items():
        if w not in ws:
            continue
        for kind, rx, msg in rules:
            hit = re.search(rx, joined) is not None
            if (kind == "need" and not hit) or (kind == "ban" and hit):
                (hard if is_hard else soft).append(msg)
    return hard, soft


def targeted(segment):
    """Does the pronunciation check have anything specific to hear in this segment?"""
    ws = set(words(segment))
    return ("ق" in segment or "ج" in segment or "ث" in segment or "ذ" in segment
            or any(w in ws for w in TARGETS) or any(w in ws for w in SAY))


# ── the word listener's spelling fold ───────────────────────────────────────────────────────────────
_DIAC = re.compile("[%s-%s%s%s]" % (chr(0x64B), chr(0x652), chr(0x670), chr(0x640)))


def fold(s):
    """One spelling for a word however it was transcribed: Egyptian and MSA spellings of the same sound
    (ق / ء / أ / ئ / ؤ — the glottal stop; ث→ت, ذ→د, ظ→ض), and ة / ى, collapse; diacritics go."""
    s = _DIAC.sub("", s)
    for a in "قءأإآئؤ":
        s = s.replace(a, "ا")
    return (s.replace("ث", "ت").replace("ذ", "د").replace("ظ", "ض")
             .replace("ة", "ه").replace("ى", "ي"))


def missing_words(segment, heard):
    """Words of the segment the transcript does not contain (after the fold) — a word the voice said
    as a DIFFERENT word (فطار → فطور) or swallowed. Two-letter particles are not judged."""
    h = fold(heard).replace(" ", "")
    for p in _PUNCT + ".,!?":
        h = h.replace(p, "")
    out = []
    for w in words(segment):
        f = fold(w)
        if len(f) >= 3 and f not in h:
            out.append(w)
    return out


# ── 3. the consonant skeleton — a GARBLED word, whatever the word ───────────────────────────────────
# The word listener (an ASR) repairs what it hears into the word it expects: it wrote «وارجعي» for the
# take Khaled heard as «والبع», and «الفطار» for «فطور». The phone listener does not repair. Every
# word's EXPECTED Cairene consonants (ق = ʔ, ج = g, ث = t, ذ = d, the article's l assimilated before a
# sun letter) are aligned with the consonants HEARD; a word that loses or changes its strong
# consonants is reported. Glottal / pharyngeal / glide consonants are weak (the recogniser drops them
# freely), vowels are ignored — this sees garbling and swapped consonants, never a wrong vowel.
_LETTER = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "g", "ح": "ħ", "خ": "x", "د": "d", "ذ": "d", "ر": "r", "ز": "z",
    "س": "s", "ش": "ʃ", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "ʕ", "غ": "ɣ", "ف": "f", "ق": "ʔ",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "j", "ء": "ʔ", "أ": "ʔ", "إ": "ʔ",
    "آ": "ʔ", "ئ": "ʔ", "ؤ": "ʔ",
}
_SUN = set("تثدذرزسشصضطظلن")
WEAK = set("ʔhʕħwj")
_PHONE = {
    "b": "b", "p": "b", "t": "t", "tˤ": "t", "t̪": "t", "d": "d", "dˤ": "d", "d̪": "d", "r": "r", "ɾ": "r",
    "ɹ": "r", "r̩": "r", "z": "z", "zˤ": "z", "s": "s", "sˤ": "s", "ʃ": "ʃ", "f": "f", "v": "f", "k": "k",
    "g": "g", "ɡ": "g", "q": "q", "ɢ": "q", "x": "x", "χ": "x", "ɣ": "ɣ", "ʁ": "ɣ", "ħ": "ħ", "ʕ": "ʕ",
    "h": "h", "ɦ": "h", "ʔ": "ʔ", "l": "l", "ɫ": "l", "m": "m", "n": "n", "ŋ": "n", "ɲ": "n", "w": "w",
    "j": "j", "θ": "θ", "ð": "ð", "dʒ": "dʒ", "ʒ": "dʒ", "d͡ʒ": "dʒ", "ç": "ʃ", "ʂ": "ʃ",
}
_CLOSE = {frozenset(p) for p in [("r", "l"), ("t", "d"), ("s", "z"), ("k", "g"), ("x", "ɣ"), ("x", "h"),
                                 ("ħ", "h"), ("ʕ", "ʔ"), ("ʔ", "h"), ("q", "ɣ"), ("q", "k"), ("q", "ʔ"),
                                 ("θ", "t"), ("θ", "s"), ("ð", "d"), ("ð", "z"), ("m", "n"), ("g", "dʒ"),
                                 ("ɣ", "g"), ("ʃ", "s"), ("b", "f"), ("w", "b"), ("ħ", "ʕ"), ("x", "k")]}


def expected_consonants(word):
    w = _DIAC.sub("", word)
    out = []
    i = 0
    if w.startswith("ال") and len(w) > 2:
        out.append("ʔ")                       # the article's hamzat al-wasl, weak
        if w[2] not in _SUN:
            out.append("l")
        i = 2
    elif len(w) > 1 and w[0] == "و" and w[1:].startswith("ال"):
        out.append("w")
        if len(w) > 3 and w[3] not in _SUN:
            out.append("l")
        i = 3
    for ch in w[i:]:
        c = _LETTER.get(ch)
        if c:
            out.append(c)
    return out


def heard_consonants(ipa_str):
    out = []
    for t in ipa_str.split():
        if t == "ts":
            out += ["t", "s"]
        elif t == "tʃ":
            out += ["t", "ʃ"]
        elif t in _PHONE:
            out.append(_PHONE[t])
    return out


def _sub(a, b):
    if a == b:
        return 0.0
    if frozenset((a, b)) in _CLOSE:
        return 0.4
    return 1.0


def garbled_words(segment, ipa_str):
    """→ [(word, cost, strong)] for words whose strong consonants did not come through."""
    ws = [w for w in words(segment) if w]
    exp, owner = [], []
    for k, w in enumerate(ws):
        for c in expected_consonants(w):
            exp.append(c)
            owner.append(k)
    hrd = heard_consonants(ipa_str)
    n, m = len(exp), len(hrd)
    INS = 0.3
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + (0.2 if exp[i - 1] in WEAK else 1.0)
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + INS
        bt[0][j] = "I"
    for i in range(1, n + 1):
        de = 0.2 if exp[i - 1] in WEAK else 1.0
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (_sub(exp[i - 1], hrd[j - 1]) * (0.3 if exp[i - 1] in WEAK else 1.0))
            dele = dp[i - 1][j] + de
            ins = dp[i][j - 1] + INS
            best = min(sub, dele, ins)
            dp[i][j] = best
            bt[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    cost = [0.0] * len(ws)
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "S":
            cost[owner[i - 1]] += _sub(exp[i - 1], hrd[j - 1]) * (0.3 if exp[i - 1] in WEAK else 1.0)
            i, j = i - 1, j - 1
        elif op == "D":
            cost[owner[i - 1]] += 0.2 if exp[i - 1] in WEAK else 1.0
            i -= 1
        else:
            j -= 1
    out = []
    for k, w in enumerate(ws):
        strong = sum(1 for c in expected_consonants(w) if c not in WEAK)
        out.append((w, round(cost[k], 2), strong))
    return out


GARBLE_MIN_COST = 1.4      # calibrated on the bank (tools/masri_voice/pronounce.py --calibrate)
GARBLE_MIN_RATIO = 0.6


def garbles(segment, ipa_str):
    return [(w, c, st) for w, c, st in garbled_words(segment, ipa_str)
            if st >= 2 and c >= GARBLE_MIN_COST and c / st >= GARBLE_MIN_RATIO]


# ── 4. words the TRANSCRIBER cannot be trusted on — the phones decide ───────────────────────────────
# The in-line check (build_bank.context_wav) asks a transcriber whether the direction word / ordinal
# was heard. Two classes it gets wrong on audio the phone listener hears clearly (2026-09-24, six
# listens each): Cairo's «ملف» (it writes «يوتيرن», «يمين» — it does not know the word), and the rare
# ordinals (it pulls «تامن» to «تاني» / «تالت»). For these a "lost" verdict is overruled when the
# phone listener heard the word's own consonant-vowel shape.
PHONE_HEARD = {
    "ملف": r"m" + V_A + r"?l" + V_A + r"?f",
    "أول": r"[ʔ]?" + V_A + r"w+" + V_A + r"?l",
    "تاني": r"t" + V_A + r"ː?n[iɪeɛ]",
    "تالت": r"t" + V_A + r"ː?l" + V_A + r"?t",
    "رابع": r"r" + V_A + r"ː?b",
    "خامس": r"[xh]" + V_A + r"ː?m" + V_A + r"?s",
    "سادس": r"s" + V_A + r"ː?d",
    "سابع": r"s" + V_A + r"ː?b",
    "تامن": r"t" + V_A + r"ː?m" + V_A + r"?n",
}


def heard_by_phones(word, ipa_str):
    rx = PHONE_HEARD.get(word)
    return rx is not None and re.search(rx, "".join((ipa_str or "").split())) is not None
