#!/usr/bin/env python3
"""
Script-aware post-processing, linguistic normalization, and degeneration/hallucination detection for Indic OCR.
Handles Sanskrit-specific corrections, Visarga ASCII colons, Devanagari nukta stripping,
English ligature restorations, and server-side repetition/hallucination detection.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

# Sanskrit has no nuktas. Map nukta characters to base consonants.
SANSKRIT_NUKTA_MAP = {
    'ड़': 'ड',
    'ढ़': 'ढ',
    'क़': 'क',
    'ख़': 'ख',
    'ग़': 'ग',
    'ज़': 'ज',
    'फ़': 'फ',
    '\u093C': '',  # Nukta diacritic mark itself
}

# Common English ligature drop corrections (e.g. ffi, ff, fl)
ENGLISH_LIGATURE_REPLACEMENTS = [
    (re.compile(r'\bDificulty\b', re.IGNORECASE), lambda m: 'Difficulty' if m.group(0)[0].isupper() else 'difficulty'),
    (re.compile(r'\bStifness\b', re.IGNORECASE), lambda m: 'Stiffness' if m.group(0)[0].isupper() else 'stiffness'),
    (re.compile(r'\bRefection\b', re.IGNORECASE), lambda m: 'Reflection' if m.group(0)[0].isupper() else 'reflection'),
    (re.compile(r'\bAfect\b', re.IGNORECASE), lambda m: 'Affect' if m.group(0)[0].isupper() else 'affect'),
    (re.compile(r'\bEfect\b', re.IGNORECASE), lambda m: 'Effect' if m.group(0)[0].isupper() else 'effect'),
    (re.compile(r'\bInfamm', re.IGNORECASE), lambda m: 'Inflamm' if m.group(0)[0].isupper() else 'inflamm'),
    (re.compile(r'\bSwel\b', re.IGNORECASE), lambda m: 'Swell' if m.group(0)[0].isupper() else 'swell'),
]

# Visarga (ः) commonly confused as ASCII colon (:) when directly following Devanagari text
VISARGA_COLON_RE = re.compile(r'(?<=[\u0900-\u097F]):(?!\d)')

# Conversational Hindi phrases that DO NOT exist in classical Sanskrit texts
HINDI_CONVERSATIONAL_MARKERS = [
    'के लिए',
    'प्रतिनिधि के लिए',
    'प्रतिनिधि',
    'करता है',
    'करते हैं',
    'होता है',
    'होती है',
    'होते हैं',
    'होने के',
    'सकता है',
    'सकते हैं',
    'सकती है',
    'किया जाता',
    'दिया जाता',
    'के साथ',
    'की वजह',
    'की ओर',
    'नहीं है',
    'नहीं होता',
    'प्रत्याशन के लिए',
]

# Ayurvedic Sanskrit minimal pair corrections (ब vs व confusion)
SANSKRIT_AYURVEDIC_DOMAIN_REPLACEMENTS = [
    # व्रण (wound/ulcer) - ubiquitous in Yogaratnakara, Sushruta, Charaka
    (re.compile(r'\bब्रण\b'), 'व्रण'),
    (re.compile(r'\bब्रणा'), 'व्रणा'),
    (re.compile(r'\bब्रणी'), 'व्रणी'),
    (re.compile(r'\bब्रणे'), 'व्रणे'),
    (re.compile(r'\bब्रणै'), 'व्रणै'),
    (re.compile(r'\bब्रणरोपण'), 'व्रणरोपण'),
    (re.compile(r'\bब्रणशोथ'), 'व्रणशोथ'),
    (re.compile(r'\bदुष्टब्रण'), 'दुष्टव्रण'),
    # वात (Vata dosha) in typical compounds
    (re.compile(r'\bबातव्याधि'), 'वातव्याधि'),
    (re.compile(r'\bबातपित्त'), 'वातपित्त'),
    (re.compile(r'\bबातश्लेष्म'), 'वातश्लेष्म'),
    (re.compile(r'\bबातज'), 'वातज'),
    (re.compile(r'\bबातघ्न'), 'वातघ्न'),
    (re.compile(r'\bबातकोप'), 'वातकोप'),
    (re.compile(r'\bबातप्रकोप'), 'वातप्रकोप'),
    # व्याधि (disease)
    (re.compile(r'\bब्याधि'), 'व्याधि'),
    (re.compile(r'\bब्याधय'), 'व्याधय'),
    # वृद्धि (hernia / enlargement)
    (re.compile(r'\bबृद्धि\b'), 'वृद्धि'),
    # विपाक (post-digestive effect)
    (re.compile(r'\bबिपाक'), 'विपाक'),
    # विरेचन (purgation)
    (re.compile(r'\bबिरेचन'), 'विरेचन'),
    # वीर्य (potency)
    (re.compile(r'\bबीर्य'), 'वीर्य'),
]


def postprocess_text(text: str, script_hint: str = "auto") -> str:
    """
    Apply linguistic and script-specific postprocessing to OCR output.
    
    Args:
        text: The extracted markdown or text.
        script_hint: 'auto', 'sanskrit', 'devanagari-sanskrit', 'hindi', or 'english'.
    """
    if not text:
        return text

    # 1. Normalize Unicode (NFC)
    text = unicodedata.normalize('NFC', text)

    # 2. Visarga repair: Convert ASCII colon ':' directly following Devanagari letters to Visarga 'ः'
    text = VISARGA_COLON_RE.sub('ः', text)

    # 3. Sanskrit-specific normalizations
    script_lower = (script_hint or "auto").lower()
    is_sanskrit = ("sanskrit" in script_lower) or ("devanagari-sanskrit" in script_lower)

    if is_sanskrit:
        # Strip all nuktas (Sanskrit classical orthography has no nuktas)
        for nukta_char, replacement in SANSKRIT_NUKTA_MAP.items():
            text = text.replace(nukta_char, replacement)

        # Fix specific classical Sanskrit medical/philosophical domain minimal pairs
        for pattern, replacement in SANSKRIT_AYURVEDIC_DOMAIN_REPLACEMENTS:
            text = pattern.sub(replacement, text)

    # 4. English ligature restoration (fixes ffi, ff, fl dropped characters)
    for pattern, repl in ENGLISH_LIGATURE_REPLACEMENTS:
        text = pattern.sub(repl, text)

    return text


def detect_degeneration_and_hallucination(
    text: str,
    script_hint: str = "auto",
    token_count: int = 0,
) -> Dict[str, Any]:
    """
    Server-side detector for model degeneration, repetition loops, and language hallucinations.
    
    Provides 'fail loudly' semantics when the vision decoder loses visual grounding
    and enters an autoregressive loop.
    
    Returns:
        dict containing:
            - degenerated: bool
            - hallucination_flag: bool
            - repetition_ratio: float (0.0 to 1.0)
            - max_repeat_count: int
            - detected_language_violations: list of str
            - confidence_modifier: float
            - degeneration_reason: Optional[str]
    """
    if not text or not text.strip():
        return {
            "degenerated": False,
            "hallucination_flag": False,
            "repetition_ratio": 0.0,
            "max_repeat_count": 0,
            "detected_language_violations": [],
            "confidence_modifier": 1.0,
            "degeneration_reason": None,
        }

    # Extract alphanumeric/Devanagari word tokens
    words = re.findall(r'[\w\u0900-\u097F]+', text)
    total_words = len(words)
    script_lower = (script_hint or "auto").lower()
    is_sanskrit = ("sanskrit" in script_lower) or ("devanagari-sanskrit" in script_lower)

    violations: List[str] = []
    max_repeat_count = 0
    repetition_ratio = 0.0

    # 1. Check for n-gram repetition loops (phrase cycles of length 2 to 5 words)
    if total_words >= 6:
        # 3-gram analysis
        trigrams = [' '.join(words[i:i + 3]) for i in range(total_words - 2)]
        if trigrams:
            counts = Counter(trigrams)
            top_ngram, top_cnt = counts.most_common(1)[0]
            max_repeat_count = top_cnt
            unique_ratio = len(counts) / len(trigrams)
            repetition_ratio = round(1.0 - unique_ratio, 3)

            # If any 3-gram repeats 3 or more times, or repetition ratio > 0.35 with total words >= 15
            if top_cnt >= 3:
                violations.append(f"Repetition loop: 3-gram '{top_ngram}' repeated {top_cnt} times")
            elif repetition_ratio > 0.40 and total_words >= 20:
                violations.append(f"High repetition ratio ({repetition_ratio:.2f}) across {total_words} words")

        # 2-gram check for very short loops
        bigrams = [' '.join(words[i:i + 2]) for i in range(total_words - 1)]
        if bigrams:
            b_counts = Counter(bigrams)
            top_b, top_b_cnt = b_counts.most_common(1)[0]
            if top_b_cnt >= 4:
                violations.append(f"Repetition loop: bigram '{top_b}' repeated {top_b_cnt} times")

    # 2. Sanskrit script constraint: check for modern conversational Hindi intrusions
    found_hindi = []
    if is_sanskrit:
        for marker in HINDI_CONVERSATIONAL_MARKERS:
            if marker in text:
                found_hindi.append(marker)

        if found_hindi:
            violations.append(f"Hindi language intrusion under Sanskrit constraint: {', '.join(found_hindi[:4])}")

        # Check for Hindi nuktas which are strictly absent from classical Sanskrit
        # Precomposed Devanagari nuktas (\u0958-\u095F) or the combining nukta mark (\u093C)
        nukta_matches = re.findall(r'[\u0958-\u095F]|\u093C', text)
        if len(nukta_matches) >= 3:
            violations.append(f"Found {len(nukta_matches)} Hindi nukta characters in Sanskrit mode")

    # 3. Aggregate assessment
    is_degenerated = len(violations) > 0
    hallucination_flag = is_degenerated

    # Calibrate confidence: drop to 0.20-0.30 if hallucinated so client thresholding works
    if is_degenerated:
        confidence = 0.20 if (max_repeat_count >= 3 or len(found_hindi) >= 2) else 0.35
        reason = " | ".join(violations)
    else:
        confidence = 0.95
        reason = None

    return {
        "degenerated": is_degenerated,
        "hallucination_flag": hallucination_flag,
        "repetition_ratio": repetition_ratio,
        "max_repeat_count": max_repeat_count,
        "detected_language_violations": found_hindi,
        "confidence": confidence,
        "degeneration_reason": reason,
    }


def check_token_repetition_loop(gen_ids: List[int], min_cycle: int = 3, max_cycle: int = 25, min_repeats: int = 3) -> Tuple[bool, int]:
    """
    Real-time generation check for token-level repetitive attractor loops.
    
    Checks if the tail of the generated tokens matches a cycle of length [min_cycle, max_cycle]
    repeated consecutively at least `min_repeats` times.
    """
    n = len(gen_ids)
    for c in range(min_cycle, min(max_cycle + 1, n // min_repeats + 1)):
        pattern = gen_ids[-c:]
        is_loop = True
        for r in range(1, min_repeats):
            start = n - (r + 1) * c
            end = n - r * c
            if gen_ids[start:end] != pattern:
                is_loop = False
                break
        if is_loop:
            return True, c
    return False, 0
