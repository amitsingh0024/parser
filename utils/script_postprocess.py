#!/usr/bin/env python3
"""
Script-aware post-processing and linguistic normalization for Indic OCR output.
Handles Sanskrit-specific corrections, Visarga ASCII colons, Devanagari nukta stripping,
and English ligature restorations based on empirical benchmark feedback.
"""

import re
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
    # e.g., 'नम:' -> 'नमः', 'रोग:' -> 'रोगः', 'पुत्र:' -> 'पुत्रः'
    text = VISARGA_COLON_RE.sub('ः', text)

    # 3. Sanskrit-specific normalizations
    script_lower = (script_hint or "auto").lower()
    is_sanskrit = "sanskrit" in script_lower

    if is_sanskrit:
        # Strip all nuktas (Sanskrit classical orthography has no nuktas)
        for nukta_char, replacement in SANSKRIT_NUKTA_MAP.items():
            text = text.replace(nukta_char, replacement)

        # Fix specific classical Sanskrit medical/philosophical domain minimal pairs:
        # e.g. व्रण (wound/ulcer) consistently confused as ब्रण (non-word)
        text = re.sub(r'\bब्रण', 'व्रण', text)
        text = re.sub(r'\bब्रणी', 'व्रणी', text)

    # 4. English ligature restoration (fixes ffi, ff, fl dropped characters)
    for pattern, repl in ENGLISH_LIGATURE_REPLACEMENTS:
        text = pattern.sub(repl, text)

    return text
