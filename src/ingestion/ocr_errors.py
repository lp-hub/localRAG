import re
import difflib
from spellchecker import SpellChecker

spell = SpellChecker()

def detect_potential_ocr_errors(text: str) -> dict[str, str]:
    words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text))  # Avoid very short tokens
    misspelled = spell.unknown(words)
    suggestions = {}

    for word in misspelled:
        candidates = spell.candidates(word)
        if not candidates:
            continue
        best_match = max(candidates, key=lambda c: difflib.SequenceMatcher(None, word, c).ratio())
        if best_match != word:
            suggestions[word] = best_match
    return suggestions