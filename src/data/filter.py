import re
import unicodedata
import ftfy
from spellchecker import SpellChecker
spell = SpellChecker()
from data.jsonhandler import apply_normalization, load_normalization_map

# ========== Load Normalization Rules ==========
_normalization_rules_cache = None
'''
The normalization JSON is used here to clean and normalize 
the entire raw text (fixing ligatures, punctuation, OCR artifacts, etc).
This filtered text is cleaned and normalized, ready to be chunked.
'''
def normalization_rules():
    global _normalization_rules_cache
    if _normalization_rules_cache is None:
        _normalization_rules_cache = load_normalization_map(create_if_missing=False)
    return _normalization_rules_cache

def normalize_unicode(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    return apply_normalization(text, normalization_rules())

# Export to chunker >>>
def clean_text(raw: str) -> str:
    print(f"[Cleaning] Input length: {len(raw)}")
    text = normalize_unicode(raw)
    text = text.strip()

    # Normalize line spacing and inline linebreaks
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"(?<![.?!])\n(?![A-Z])", " ", text)
    
    # Remove ALL CAPS headers (too aggressive?)
    text = re.sub(r"^[A-Z\s\.\'\"]{10,}$", "", text, flags=re.MULTILINE)

    # Remove common editorial boilerplate
    text = re.sub(r"(?:Edited by|Translated by|PENES NOS|MDC.*|©.*)", "", text, flags=re.IGNORECASE)

    text = re.sub(r" {2,}", " ", text)  # Remove double spaces
    print(f"[Cleaning] Output length: {len(text)}")
    return text

# Export to chunker >>>
def is_clean_text(text: str, max_misspelled_ratio: float = 0.01, sample_size: int = 200) -> bool:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    sample = words[:sample_size]
    misspelled = spell.unknown(sample)
    ratio = len(misspelled) / len(sample) if sample else 0
    print(f"[HEURISTIC] Misspelled ratio: {ratio:.3f}")
    return ratio < max_misspelled_ratio