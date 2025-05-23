import re
import unicodedata
import ftfy
from data.jsonhandler import apply_normalization, load_normalization_map

# === Load Normalization Rules ===
normalization_rules = load_normalization_map(create_if_missing=False)  # Print loading/failure message
'''
The normalization JSON is used here to clean and normalize 
the entire raw text (fixing ligatures, punctuation, OCR artifacts, etc).
This filtered text is cleaned and normalized, ready to be chunked.
'''
def normalize_unicode(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    return apply_normalization(text, normalization_rules)

# === Export to chunker >>>
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