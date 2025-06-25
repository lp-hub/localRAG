import os
import re
import unicodedata
import ftfy
from datetime import datetime
from spellchecker import SpellChecker
spell = SpellChecker()
from data.jsonhandler import apply_normalization, load_normalization_map, detect_potential_ocr_errors

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

def process_text_for_chunking(text: str, filename: str = "", enable_ocr: bool = True) -> str:
    '''
    Handles text cleaning and optional OCR artifact detection.
    '''
    is_txt = filename.lower().endswith(".txt")
    norm_map = normalization_rules()

    if is_txt:
        print(f"[SKIP] OCR skipped for .txt file: {filename}")
        return apply_normalization(text.strip(), norm_map)

    cleaned = clean_text(text)

    if enable_ocr and not is_clean_text(cleaned):
        print("[OCR] Text looks noisy, scanning for OCR artifacts...")
        ocr_fixes = detect_potential_ocr_errors(cleaned)

        if ocr_fixes:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            log_path = os.path.join(log_dir, f"ocr_artifacts_{timestamp}.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                for bad, good in sorted(ocr_fixes.items()):
                    log_msg = f"[OCR] Suggest fix: '{bad}' → '{good}'"
                    print(log_msg)
                    f.write(log_msg + "\n")
        else:
            print("[OCR] No significant OCR artifacts found.")

    return apply_normalization(cleaned, norm_map)