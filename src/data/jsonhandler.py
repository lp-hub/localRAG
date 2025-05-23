import json
import re
import os
import logging
import difflib
from pathlib import Path
from spellchecker import SpellChecker

spell = SpellChecker()
'''
Creation of default normalization_map.json
To update it constantly, call add_normalization_entry(...) in chunker
'''
# === Constants ===
BASE_DIR = Path(__file__).parent
JSON_PATH = (BASE_DIR / "../../db/normalization_map.json").resolve()
DEFAULT_STRUCTURE = {
    "ligatures": {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl"
    },
    "punctuation": {
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": "\"",
        "”": "\"",
        "…": "..."
    },
    "ocr_artifacts": {
        r"\bfa9ade\b": "façade",
        r"\bmedireval\b": "mediaeval",
        r"\bsub- sequent\b": "subsequent",
        r"\bHermetic A rcanum\b": "Hermetic Arcanum",
        r"\bAutJuw\b": "Author",
        r"\bTableaz£ de l'inconstance\b": "Tableau de l'inconstance",
        r"\bPhysictZ RestituttZ\b": "Physica Restituta",
    }
}

# === Logging ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# === File Handling ===
def ensure_normalization_json(path: Path = JSON_PATH, force=False):
    if not force and not path.exists():
        raise RuntimeError(f"[Abort] normalization_map.json missing: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_STRUCTURE, f, indent=4, ensure_ascii=False)
            logger.info(f"Normalization map created at {path}")   
    except Exception as e:
        logger.error(f"Error ensuring normalization map at {path}: {e}")

def load_normalization_map(path: Path = JSON_PATH, create_if_missing: bool = False) -> dict:
    if create_if_missing:
        ensure_normalization_json(path)
    elif not path.exists():
        logger.warning(f"Normalization map not found at {path}")
        print("Run with --rebuild-db to generate it.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Normalization map loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading normalization map from {path}: {e}")
        return {}

def save_normalization_map(data: dict, path: Path = JSON_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Normalization map saved to {path}")
    except Exception as e:
        logger.error(f"Error saving normalization map to {path}: {e}")

# === Apply Normalization ===
def apply_normalization(text: str, norm_map: dict) -> str:
    for cat in ["ligatures", "punctuation"]:
        for bad, good in norm_map.get(cat, {}).items():
            text = text.replace(bad, good)
    for pattern, repl in norm_map.get("ocr_artifacts", {}).items():
        text = re.sub(pattern, repl, text)
    return text

def apply_regex_normalization(text: str, regex_rules: list[tuple[str, str]]) -> str:
    for pattern, repl in regex_rules:
        text = re.sub(pattern, repl, text)
    return text

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

# === Add/Edit Entry ===
def add_normalization_entry(category: str, bad: str, good: str, path: str = JSON_PATH):
    
    data = load_normalization_map(path, create_if_missing=False)
    logger.debug(f"Normalization data keys being saved: {list(data.keys())}")

    if category not in data:
        data[category] = {}
    if bad in data[category]:
        logger.info(f"Updating existing entry in '{category}': '{bad}' -> '{good}'")
    else:
        logger.info(f"Adding new entry in '{category}': '{bad}' -> '{good}'")

    data[category][bad] = good
    save_normalization_map(data, path)