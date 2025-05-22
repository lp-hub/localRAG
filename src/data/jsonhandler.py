import json
import re
import os
import logging
import difflib
from spellchecker import SpellChecker

spell = SpellChecker()
'''
Creation of default normalization_map.json
To update it constantly, call add_normalization_entry(...) in chunker
'''
# === Setup logging ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to INFO or WARNING to reduce verbosity if needed

ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# === Constants ===
BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../db/normalization_map.json"))
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

# === File Handling ===
def ensure_normalization_json(path: str = JSON_PATH, force: bool = False):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if force or not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_STRUCTURE, f, indent=4, ensure_ascii=False)
            logger.info(f"Normalization map created at {path}")
        else:
            logger.debug(f"Normalization map already exists at {path}, skipping creation.")
    except Exception as e:
        logger.error(f"Error creating normalization map at {path}: {e}")

def load_normalization_map(path: str = JSON_PATH, create_if_missing: bool = True) -> dict:
    if create_if_missing:
        ensure_normalization_json(path)
    elif not os.path.exists(path):
        logger.warning(f"Normalization map not found at {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Normalization map loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading normalization map from {path}: {e}")
        return {}

def save_normalization_map(data: dict, path: str = JSON_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Normalization map saved to {path}")
    except Exception as e:
        logger.error(f"Error saving normalization map to {path}: {e}")

# === Application ===
def apply_normalization(text: str, norm_map: dict) -> str:
    # Apply literal replacements
    for category_name in ["ligatures", "punctuation"]:
        category = norm_map.get(category_name, {})
        for bad, good in category.items():
            text = text.replace(bad, good)
    # Apply regex replacements for OCR artifacts
    ocr_artifacts = norm_map.get("ocr_artifacts", {})
    for pattern, repl in ocr_artifacts.items():
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