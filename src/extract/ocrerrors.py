import os
import re
import json
from pathlib import Path
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv
load_dotenv()

# ========== Configuration ==========
DST_DIR = Path(os.getenv("DST_DIR", "text_files"))  # fallback for manual testing
FREQ_DICT = "frequency_dictionary_en_82_765.txt"
WHITELIST = "whitelist.txt"
OUTPUT_JSON = "ocr_corrections.json"
OUTPUT_TXT = "ocr_suggestions_report.txt"
MAX_EDIT_DISTANCE = 2

# ========== Normalization Maps ==========
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
PUNCTUATION = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."}

REGEX_FIXES = {
    r"\bfa9ade\b": "façade",
    r"\bmedireval\b": "mediaeval",
    r"\bsub- sequent\b": "subsequent",
    r"\bHermetic A rcanum\b": "Hermetic Arcanum",
    r"\bAutJuw\b": "Author",
    r"\bTableaz£ de l'inconstance\b": "Tableau de l'inconstance",
    r"\bPhysictZ RestituttZ\b": "Physica Restituta",
}

# ========== Helper Functions ==========
def normalize(text):
    for src, tgt in {**LIGATURES, **PUNCTUATION}.items():
        text = text.replace(src, tgt)
    return text

def extract_words(text):
    return re.findall(r"\b[a-zA-Z0-9’'-]{3,}\b", text)

def load_whitelist(path):
    if not Path(path).exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

# ========== Load SymSpell and Dictionary ==========
sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)
if not sym_spell.load_dictionary(FREQ_DICT, term_index=0, count_index=1):
    raise RuntimeError(f"Failed to load dictionary from {FREQ_DICT}")

whitelist = load_whitelist(WHITELIST)

# ========== Process Text Files ==========
corrections = {}
lines_with_corrections = []

for file_path in DST_DIR.rglob("*.txt"):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = normalize(raw_line)

            # Apply regex artifact rules
            for pattern, replacement in REGEX_FIXES.items():
                if re.search(pattern, line):
                    fixed = re.sub(pattern, replacement, line)
                    lines_with_corrections.append({
                        "file": str(file_path),
                        "line": line_num,
                        "original": line.strip(),
                        "suggested": fixed.strip()
                    })
                    corrections[pattern] = replacement

            # Spellcheck individual words
            words = set(extract_words(line.lower()))
            for word in words:
                if word in whitelist or sym_spell.word_frequency.lookup(word):
                    continue

                suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=MAX_EDIT_DISTANCE)
                if suggestions:
                    best = suggestions[0]
                    if best.term != word:
                        lines_with_corrections.append({
                            "file": str(file_path),
                            "line": line_num,
                            "original": word,
                            "suggested": best.term
                        })
                        corrections[word] = best.term

# ========== Output Results ==========
with open(OUTPUT_JSON, "w", encoding="utf-8") as f: # Save JSON report
    json.dump({
        "corrections": corrections,
        "lines": lines_with_corrections
    }, f, indent=2, ensure_ascii=False)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f: # Save text report
    for entry in lines_with_corrections:
        f.write(f"[{entry['file']}:{entry['line']}] '{entry['original']}' → '{entry['suggested']}'\n")

print(f"[DONE] Corrections saved to {OUTPUT_JSON} and {OUTPUT_TXT}")

# Regex-based artifact replacement
# Ligature/punctuation normalization
# SymSpell-based OCR correction
# Whitelist filtering
# .json + .txt output
# File and line-level logging
# You can also post-process the JSON output like this:
# regex_map = {fr"\\b{k}\\b": v for k, v in corrections.items()}
