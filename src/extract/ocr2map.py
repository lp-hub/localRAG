import json
import os
import re
from glob import glob
'''
    It preserves ligatures and punctuation sections.
    Escapes regex safely using re.escape.
    Automatically deduplicates fixes across multiple logs.
    You can re-run it anytime without duplicating entries.
    python3 scripts/orc2map.py
'''
LOG_DIR = "logs"
MAP_FILE = os.path.join("db", "normalization_map.json")

# Regex to extract fixes from log lines
OCR_FIX_REGEX = re.compile(r"\[OCR\] Suggest fix: '(.+?)' → '(.+?)'")

# Load existing normalization map
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        norm_map = json.load(f)
else:
    norm_map = {
        "ligatures": {
            "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"
        },
        "punctuation": {
            "–": "-", "—": "-", "‘": "'", "’": "'", "“": "\"", "”": "\"", "…": "..."
        },
        "ocr_artifacts": {}
    }

# Collect all OCR fixes
ocr_fixes = norm_map.get("ocr_artifacts", {})

log_files = glob(os.path.join(LOG_DIR, "ocr_artifacts*.txt"))
print(f"[INFO] Found {len(log_files)} log files.")

for log_file in log_files:
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = OCR_FIX_REGEX.search(line)
            if match:
                wrong, correct = match.groups()
                key = f"\\b{re.escape(wrong)}\\b"
                if key not in ocr_fixes:
                    print(f"[ADD] {wrong} → {correct}")
                    ocr_fixes[key] = correct

# Save updated normalization map
os.makedirs("db", exist_ok=True) # create folder db if it does not exist
norm_map["ocr_artifacts"] = dict(sorted(ocr_fixes.items()))

with open(MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(norm_map, f, indent=4, ensure_ascii=False)

print(f"[DONE] normalization_map.json updated with {len(ocr_fixes)} OCR fixes.")