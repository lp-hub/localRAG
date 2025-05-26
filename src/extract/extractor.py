import os
import re
import sys
from pathlib import Path
from datetime import datetime
# PYTHONPATH=./src python scripts/extractor.py
from ingest.chunker import detect_and_load_text
from ocr import check_and_ocr_empty_outputs
from dotenv import load_dotenv
load_dotenv()

SRC_DIR = Path(os.getenv("SRC_DIR"))  # change this to your source folder
DST_DIR = Path(os.getenv("DST_DIR"))  # change this to your output folder
LOG_FILE = Path(os.getenv("LOG_FILE")) # full absolute path from .env

timestamp_pattern = re.compile(r"_\d{8}_\d{6}$")  # e.g., _20250523_153012

def assert_dirs_exist(*dirs):
    for d in dirs:
        if d is None:
            print(f"[FATAL] Environment variable for a directory is missing.")
            sys.exit(1)
        if not d.exists() or not d.is_dir():
            print(f"[FATAL] Directory does not exist: {d}")
            sys.exit(1)
            # 
assert_dirs_exist(DST_DIR, SRC_DIR, LOG_FILE.parent)

print("DST_DIR:", DST_DIR)
print("SRC_DIR:", SRC_DIR)
print("LOG_FILE:", LOG_FILE)

# Populate log file from existing .txt files in DST_DIR
def initialize_log_from_existing_outputs():
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        print("[INIT] Log already exists. Skipping initialization.")
        return

    found = set()
    for txt_file in DST_DIR.rglob("*.txt"):
        stem = txt_file.stem  # filename without extension, e.g. "Origami"
        # Remove timestamp suffix if present
        match = re.match(r"^(.*\.\w+)_\d{8}_\d{6}$", txt_file.stem)
        if match:
            base_name = match.group(1)
            found.add(base_name)
        else:
            print(f"[WARN] Could not parse filename from {txt_file.name}")

    if found:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("w", encoding="utf-8") as log_f:
            for entry in sorted(found):
                log_f.write(f"{entry}\n")
        print(f"[INIT] Populated log with {len(found)} existing extracted files.")
    else:
        print("[INIT] No existing outputs found to initialize log.")

# Read list of already processed files (without timestamp suffix)
def get_already_processed():
    if not LOG_FILE.exists():
        return set()
    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        print(f"[DEBUG] Already processed files in log: {len(lines)}")
        return set(lines)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files_processed = 0
    processed_files = get_already_processed()
    new_log_entries = []

    for file_path in SRC_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        original_filename = file_path.name  # e.g. "book.pdf"

        if original_filename in processed_files:
            print(f"[SKIP] Already extracted: {file_path}")
            continue

        docs = detect_and_load_text(str(file_path))
        if not docs:
            continue

        # Join all pages into one text blob
        text = "\n\n".join(doc.page_content for doc in docs)

        # Save extracted text with timestamp suffix
        target_dir = DST_DIR / file_path.relative_to(SRC_DIR).parent
        target_dir.mkdir(parents=True, exist_ok=True)

        new_filename = f"{original_filename}_{timestamp}.txt"  # book.pdf_20250524_153012.txt
        target_path = target_dir / new_filename

        with target_path.open("w", encoding="utf-8") as f:
            f.write(text)

        print(f"[EXTRACTED] {file_path} → {target_path}")
        files_processed += 1
        new_log_entries.append(original_filename)

    # Append to log
    if new_log_entries:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as log_f:
            for entry in new_log_entries:
                log_f.write(f"{entry}\n")
        print(f"[LOG] Writing {len(new_log_entries)} new entries to log file: {LOG_FILE}")

    print(f"[DONE] Extracted text from {files_processed} files.")

if __name__ == "__main__":
    initialize_log_from_existing_outputs()
    main()
    # if os.getenv("OCR_ON_EMPTY", "false").lower() == "true":
    #      check_and_ocr_empty_outputs()

'''
   OCR only proceeds:
   if .env flag is true,
   and the user confirms at the prompt.
'''
'''
First run: populates log and skips extracted files.
Second run: nothing duplicated.
LOG_FILE acts as ground truth.
    # After extraction, scan the destination folder for extracted .txt files.
    # If any .txt file is empty, look back at the source folder for files with the same base name (stem).
    # For those source files, perform OCR (e.g., using Tesseract) to extract text.
    # Save this OCR output into a different output folder (or maybe overwrite?).
    # Integrate this as an optional step (flag-controlled) into your existing extractor pipeline.
Add a flag OCR_ON_EMPTY = os.getenv("OCR_ON_EMPTY", "false").lower() == "true" at the top.
At the end of your main extraction, run: if OCR_ON_EMPTY: check_and_ocr_empty_outputs()
'''