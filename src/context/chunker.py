import os
from datetime import datetime
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.filter import clean_text, is_clean_text
from data.jsonhandler import load_normalization_map, apply_normalization, detect_potential_ocr_errors

# ========== Text Splitter ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ========== Chunking Logic ==========
def split_into_chunks(text: str, update_map: bool = False, skip_ocr_check_if_clean: bool = True, filename: Path | str = "") -> list[str]:
    print("[DEBUG] Starting split_into_chunks")
    
    filename_str = str(filename) # convert filename to string early
    if filename_str.lower().endswith(".txt"):
        skip_ocr_check_if_clean = True
        update_map = False
        cleaned = text  # no cleaning for txt files
        print(f"[SKIP] OCR check skipped for TXT file: {filename_str}")
    else:
        cleaned = clean_text(text)
        print("[DEBUG] Finished clean_text")

        if skip_ocr_check_if_clean:
            if is_clean_text(cleaned):
                print("[SKIP] Text looks clean. Skipping OCR artifact detection.")
            else:
                print("[DEBUG] Detected potential noise, running OCR artifact scan...")
                update_map = True  # force scan

    if update_map:
        ocr_fixes = detect_potential_ocr_errors(cleaned)
        print(f"[DEBUG] Found {len(ocr_fixes)} OCR fixes")

        if ocr_fixes:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            log_filename = f"ocr_artifacts_{timestamp}.txt"
            log_path = os.path.join(log_dir, log_filename)

            with open(log_path, "a", encoding="utf-8") as f:
                for bad, good in sorted(ocr_fixes.items()):
                    log_msg = f"[OCR] Suggest fix: '{bad}' → '{good}'"
                    print(log_msg)
                    f.write(log_msg + "\n")
                    print(f"[LOG] Added to log: {log_msg}")

        norm_map = load_normalization_map()
        normalized = apply_normalization(cleaned, norm_map)
    else:
        # No OCR check, no normalization needed
        normalized = cleaned

    print("[DEBUG] Splitting with text splitter")
    return [doc.page_content for doc in splitter.split_documents([Document(page_content=normalized)])]