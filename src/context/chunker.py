import os
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.filter import clean_text
from data.jsonhandler import load_normalization_map, apply_normalization, detect_potential_ocr_errors

# ========== Text Splitter ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ========== Chunking Logic ==========
def split_into_chunks(text: str, update_map: bool = False) -> list[str]:
    print("[DEBUG] Starting split_into_chunks")
    cleaned = clean_text(text)
    print("[DEBUG] Finished clean_text")

    if update_map:
        print("[DEBUG] Detecting OCR artifacts (logging only, no map update)")
        ocr_fixes = detect_potential_ocr_errors(cleaned)
        print(f"[DEBUG] Found {len(ocr_fixes)} OCR fixes")

        if ocr_fixes:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            log_filename = f"ocr_artifacts_{timestamp}.txt"
            log_path = os.path.join("logs", log_filename)

            with open(log_path, "a", encoding="utf-8") as f:
                for bad, good in sorted(ocr_fixes.items()):
                    log_msg = f"[OCR] Suggest fix: '{bad}' → '{good}'"
                    print(log_msg)
                    f.write(log_msg + "\n")
                    print(f"[LOG] Added to log: {log_msg}")

    # Apply Normalization Rules (includes updated fixes)
    norm_map = load_normalization_map()
    normalized = apply_normalization(cleaned, norm_map)

    print("[DEBUG] Splitting with text splitter")
    return [doc.page_content for doc in splitter.split_documents([Document(page_content=cleaned)])]