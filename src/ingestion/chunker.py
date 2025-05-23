import os
import time

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.filter import clean_text
from data.jsonhandler import load_normalization_map, apply_normalization, detect_potential_ocr_errors

# === Custom Safe Loader for .txt ===
class SafeTextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding='iso-8859-1', autodetect_encoding=False)

# === Text Splitter ===
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# === Chunking Logic ===
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

            from datetime import datetime
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


# === Loaders ===
# Not Included in LangChain so we create our own classes ===
# --- .doc loader (fallback using unstructured) ---
from unstructured.partition.doc import partition_doc

class UnstructuredDocLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        elements = partition_doc(filename=self.file_path)
        return [Document(page_content=str(el)) for el in elements]

# --- .rtf loader using striprtf ---
from striprtf.striprtf import rtf_to_text

class RTFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = rtf_to_text(f.read())
        return [Document(page_content=content)]

# --- .djvu loader using djvu.decode (basic) ---
import shutil
import subprocess

class DidjvuLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        if not shutil.which("djvutxt"):
            raise EnvironmentError("djvutxt is not installed. sudo apt install djvulibre-bin")
        
        djvu_path = Path(self.file_path)

        if not djvu_path.exists():
            raise FileNotFoundError(f"DjVu file not found: {self.file_path}")

        try:
            # Extract text using djvutxt
            result = subprocess.run(
                ["djvutxt", self.file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            full_text = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"djvutxt failed: {e.stderr}")

        return [Document(page_content=full_text)]
        
# --- .chm loader using extract_chmlib ---
import subprocess
from pathlib import Path

class CHMLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        extract_dir = Path("/tmp/chm_extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["extract_chmLib", self.file_path, str(extract_dir)])
        text = ""
        for html_file in extract_dir.rglob("*.htm*"):
            with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n"
        return [Document(page_content=text)]

# === Loader Dispatcher ===
def detect_and_load_text(file_path: str) -> list[Document] | None:
    ext = os.path.splitext(file_path)[-1].lower()
    loader_cls = {
        ".pdf": PyPDFLoader,
        ".txt": SafeTextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".epub": UnstructuredEPubLoader,
        ".doc": UnstructuredDocLoader,
        ".rtf": RTFLoader,
        ".djvu": DidjvuLoader,
        ".chm": CHMLoader,
    }.get(ext)
    return loader_cls(file_path).load() if loader_cls else None