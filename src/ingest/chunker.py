import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
from unstructured.partition.doc import partition_doc
from unstructured.partition.html import partition_html

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

# --- .doc loader (fallback using unstructured) ---
class UnstructuredDocLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        elements = partition_doc(filename=self.file_path)
        return [Document(page_content=str(el)) for el in elements]

# --- .rtf loader using striprtf ---
class RTFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = rtf_to_text(f.read())
        return [Document(page_content=content)]

# --- .djvu loader using djvu.decode (basic) ---
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
class CHMLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self) -> list[Document]:
        extract_dir = Path("/tmp/chm_extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Extracting CHM file {self.file_path} to {extract_dir}")
        
        subprocess.run(["extract_chmLib", self.file_path, str(extract_dir)])
        text = ""
        for html_file in extract_dir.rglob("*.htm*"):
            with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n"
        print(f"[DEBUG] Finished extracting CHM, total length {len(text)} chars")
        return [Document(page_content=text)]
        
# Some .chm files can't be parsed well because they're binary-encoded archives.
#     Extract .chm manually:
# archmage mybook.chm output_dir/
# or:
#     7z x mybook.chm -ooutput_dir/
#     Then recursively process .html files from the extracted content.
    
# --- .htm .html loader using unstructured ---
class UnstructuredHTMLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            elements = partition_html(text=f.read())
        return [Document(page_content=el.text) for el in elements if el.text]

# --- .mobi loader using ebooklib and bs4 ---
# Class to fix Path vs str problem in UnstructuredEPubLoader
class FixedEPubLoader(UnstructuredEPubLoader):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(str(file_path), *args, **kwargs)     
# MOBI is not directly supported. Convert using Calibre CLI to EPUB before ingestion.
# ebook-convert input.mobi output.epub
class MOBILoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        if not shutil.which("ebook-convert"):
            raise EnvironmentError("'ebook-convert' not found. Please install Calibre CLI.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            epub_path = tmpdir_path / (self.file_path.stem + ".epub")

            try:
                subprocess.run(
                    ["ebook-convert", str(self.file_path), str(epub_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to convert MOBI to EPUB: {e}")

            if not epub_path.exists():
                raise FileNotFoundError(f"Conversion failed, EPUB not found at {epub_path}")
            return FixedEPubLoader(epub_path).load()

class PyPDFLoaderWithPassword(PyPDFLoader):
    def __init__(self, file_path, password=None):
        super().__init__(file_path)
        self.password = password

    def load(self) -> list[Document]:
        reader = PdfReader(self.file_path, password=self.password)
        texts = [page.extract_text() or "" for page in reader.pages]
        return [Document(page_content="\n".join(texts))]

# === Loader Dispatcher ===
def detect_and_load_text(file_path: str, pdf_password: str = None) -> list[Document] | None:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoaderWithPassword(file_path, password=pdf_password)
    else:
        loader_map = {
        # ".pdf": PyPDFLoaderWithPassword, # PyPDFLoader replaced to fix pypdf/_encryption.py
        ".txt": SafeTextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".epub": FixedEPubLoader,  # UnstructuredEPubLoader replaced to globally fix .epub loading
        ".doc": UnstructuredDocLoader,
        ".rtf": RTFLoader,
        ".djvu": DidjvuLoader,
        ".chm": CHMLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".mobi": MOBILoader,  # custom MOBI loader using Calibre conversion
        }
        loader_cls = loader_map.get(ext)
        if loader_cls is None:          
            return None
        loader = loader_cls(file_path)
    try:
        return loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []
