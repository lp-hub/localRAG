import os
import re
import unicodedata
import ftfy

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.jsonhandler import ( add_normalization_entry,load_normalization_map, apply_normalization)

# === Custom Safe Loader for .txt ===
class SafeTextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding='iso-8859-1', autodetect_encoding=False)

# === Load Normalization Rules ===
normalization_rules = load_normalization_map()

# === Text Splitter ===
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# === Unicode + Ligature Normalization ===
def normalize_unicode(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    return apply_normalization(text, normalization_rules)

# === Full Cleaning Pipeline ===
def clean_text(raw: str) -> str:
    text = normalize_unicode(raw)
    text = text.strip()

    # Normalize line spacing and inline linebreaks
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"(?<![.?!])\n(?![A-Z])", " ", text)
    
    # Remove ALL CAPS headers
    text = re.sub(r"^[A-Z\s\.\'\"]{10,}$", "", text, flags=re.MULTILINE)

    # Remove common editorial boilerplate
    text = re.sub(r"(?:Edited by|Translated by|PENES NOS|MDC.*|©.*)", "", text, flags=re.IGNORECASE)

    # OCR issues (expand as needed)
    ocr_fixes = {}
    for bad, good in ocr_fixes.items():
        add_normalization_entry("ocr_artifacts", bad, good)
        text = re.sub(bad, good, text)
    return re.sub(r" {2,}", " ", text) # Remove double spaces

# === Chunking Logic ===
def split_into_chunks(text: str) -> list[str]:
    cleaned = clean_text(text)
    return [doc.page_content for doc in splitter.split_documents([Document(page_content=cleaned)])]

# === Loader Dispatcher ===
def detect_and_load_text(file_path: str) -> list[Document] | None:
    ext = os.path.splitext(file_path)[-1].lower()
    loader_cls = {
        ".pdf": PyPDFLoader,
        ".txt": SafeTextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".epub": UnstructuredEPubLoader
    }.get(ext)
    return loader_cls(file_path).load() if loader_cls else None