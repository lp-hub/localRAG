import os

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.filter import clean_text

# === Custom Safe Loader for .txt ===
class SafeTextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding='iso-8859-1', autodetect_encoding=False)

# === Text Splitter ===
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

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