import os

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# === Text Chunking ===

class SafeTextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding='iso-8859-1', autodetect_encoding=False)

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def split_into_chunks(text):
    # Wrap plain text into a Document first
    from langchain.schema import Document
    return [doc.page_content for doc in splitter.split_documents([Document(page_content=text)])]

def detect_and_load_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    loader = {
        ".pdf": PyPDFLoader,
        ".txt": SafeTextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".epub": UnstructuredEPubLoader
    }.get(ext)
    return loader(file_path).load() if loader else None