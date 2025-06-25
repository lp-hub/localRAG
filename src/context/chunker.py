from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.filter import process_text_for_chunking
from server.llm import parse_args

# ========== Text Splitter ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ========== Chunking Logic ==========
def split_into_chunks(text: str, filename: Path | str = "") -> list[str]:
    print("[DEBUG] Starting split_into_chunks")

    args = parse_args()
    normalized = process_text_for_chunking(
        text,
        filename=str(filename),
        enable_ocr=not args.ocr_skip)

    print("[DEBUG] Splitting with text splitter")
    return [doc.page_content for doc in splitter.split_documents([Document(page_content=normalized)])]