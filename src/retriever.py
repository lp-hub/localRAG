import hashlib
import re
import string
import fitz  # PyMuPDF

from pathlib import Path
from db import insert_document, insert_chunks, get_existing_hashes
from config import EMBED_MODEL_NAME
from langchain.schema import Document


def hash_file(file_path):
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


def is_page_ocr_or_image(doc_path):
    doc = fitz.open(doc_path)
    for page in doc:
        text = page.get_text("text")
        if text and len(text.strip()) > 30:
            return True
    return False


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return "\n".join(texts)

def is_trash(chunk):
    chunk = chunk.strip()
    if len(chunk) < 10:
        return True

    # Remove overly aggressive Unicode exclusion
    weird_unicode = sum(1 for c in chunk if ord(c) > 2000)
    if weird_unicode / len(chunk) > 0.3:
        return True

    printable_ratio = sum(c in string.printable for c in chunk) / len(chunk)
    alnum_ratio = sum(c.isalnum() for c in chunk) / len(chunk)

    if printable_ratio < 0.6:
        return True
    if alnum_ratio < 0.2:
        return True

    return False


def chunk_documents(data_dir, split_func):
    docs = []
    existing_hashes = get_existing_hashes()

    for path in Path(data_dir).rglob("*"):
        if not path.is_file():
            continue

        file_hash = hash_file(path)
        if file_hash in existing_hashes:
            print(f"[SKIP] Already indexed: {path}(hash: {file_hash})")
            continue

        try:
            if path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(str(path))
            else:
                text = path.read_text(errors="ignore")
        except Exception as e:
            print(f"[ERROR] Cannot read file {path}: {e}")
            continue

        chunks = split_func(text)
        print(f"Indexed: {path} | Chunks: {len(chunks)}")

        trash_count = sum(1 for chunk in chunks if is_trash(chunk))
#        if trash_count / len(chunks) > 0.7:
#            print(f"[SKIP] File mostly garbage: {path}")
#            continue

        doc_id = insert_document(
            str(path), path.stem, file_hash, path.suffix[1:], EMBED_MODEL_NAME
        )
        insert_chunks(doc_id, chunks)

        accepted = 0
        for idx, item in enumerate(chunks):
            if isinstance(item, tuple):
                chunk, page_num = item
            else:
                chunk = item
            page_num = "?"  # Or extract this from tuple if split_func returns it
            chunk = ' '.join(chunk.split())
            if is_trash(chunk):
                print(f"[FILTERED] Trash chunk skipped: {chunk[:50]}...")
                continue
            accepted += 1
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "doc_id": doc_id,
                    "path": str(path),
                    "title": path.stem,
                    "chunk_index": idx,
                    "page": page_num
                }
            ))

            print(f"Accepted {accepted}/{len(chunks)} chunks from {path}")

    return docs