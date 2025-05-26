import hashlib
import string

from pathlib import Path
from data import insert_document,insert_chunks, get_existing_hashes
from config import EMBED_MODEL_NAME, GARBAGE_THRESHOLD
from langchain.schema import Document

from ingest.chunker import detect_and_load_text

#For large files, consider reading in chunks:
def hash_file(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() 

def is_good_chunk(chunk: str) -> bool:
    chunk = chunk.strip()
    if len(chunk) < 10:
        return False  # Too short

    # Too many high Unicode chars? => noisy
    weird_unicode_ratio = sum(1 for c in chunk if ord(c) > 2000) / len(chunk)
    if weird_unicode_ratio > 0.05:  # lower threshold for good chunks
        return False

    # Check printable and alnum ratio more strictly
    printable_ratio = sum(c in string.printable for c in chunk) / len(chunk)
    alnum_ratio = sum(c.isalnum() for c in chunk) / len(chunk)

    if printable_ratio < 0.9:  # pretty much all printable chars for good chunk
        return False
    if alnum_ratio < 0.5:  # at least half alnum
        return False

    # Passed all checks → chunk is "good"
    return True

# Filtering bad chunks
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

def chunk_documents(data_dir: str, split_func: callable) -> list[Document]:
    """Load files from data_dir, extract and chunk text, filter trash,
    and return list of Document objects with metadata."""
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
            docs_from_loader = detect_and_load_text(str(path))
            print(f"[DEBUG] Running OCR artifact detection: {path.stem}")
            if not docs_from_loader:
                print(f"[SKIP] Unsupported file type: {path}")
                continue
            text = "\n\n".join(doc.page_content for doc in docs_from_loader)
        except Exception as e:
            print(f"[ERROR] Cannot load file {path}: {e}")
            continue

        chunks = split_func(text)
        if not chunks:
            print(f"[SKIP] No chunks extracted: {path}")
            continue

        print(f"Indexed: {path} | Chunks: {len(chunks)}")

        trash_count = sum(1 for chunk in chunks if is_trash(chunk))
        if trash_count / len(chunks) > GARBAGE_THRESHOLD:
            print(f"[SKIP] File mostly garbage: {path} ({trash_count}/{len(chunks)} chunks)")
            continue

        # Filter trash chunks and add OCR metadata
        filtered_chunks = []
        for chunk in chunks:
            if is_trash(chunk):
                continue
            skip_ocr_fix = is_good_chunk(chunk)
            filtered_chunks.append((chunk, {"skip_ocr_fix": skip_ocr_fix}))

        doc_id = insert_document(
            str(path), path.stem, file_hash, path.suffix[1:], EMBED_MODEL_NAME
        )

        accepted = 0
        final_chunks = []
        for idx, (chunk, metadata) in enumerate(filtered_chunks): 
            page_num = "?" # update page data here if needed
            chunk = ' '.join(chunk.split())
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "doc_id": doc_id,
                    "path": str(path),
                    "title": path.stem,
                    "chunk_index": idx,
                    "page": page_num,
                    "skip_ocr_fix": metadata.get("skip_ocr_fix", False),
                }
            ))
            final_chunks.append((chunk, metadata))
            accepted += 1

        if final_chunks:
            print(f"[DB] Inserting {len(final_chunks)} chunks to DB for {path.name}")
            insert_chunks(doc_id, final_chunks)

        print(f"Accepted {accepted}/{len(chunks)} chunks from {path.stem}")

    return docs