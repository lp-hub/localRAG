import hashlib
import string

from pathlib import Path
from data import insert_document, insert_chunks, get_existing_hashes
from datetime import datetime
from config import EMBED_MODEL_NAME, GARBAGE_THRESHOLD
from langchain.schema import Document

from ingestion.chunker import (
    PyPDFLoader,
    SafeTextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader,
    UnstructuredDocLoader,
    RTFLoader,
    DidjvuLoader,
    CHMLoader,
)

def detect_and_load_text(file_path: str) -> list[Document] | None:
    ext = Path(file_path).suffix.lower()
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

def hash_file(file_path):
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


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
            docs_from_loader = detect_and_load_text(str(path))
            if not docs_from_loader:
                print(f"[SKIP] Unsupported file type: {path}")
                continue
            # Flatten all content into one string if needed
            text = "\n\n".join(doc.page_content for doc in docs_from_loader)

            # Optional: export extracted raw text to .txt
            # logs/book_20250523_174501.txt
            export_raw = True  # toggle this
            if export_raw:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_dir = Path("logs")
                export_path = export_dir / f"{path.stem}_{timestamp}.txt"
                export_dir.mkdir(parents=True, exist_ok=True)
                with export_path.open("w", encoding="utf-8") as f:
                    f.write(text) 
            # If you're dealing with noisy formats (e.g. .djvu, .chm, OCR .pdfs), 
            # it's worth exporting once to build a cleaner version.

        except Exception as e:
            print(f"[ERROR] Cannot load file {path}: {e}")
            continue

        chunks = split_func(text)
        if not chunks:
            print(f"[SKIP] No chunks extracted: {path}")
            continue
        
        print(f"Indexed: {path} | Chunks: {len(chunks)}")
        
        trash_count = sum(1 for chunk in chunks if is_trash(chunk))
        if len(chunks) == 0 or trash_count / len(chunks) > GARBAGE_THRESHOLD:
            print(f"[SKIP] File mostly garbage: {path} ({trash_count}/{len(chunks)} chunks)")
            continue
        

        # Filter trash chunks out and add skip_ocr_fix metadata
        filtered_chunks = []
        for chunk in chunks:
            if is_trash(chunk):
                continue
            skip_ocr_fix = is_good_chunk(chunk)
            filtered_chunks.append((chunk, {"skip_ocr_fix": skip_ocr_fix}))


        for idx, chunk in enumerate(chunks):
            if is_good_chunk(chunk):
                 # Skip OCR fixes on this chunk — it's already good quality
                skip_ocr_fix = True
            else:
                skip_ocr_fix = False
                                # Run OCR fix suggestion logic here, e.g. logging
                # [LOG] Added to log: [OCR] Suggest fix: 'theogenesis' → 'thermogenesis'

        doc_id = insert_document(
            str(path), path.stem, file_hash, path.suffix[1:], EMBED_MODEL_NAME
        )

        accepted = 0
        for idx, (chunk, metadata) in enumerate(filtered_chunks): 
            page_num = "?"  # If you later add page data, update here
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
                    "page": page_num,
                    "skip_ocr_fix": metadata.get("skip_ocr_fix", False),
                }
            ))
        print(f"Accepted {accepted}/{len(chunks)} chunks from {path}")

    return docs