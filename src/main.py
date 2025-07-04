import os
import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from data.db import init_db, is_metadata_db_empty, get_existing_hashes, get_all_chunks
from server.llm import run_rag, parse_args, start_llama_server
from server.logger import log_exception
from server.ramdisk import mount_ramdisk, copy_to_ramdisk, safe_load
from server.watchdog import start_watchdog
from context.retriever import chunk_documents, hash_file, write_stats
from context.store import create_vector_store, load_vector_store
from context.chunker import split_into_chunks

# from config import EMBED_MODEL_SNAPHOTS, EMBED_MODEL_NAME_PATH, EMBED_MODEL_NAME # imported from .env

# Set fallback env vars somewhere in your environment or config:
# e.g. DATA_DIR=/path/to/data, DB_DIR=/path/to/db on HDD
# data_dir = safe_load("RAM_DATA_DIR", "DATA_DIR")
db_dir = safe_load("RAM_DB_DIR", "DB_DIR") # will fall back if RAM copy missing
embed_model_dir = safe_load("RAM_EMBED_MODEL_NAME_PATH", "EMBED_MODEL_NAME_PATH")
# print(">>>" + data_dir)
# print(">>>" + db_dir)
# print(">>>" + embed_model_dir)

# ========== RAG loading ==========
def setup_retriever(args):
    topic = args.topic
    data_path = os.path.join(args.data_dir, topic)
    db_path = os.path.join(args.db_dir, topic)

    print(f"Using data dir: {data_path}")
    print(f"Using db dir: {db_path}")

    os.makedirs(db_path, exist_ok=True)

    # Consistent check for critical files
    print(f"Checking if metadata DB exists at: {args.db_dir}")
    metadata_path = os.path.join(db_path, "metadata.db")
    faiss_path = os.path.join(db_path, "index.faiss")
    metadata_exists = os.path.exists(metadata_path)
    faiss_exists = os.path.exists(faiss_path)
    print(f"Metadata exists: {metadata_exists}, FAISS index exists: {faiss_exists}")

    # Use embed_model_dir from earlier safe_load()
    if not embed_model_dir:
        print("[Fatal] EMBED_MODEL_NAME_PATH not set. Check your .env or environment.")
        sys.exit(1)

    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_dir + os.getenv("EMBED_MODEL_SNAPHOTS"),
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        # # This line forces it to use Transformers backend instead of SentenceTransformers
        # cache_folder=None,  # optional, to prevent slow re-download
    )
    print(f"Loading embedding model: {embed_model_dir}")
    print(f"Embedding dimension: {len(embedding.embed_query('test'))}")

    # ========== Step 0: Check if critical files exist ==========
    if not metadata_exists or not faiss_exists:
        if not (args.rebuild_db or args.rebuild_index):
            print("[Error] Missing metadata.db or FAISS index.")
            print("[Hint] Run with --rebuild-db or --rebuild-index to initialize database and index.")
            sys.exit(1)

    # ========== Step 1: Ensure DB exists ==========
    metadata_empty = not metadata_exists or is_metadata_db_empty()
    need_rebuild = args.rebuild_db or metadata_empty

    print(f"[DB] {'(Re)initializing' if need_rebuild else 'Using existing'} metadata.db")
    init_db(rebuild=need_rebuild)

    # ========== Step 2: Index files if needed ==========
    if args.rebuild_db or args.rebuild_index:
        new_files = []
        existing_hashes = get_existing_hashes()

        for path in Path(data_path).rglob("*"):
            if not path.is_file():
                continue
            if hash_file(path) not in existing_hashes:
                new_files.append(path)

        if new_files:
            print(f"[DB] Found {len(new_files)} new files to index.")
            chunk_documents(data_path, lambda text, path: split_into_chunks(text, filename=path))
        else:
            print("[DB] No new files to index. Skipping chunking.")
    else:
        print("[Info] No rebuild flags — skipping file scan.")

    # === Step 3: Load all chunks from DB ===
    chunks = get_all_chunks(topic)
    if not chunks:
        raise ValueError("No chunks available to build FAISS index.")

    # === Step 4: Write stats and build index ===
    write_stats(
        doc_count=len({doc.metadata['doc_id'] for doc in chunks}),
        chunk_count=len(chunks),
        topic=topic,
        model_name=os.getenv("EMBED_MODEL_SNAPHOTS")
    )
    print(f"[Info] {len(chunks)} chunks indexed.")
    return create_vector_store(db_path, chunks, embedding) if args.rebuild_index or not faiss_exists else load_vector_store(db_path, embedding)
    
# First time (wipe everything):
# python src/main.py --topic tech --rebuild-db
# Resume interrupted session, add new files only, keep
# python src/main.py --topic tech --rebuild-index
# Normal usage (nothing is rebuilt unless missing):
# python src/main.py --topic tech

# ========== Ensure setup_retriever() is used ==========
def main():
    args = parse_args()
    os.environ["TOPIC"] = args.topic
    retriever = setup_retriever(args)
    print("=== Local RAG Client Ready ===")
    print("Use this program to ask questions over your document database.")
    print("Interactive RAG CLI started. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        try:
            sources, response = run_rag(query, retriever)
            print("\nw\n", sources)
            print("\nAssistant:\n", response)
        except Exception as e:
            log_exception("Error during RAG pipeline", e, context=query)
    return retriever

if __name__ == "__main__":
    start_llama_server()
    mount_ramdisk() # COMMENT TO TURN OFF IF NOT USED   
    copy_to_ramdisk(["DB_DIR", "EMBED_MODEL_NAME_PATH"])  # Add "DATA_DIR" if you rebuild indexes frequently.
    start_watchdog() # COMMENT TO TURN OFF IF NOT USED
    main()