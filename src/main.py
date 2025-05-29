import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings

from data.db import init_db, is_metadata_db_empty
from server.llm import run_rag, parse_args, start_llama_server
from server.logger import log_exception
from server.ramdisk import mount_ramdisk, copy_to_ramdisk, safe_load
from know.retriever import chunk_documents
from know.store import create_vector_store, load_vector_store
from ingest.chunker import split_into_chunks

# from config import EMBED_MODEL_SNAPHOTS, EMBED_MODEL_NAME_PATH, EMBED_MODEL_NAME # imported from .env

# ========== Server loading ==========
start_llama_server()
# mount_ramdisk() # COMMENT TO TURN OFF IF NOT USED
copy_to_ramdisk(["DB_DIR", "DATA_DIR", "EMBED_MODEL_NAME_PATH"])  # Add "DATA_DIR" if you rebuild indexes frequently.

# Set fallback env vars somewhere in your environment or config:
# e.g. DATA_DIR=/path/to/data, DB_DIR=/path/to/db on HDD
data_dir = safe_load("RAM_DATA_DIR", "DATA_DIR")  # will fall back if RAM copy missing
db_dir = safe_load("RAM_DB_DIR", "DB_DIR")
embed_model_dir = safe_load("RAM_EMBED_MODEL_NAME_PATH", "EMBED_MODEL_NAME_PATH")
# print(">>>" + data_dir)
# print(">>>" + db_dir)
# print(">>>" + embed_model_dir)

# ========== RAG loading ==========
def setup_retriever():
    print(f"Setting up retriever with DB dir: {db_dir} and Data dir: {data_dir}")
    args = parse_args()

    # Override args.db_dir and args.data_dir with RAM disk paths for speed
    args.db_dir = db_dir
    args.data_dir = data_dir

    # Consistent check for critical files
    # print(f"Checking if metadata DB exists at: {args.db_dir}")
    metadata_exists = not is_metadata_db_empty()
    faiss_exists = os.path.exists(os.path.join(args.db_dir, "index.faiss"))
    # print(f"Metadata exists: {metadata_exists}, FAISS index exists: {faiss_exists}")

    if not metadata_exists or not faiss_exists:
        if not args.rebuild_db:
            print("[Eror] Missing metadata.db or FAISS index.")
            print("[Hint] Run with --rebuild-db to regenerate database and index.")
            sys.exit(1)

    init_db(rebuild=args.rebuild_db)
    print("Database initialized.")

    # Use embed_model_dir from earlier safe_load()
    if not embed_model_dir:
        print("[Fatal] No valid embedding model directory found. Check your environment variables.")
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

    if args.rebuild_db or is_metadata_db_empty() or not os.path.exists(os.path.join(args.db_dir, "index.faiss")):
        chunks = chunk_documents(args.data_dir, lambda text: split_into_chunks(text, update_map=args.rebuild_db))
        print(f"[Info] {len(chunks)} good chunks indexed.")

        if not chunks:
            raise ValueError("No chunks found. Check your data directory or chunking logic.")
        return create_vector_store(args.db_dir, chunks, embedding)
    else:
        return load_vector_store(args.db_dir, embedding)

# ========== Ensure setup_retriever() is used ==========
def main():
    args = parse_args()
    retriever = setup_retriever()    
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
    main()