import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL_NAME
from data.db import init_db, is_metadata_db_empty
from llm import run_rag, parse_args
from logger import log_exception
from know.retriever import chunk_documents
from know.store import create_vector_store, load_vector_store
from ingest.chunker import split_into_chunks

def setup_retriever():
    args = parse_args()

# --- Consistent check for critical files ---
    metadata_exists = not is_metadata_db_empty()
    faiss_exists = os.path.exists(os.path.join(args.db_dir, "index.faiss"))

    if not metadata_exists or not faiss_exists:
        if not args.rebuild_db:
            print("[Eror] Missing metadata.db or FAISS index.")
            print("[Hint] Run with --rebuild-db to regenerate database and index.")
            sys.exit(1)

    init_db(rebuild=args.rebuild_db)
    print("Database initialized.")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    print("Loading model:", EMBED_MODEL_NAME)
    print("Embedding dimension:", len(embedding.embed_query("test")))

    if args.rebuild_db or is_metadata_db_empty() or not os.path.exists(os.path.join(args.db_dir, "index.faiss")):
        chunks = chunk_documents(args.data_dir, lambda text: split_into_chunks(text, update_map=args.rebuild_db))
        print(f"[Info] {len(chunks)} good chunks indexed.")

        if not chunks:
            raise ValueError("No chunks found. Check your data directory or chunking logic.")
        return create_vector_store(args.db_dir, chunks, embedding)
    else:
        return load_vector_store(args.db_dir, embedding)

def main():
    args = parse_args()
    retriever = setup_retriever()

    print("Interactive RAG CLI started. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            sources, response = run_rag(query, retriever, args.model_path)
            print("\nw\n", sources)
            print("\nAssistant:\n", response)
        except Exception as e:
            log_exception("Error during RAG pipeline", e, context=query)
    return retriever

if __name__ == "__main__":
    main()