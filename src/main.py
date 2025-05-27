import os
import shutil
import subprocess
import sys
from langchain_huggingface import HuggingFaceEmbeddings

from data.db import init_db, is_metadata_db_empty
from server.llm import run_rag, parse_args
from server.logger import log_exception
from know.retriever import chunk_documents
from know.store import create_vector_store, load_vector_store
from ingest.chunker import split_into_chunks

from config import DATA_DIR, DB_DIR, EMBED_MODEL_NAME_PATH, EMBED_MODEL_SNAPHOTS
START_LAMMA = "./scripts/start_llama_server.sh"
ramdisk_root = os.getenv("RAMDISK_ROOT")

# --- Start server --
def start_llama_server():
    if not os.path.exists(START_LAMMA):
        raise FileNotFoundError(f"Script not found: {START_LAMMA}")
    print(f"[Info] Launching llama-server using: {START_LAMMA}")

    try:
        subprocess.Popen(
            ["bash", START_LAMMA],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("[Info] llama-server started in background.")
    except Exception as e:
        print(f"[Error] Failed to start llama-server: {e}")
start_llama_server()


# --- Copy Specific Directories to RAM Disk ---
def copy_to_ramdisk(env_vars, ramdisk_path=ramdisk_root):
    for var in env_vars:
        original_path = os.getenv(var)
        if original_path is None:
            print(f"[Warning] Environment variable {var} is not set; skipping.")
            continue

        if not os.path.exists(original_path):
            print(f"[SKIP] {var} does not point to a real path: {original_path}")
            continue

        base_name = os.path.basename(original_path.rstrip("/"))
        ram_path = os.path.join(ramdisk_path, base_name)

        print(f"Copying {var} from {original_path} to RAM disk at {ram_path}...")
        # If destination exists, remove it first to avoid copytree errors
        try:

            if os.path.exists(ram_path):
                print(f"RAM path {ram_path} exists, removing before copy...")
                shutil.rmtree(ram_path)

            shutil.copytree(original_path, ram_path)
                        
            ram_var = "RAM_" + var # Set separate RAM var
            os.environ[ram_var] = ram_path # Override environment variable to RAM disk => RAM_
            print(f"{ram_var} set to {ram_path}")
            print(f"{var} copied to RAM disk successfully.")
        except Exception as e:
            print(f"Failed to copy {original_path} to RAM disk: {e}")

# --- Mount RAMdisk ---
def mount_ramdisk():
    script_path = os.path.join(os.path.dirname(__file__), 'server/mount_ramdisk.sh')
    print(f"Mounting ramdisk using script at: {script_path}")
    try:
        subprocess.run(['sudo', script_path], check=True)
        print("RAM disk mounted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to mount ramdisk: {e}")
        exit(1)
# If you want to run it without sudo prompts, you can allow passwordless execution via /etc/sudoers:
# your_username ALL=(ALL) NOPASSWD: /full/path/to/mount_ramdisk.sh
# Then change the Python call to:
# subprocess.run(['sudo', '/full/path/to/mount_ramdisk.sh'], check=True)
mount_ramdisk() # COMMENT TO TURN OFF IF NOT USED
copy_to_ramdisk(["DB_DIR", "DATA_DIR", "EMBED_MODEL_NAME_PATH"])  # Add "DATA_DIR" if you rebuild indexes frequently.


# --- Fallback to HDD if RAM Disk Fails ---
def safe_load(path_var, fallback_env):
    path = os.getenv(path_var)
    print(f"Trying to load from {path_var}: {path}")
    if path and os.path.exists(path):
        print(f"Successfully found {path_var} at: {path}")
        return path
    else:
        print(f"[Warning] {path_var} not found or path does not exist: {path}")
        fallback_path = os.getenv(fallback_env)
        print(f"Trying fallback path {fallback_env}: {fallback_path}")
        if fallback_path and os.path.exists(fallback_path):
            print(f"Successfully found fallback {fallback_env} at: {fallback_path}")
            return fallback_path
        else:
            print(f"[Error] Neither {path_var} nor fallback {fallback_env} paths exist.")
            return None

# Set fallback env vars somewhere in your environment or config:
# e.g. DATA_DIR_HDD=/path/to/data, DB_DIR_HDD=/path/to/db
data_dir = safe_load("RAM_DATA_DIR", "DATA_DIR")  # will fall back if RAM copy missing
db_dir = safe_load("RAM_DB_DIR", "DB_DIR")
embed_model_dir = safe_load("RAM_EMBED_MODEL_NAME_PATH", "EMBED_MODEL_NAME_PATH")
# print(">>>" + data_dir)
# print(">>>" + db_dir)
# print(">>>" + embed_model_dir)


# --- RAG loading ---
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
        model_name=embed_model_dir + EMBED_MODEL_SNAPHOTS,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
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

# Just ensure setup_retriever() is used
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

    