import os
from dotenv import load_dotenv
load_dotenv()
def getenv_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")
def getenv_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))
def getenv_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

DATA_DIR = os.getenv("DATA_DIR")
DB_DIR = os.getenv("DB_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_RAM = os.getenv("MODEL_RAM")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME") # FAISS and LangChain's VectorstoreRetriever
                                # "intfloat/multilingual-e5-small" - 100 languages, compatible but basic
                                # "BAAI/bge-small-en" - for English-only documents
LLAMA_CPP_PARAMS = {
    "model_path": MODEL_PATH,   # Path to your GGUF model file
    "temperature": 0.7,         # Sampling temperature; lower = deterministic, higher = more creative
    "top_p": 0.95,              # Nucleus sampling threshold for controlled randomness
    "top_k": 50,                # Restrict to top-K tokens
    "repeat_penalty": 1.1,      # Discourage repetition
    "n_ctx": 4096,              # The number of tokens in the context window size
    "max_tokens": 4096,         # or more, if your model and RAM/GPU can handle it
    "n_gpu_layers": 36,         # If you get CUDA OOM errors, lower this number.
    "n-parallel": 12,           # Number of parallel sequences for token evaluation
    "n_threads": 12,            # Tune for CPU parallelism if no GPU
    "n-batch": 512,             # Batch size for tokens evaluated at once; tune based on VRAM
    "memory-f32": 32,           # Use 32-bit float memory for activations — much more RAM/GPU use, but more accurate         
    "f16_kv": True,             # Use FP16 key/value cache, saves RAM
    "use_mlock": False,         # If True, lock model in RAM to avoid swapping (requires root)
    "use_mmap": False,          # If False, loads model into RAM (vs memory-mapping from disk)
    "verbose": True,            # Log info from backend
}

GARBAGE_THRESHOLD = 0.7         # def chunk_documents(...) in retriever.py

# CHUNK_SIZE controls how large each document segment is (in tokens or characters depending on the loader).
# Larger chunks give more context to the LLM, but require more memory and reduce retrieval precision.
# A typical value is 512 tokens.
CHUNK_SIZE = 512

# CHUNK_OVERLAP defines how much overlap there is between consecutive chunks.
# This helps preserve context across boundaries, so sentences that span two chunks aren't cut off.
# A typical value is 10–20% of CHUNK_SIZE (e.g., 64 if CHUNK_SIZE is 512).
CHUNK_OVERLAP = 64

# These parameters control how your documents are chunked before being embedded and indexed in FAISS. 
# Well-tuned values help avoid missing relevant context during retrieval and ensure smoother RAG performance.