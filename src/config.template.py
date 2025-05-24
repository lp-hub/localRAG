from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
DB_DIR = os.getenv("DB_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small" # FAISS and LangChain's VectorstoreRetriever
                                # "intfloat/multilingual-e5-small" - 100 languages, compatible but basic
                                # "BAAI/bge-small-en" - for English-only documents
GARBAGE_THRESHOLD = 0.7         # def chunk_documents(...) in retriever.py

LLAMA_CPP_PARAMS = {
    "model_path": MODEL_PATH,
    "temperature": 0.8,         # Add creative variation
    "top_p": 0.8,               # Nucleus sampling for controlled randomness
    "top_k": 35,                # Restrict to top-K tokens
    "repeat_penalty": 1.1,      # Discourage repetition
    "n_ctx": 4096,              # The number of tokens in the context
    "n_gpu_layers": 35,         # To fit GPU memory
    "f16_kv": True,
    "use_mlock": False,
    "use_mmap": True,
    "verbose": True,
    "n_threads": 12,            # Tune for CPU parallelism if no GPU
    "max_tokens": 4096,         # or more, if your model and RAM/GPU can handle it
}

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