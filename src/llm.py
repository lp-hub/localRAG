import argparse
import datetime
import os
from pydantic import Field
import sys
import requests
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DATA_DIR, DB_DIR
from know.provenance import run_rag_with_provenance

# print(f"Connecting to llama server at {LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}...")
# print(f"Using model: {model_name} ({model_size} params) on device: {device_name}")
# print(f"Context size: {LLAMA_CPP_PARAMS['n_ctx']}")
# print(f"GPU layers: {LLAMA_CPP_PARAMS['n_gpu_layers']}")
# print(f"Batch size: {LLAMA_CPP_PARAMS['n_batch']}")
# print(f"FAISS index loaded from: {DB_DIR}, documents indexed: {num_docs}")
print("=== Local RAG Client Ready ===")
print(f"Start time: {datetime.datetime.now().isoformat()}")
print(f"Python version: {sys.version.split()[0]}")
# print(f"Running on host: {os.uname().nodename}")
print("Use this program to ask questions over your document database.")
print("Server at: 127.0.0.1:8080")


# === Connect LLM Server ===
class LlamaCppServerClient(LLM):
    server_url: str = Field(default="http://127.0.0.1:8080")
    max_tokens: int = 128
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_server"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # llama.cpp server uses /v1/completions POST endpoint with JSON payload
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop or [],
        }
        response = requests.post(f"{self.server_url}/v1/completions", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # This depends on your server's JSON format; adjust as necessary
        return data["choices"][0]["text"]

# === LLM Generation ===
def generate_answer(question, context):
    prompt = ChatPromptTemplate.from_template(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        "You are an insightful research assistant. Use the context below to construct a thoughtful, multi-layered answer. "
        "Do not speculate. If unsure, admit it honestly. Use [doc#] to cite sources.\n"
        "Question: {question} \n"
        "Context: {context} \n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    llm = LlamaCppServerClient(server_url="http://127.0.0.1:8080")  # or inject if needed
    chain = prompt | llm | StrOutputParser()
    # print("[DEBUG] Invoking LLM with context length:", len(context))
    return chain.invoke({"question": question, "context": context})

# === RAG Pipeline (Retrieval-Augmented Generation) with PROVENANCE ===
def run_rag(question: str, retriever: str) -> tuple[list[str], str]:
    # Run the RAG pipeline with provenance, returning source paths and answer.
    sources, answer = run_rag_with_provenance(question, retriever)
    return sources, answer

# === CLI Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Local RAG CLI with FAISS and LLaMA")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory with input documents")
    parser.add_argument("--db-dir", type=str, default=DB_DIR, help="Directory to store/load FAISS index")
    parser.add_argument("--rebuild-db", action="store_true", help="Force rebuild of FAISS vector store")
    return parser.parse_args()