import argparse
import llama_cpp
import os

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DATA_DIR, DB_DIR, MODEL_PATH, LLAMA_CPP_PARAMS
from know.provenance import run_rag_with_provenance

print("llama-cpp-python version:", llama_cpp.__version__)

# === LLM Generation ===
def generate_answer(question, context, model_path):
    # Generate a response from the LLM given the question and retrieved context.
    prompt = ChatPromptTemplate.from_template(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        "You are an insightful research assistant. Use the context below to construct a thoughtful, multi-layered answer. "
        "Do not speculate. If unsure, admit it honestly. Use [doc#] to cite sources.\n"
        "Question: {question} \n"
        "Context: {context} \n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    # Load LLaMA.cpp compatibal  model with GPU acceleration settings
    llm = LlamaCpp(**LLAMA_CPP_PARAMS)
   # Compose the prompt + llm + output parser chain
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# === RAG Pipeline (Retrieval-Augmented Generation) with PROVENANCE ===
def run_rag(question: str, retriever, model_path: str) -> tuple[list[str], str]:
    # Run the RAG pipeline with provenance, returning source paths and answer.
    sources, answer = run_rag_with_provenance(question, retriever, model_path)
    return sources, answer

# === CLI Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Local RAG CLI with FAISS and LLaMA")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory with input documents")
    parser.add_argument("--db-dir", type=str, default=DB_DIR, help="Directory to store/load FAISS index")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to GGUF LLaMA model")
    parser.add_argument("--rebuild-db", action="store_true", help="Force rebuild of FAISS vector store")
    return parser.parse_args()