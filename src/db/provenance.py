"""
This module provides a RAG runner that includes metadata provenance
for each retrieved chunk, injecting metadata into the prompt and
returning both sources list and answer.
"""
from typing import List, Tuple
from langchain.schema import Document
import os

def run_rag_with_provenance(
    question: str,
    retriever,
    model_path: str
) -> Tuple[str, str]:
    """
    Run RAG pipeline, retrieving documents with FAISS retriever then
    constructing a prompt that includes provenance metadata.

    Args:
        question (str): The user question.
        retriever: A LangChain retriever (e.g., FAISS-based).
        model_path (str): Path to the LLM model.

    Returns:
        sources (List[str]): List of source file paths for retrieved chunks.
        answer (str): The LLM-generated answer.
    """
    # Import here to avoid circular dependency
    from llm import generate_answer

    # Retrieve chunks as LangChain Document objects
    docs: List[Document] = retriever.get_relevant_documents(question)

    # Build context with metadata tags
    context_blocks: List[str] = []
    sources_info = set()

    for doc in docs:
        md = doc.metadata or {}
        title = md.get("title", "unknown")
        path = md.get("path", "unknown")
        page = md.get("page", "?")
        chunk_index = md.get("chunk_index", None)

        tag = f"[{title}" + (f"—chunk {chunk_index}" if chunk_index is not None else "") + "]"
        context_blocks.append(f"{tag} {doc.page_content}")

        filename = os.path.basename(path)
        snippet = doc.page_content[:80].replace("\n", " ").strip() + "..."
        line = f"{filename} ?page" if page == "?" else f"{filename} page {page}"
        sources_info.add(f"{line}\n  ↳ {snippet}")

    context_text = "\n\n".join(context_blocks)
    answer = generate_answer(question, context_text, model_path)

    sources_text = "\n\n".join(sorted(sources_info))
    return sources_text, answer