# === Retrieval Helpers ===

def retrieve_documents(retriever, question):
    return retriever.get_relevant_documents(question)

def format_context(docs): 
    # Format retrieved documents into a single string to provide context for LLM input.
    return "\n\n".join(f"[doc{idx+1}]={doc.page_content.replace('\n', ' ')}"
                       for idx, doc in enumerate(docs))

def format_sources(docs):
    print(format_sources(docs))
    # Format unique source files from retrieved documents.
    seen = set()
    output = []

    for idx, doc in enumerate(docs):
        src = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', '?')
        snippet = doc.page_content[:80].replace('\n', ' ') + "..."
        output.append(f"[doc{idx+1}] {src} (page {page}) — {snippet}")

    return "\n\n".join(output)
