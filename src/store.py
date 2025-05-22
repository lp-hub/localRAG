from langchain_community.vectorstores import FAISS

# === Vector Store Creation and Loading ===

def create_vector_store(db_dir, chunks, embedding):
    """
    Create a FAISS vector store from document chunks and save it locally.

    Args:
        db_dir (str): Directory path where FAISS index will be saved.
        chunks (list): List of LangChain Document chunks.
        embedding (Embedding model): Embedding function/model to vectorize documents.

    Returns:
        retriever: A retriever object that enables querying the vector store.
    """
    if not chunks:      
        raise ValueError("No document chunks provided for vector store creation.")
                
    print("Creating vector store with FAISS...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
    vectorstore.save_local(db_dir)
    return vectorstore.as_retriever()

def load_vector_store(db_dir, embedding):
    """
    Load an existing FAISS vector store from local disk.

    Args:
        db_dir (str): Directory path where FAISS index is stored.
        embedding (Embedding model): Embedding function/model used during index creation.

    Returns:
        retriever: A retriever object for querying the loaded vector store.
    """
    print("Loading existing FAISS vector store...")
    return FAISS.load_local(
        db_dir,
        embeddings=embedding,
        allow_dangerous_deserialization=True  # Needed due to known safety issues in deserialization
    ).as_retriever()
