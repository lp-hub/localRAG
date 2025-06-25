from time import time
from langchain_community.vectorstores import FAISS

# Create a FAISS vector store from document chunks and save it locally.
def create_vector_store(db_dir, chunks, embedding):    
    if not chunks:      
        raise ValueError("No document chunks provided for vector store creation.")
                
    print("Creating vector store with FAISS...")
    start = time()
    try: 
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
        vectorstore.save_local(db_dir)
        elapsed = time() - start
        print(f"[FAISS] Vector store saved to {db_dir} in {elapsed:.2f} seconds.")
        return vectorstore.as_retriever()
    except Exception as e:
        print(f"[ERROR] Failed to create FAISS vector store: {e}")
        raise

# Load an existing FAISS vector store from local disk.
def load_vector_store(db_dir, embedding):
    print(f"[FAISS] Loading vector store from {db_dir}...")
    try:
        return FAISS.load_local(
            db_dir,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        ).as_retriever()
    except Exception as e:
        print(f"[ERROR] Failed to load FAISS index: {e}")
        raise