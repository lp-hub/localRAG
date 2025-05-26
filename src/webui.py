import gradio as gr
import socket
import threading
import time

from config import MODEL_PATH
from main import setup_retriever
from know.provenance import run_rag_with_provenance

retriever = None

def print_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Web UI running at http://{local_ip}:7860")

def gradio_rag(query, history):
    try:
        print(f"Got query: {query}")
        sources, answer = run_rag_with_provenance(query, retriever, MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to run RAG: {e}")
        sources, answer = "Error:", str(e)
    return answer + "\n\nSources: " + sources

iface = gr.ChatInterface(
    fn=gradio_rag,
    title="Local RAG OCR",
    description="Ask questions over your local documents using a LLaMA-backed RAG system.",
    theme="soft",
)

def launch_gradio():
    chat = gr.Chatbot()
    iface = gr.ChatInterface(
        fn=gradio_rag,
        chatbot=chat,
        title="Local RAG OCR",
        description="Ask questions over your local documents using a LLaMA-backed RAG system.",
        theme="soft",
    )

    print_local_ip()
    iface.launch()

if __name__ == "__main__":
    def retriever_loader():
        global retriever
        retriever = setup_retriever()

    thread = threading.Thread(target=retriever_loader)
    thread.start()

    while retriever is None:
        print("Waiting for retriever...")
        time.sleep(1)

    launch_gradio()