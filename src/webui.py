import gradio as gr
import socket
import threading
import time

from config import MODEL_PATH
from main import setup_retriever
from db.provenance import run_rag_with_provenance

retriever = None

def print_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Web UI running at http://{local_ip}:7860")

def gradio_rag(query):
    try:
        print(f"Got query: {query}")
        sources, answer = run_rag_with_provenance(query, retriever, MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to run RAG: {e}")
        sources, answer = "Error:", str(e)
    return sources, answer

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# Local RAG Q&A")

        query_input = gr.Textbox(label="Enter your question here", lines=2)
        submit_button = gr.Button("Submit")

        sources_output = gr.Textbox(label="Sources", lines=5)
        answer_output = gr.Textbox(label="Answer", lines=10)

        # Trigger on Enter
        query_input.submit(fn=gradio_rag, inputs=query_input, outputs=[sources_output, answer_output])
        # Trigger on Button click
        submit_button.click(fn=gradio_rag, inputs=query_input, outputs=[sources_output, answer_output])

        print_local_ip()
        demo.launch()

if __name__ == "__main__":
    # Run main() in a thread so CLI is non-blocking
    def retriever_loader():
        global retriever
        retriever = setup_retriever()  # main() should return retriever
    # Load retriever in a thread
    thread = threading.Thread(target=retriever_loader)
    thread.start()

# Wait until retriever is ready
while retriever is None:
    print("Waiting for retriever...")
    time.sleep(1)

launch_gradio()