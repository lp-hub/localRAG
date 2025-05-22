import gradio as gr
from data import init_db

def list_titles_by_type(filetype):
    conn = init_db()
    cur = conn.cursor()
    cur.execute("SELECT title FROM documents WHERE source_type = ?", (filetype,))
    return [row[0] for row in cur.fetchall()]

def view_document(title):
    conn = init_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT c.content FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.title = ? ORDER BY c.chunk_index
    ''', (title,))
    chunks = cur.fetchall()
    return "\n---\n".join(chunk[0] for chunk in chunks)

def build_gradio_ui():
    with gr.Blocks() as demo:
        filetype = gr.Dropdown(choices=["txt", "pdf", "epub"], label="Filetype")
        titles = gr.Dropdown(choices=[], label="Title")
        output = gr.Textbox(label="Contents", lines=20)

        filetype.change(fn=list_titles_by_type, inputs=filetype, outputs=titles)
        titles.change(fn=view_document, inputs=titles, outputs=output)

    return demo

if __name__ == "__main__":
    ui = build_gradio_ui()
    ui.launch()
