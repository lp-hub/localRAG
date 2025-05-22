from db import init_db

def list_documents():
    conn = init_db()
    cur = conn.cursor()
    cur.execute("SELECT id, path, timestamp FROM documents")
    for row in cur.fetchall():
        print(row)

def delete_document_by_path(path):
    conn = init_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE path = ?", (path,))
    row = cur.fetchone()
    if row:
        doc_id = row[0]
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        print(f"Deleted: {path}")
    else:
        print("Document not found.")