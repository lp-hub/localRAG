import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

DB_PATH = Path("db/metadata.db")

def is_metadata_db_empty():
    if not DB_PATH.exists():
        return True
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0] == 0
    except sqlite3.OperationalError:
        # Table missing or malformed DB
        return True

def backup_old_db():
    if DB_PATH.exists():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_path = DB_PATH.with_name(f"metadata_{timestamp}.db")
        shutil.move(DB_PATH, backup_path)
        print(f"Old DB backed up as: {backup_path}")

def init_db(rebuild=False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure DB exists
    if rebuild and DB_PATH.exists():  # only backup if rebuild=True
        backup_old_db()
        if DB_PATH.exists():  # ensure DB still exists before unlink
            try:
                DB_PATH.unlink()  # delete DB if rebuild=True
            except FileNotFoundError:
                pass

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            title TEXT,
            hash TEXT UNIQUE,
            timestamp TEXT,
            source_type TEXT,
            embedding_model TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
    ''')

    conn.commit()
    return conn

def get_existing_hashes():
    conn = init_db()
    cur = conn.cursor()
    cur.execute("SELECT hash FROM documents")
    return set(row[0] for row in cur.fetchall())

def insert_document(path, title, hash_, source_type, embedding_model):
    conn = init_db()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO documents (path, title, hash, timestamp, source_type, embedding_model)
        VALUES (?, ?, ?, datetime('now'), ?, ?)
    ''', (path, title, hash_, source_type, embedding_model))
    conn.commit()
    return cur.lastrowid

def insert_chunks(document_id, chunks):
    conn = init_db()
    cur = conn.cursor()
    cur.executemany('''
        INSERT INTO chunks (document_id, chunk_index, content)
        VALUES (?, ?, ?)
    ''', [(document_id, i, chunk) for i, chunk in enumerate(chunks)])
    conn.commit()

def fetch_metadata_by_content(content_substring):
    conn = init_db()
    cur = conn.cursor()
    cur.execute('''
        SELECT d.title, d.timestamp, d.path FROM documents d
        JOIN chunks c ON c.document_id = d.id
        WHERE c.content LIKE ?
        LIMIT 1
    ''', (f"%{content_substring[:50]}%",))
    row = cur.fetchone()
    return {"title": row[0], "timestamp": row[1], "path": row[2]} if row else {}