import os
from pathlib import Path
import shutil
import sqlite3
import sys
from datetime import datetime
from langchain.schema import Document

def db_path():
    return Path("db") / os.getenv("TOPIC", "default") / "metadata.db"

def is_metadata_db_empty() -> bool:
    """Check if metadata.db exists and contains chunks."""
    if not db_path().exists():
        return True
    try:
        with sqlite3.connect(db_path()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0] == 0
    except sqlite3.OperationalError:
        return True

def backup_old_db():
    """Back up the existing metadata.db before overwriting."""
    if not db_path().exists():
        print("[Warn] backup_old_db() called, but metadata.db does not exist.")
        return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_path = db_path().with_name(f"metadata_{timestamp}.db")
        shutil.move(db_path(), backup_path)
        print(f"[Backup] Old DB moved to: {backup_path}")
    except Exception as e:
        print(f"[Error] Failed to back up old DB: {e}")

def init_db(rebuild=False) -> sqlite3.Connection:
    """Initialize the SQLite database and schema."""
    db_path().parent.mkdir(parents=True, exist_ok=True) # create db directory

    db_already_exists = db_path().exists()

    if rebuild:
        if db_already_exists:
            try:
                backup_old_db()
                db_path().unlink() # Might raise FileNotFoundError if backup moved it
                print("[Info] Deleted existing metadata.db")
            except FileNotFoundError:
                print("[Warn] Tried to delete metadata.db, but it was already missing.")
            except Exception as e:
                print(f"[Error] Unexpected error while deleting DB: {e}")
                sys.exit(1) # File is now gone
        else:
            print("[Info] No existing DB found — skipping backup and deletion.")
    
    conn = sqlite3.connect(db_path())
    if db_already_exists:
        print(f"Loaded existing metadata: {db_path().name}")
    else:
        print(f"[Info] Creating new metadata.db")
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")  # ENABLE enforcement
    # ON DELETE CASCADE - critical for cleanup
    # Deleting a document will automatically delete all chunks tied to garbage - clean and safe.

    cur.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            title TEXT,
            hash TEXT UNIQUE,
            timestamp TEXT,
            source_type TEXT,
            embedding_model TEXT,
            author TEXT,
            date TEXT,
            language TEXT,
            tags TEXT,
            source_url TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            page_num INTEGER,
            char_start INTEGER,
            char_end INTEGER,
            section TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
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

    # Try to fetch existing document ID by hash
    cur.execute("SELECT id FROM documents WHERE hash = ?", (hash_,))
    existing = cur.fetchone()
    if existing:
        return existing[0]  # document already exists

    # If not found, insert new document
    cur.execute('''
        INSERT INTO documents (path, title, hash, timestamp, source_type, embedding_model)
        VALUES (?, ?, ?, datetime('now'), ?, ?)
    ''', (path, title, hash_, source_type, embedding_model))
    conn.commit()
    return cur.lastrowid

def insert_chunks(doc_id, chunks: list[tuple[str, dict]]):
    conn = init_db()
    cur = conn.cursor()

    # Optional: Check if chunks already exist for this doc_id
    cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,))
    if cur.fetchone()[0] > 0:
        print(f"[Skip] Chunks already exist for doc_id {doc_id}")
        return
    
    cur.executemany('''
        INSERT INTO chunks (document_id, chunk_index, content)
        VALUES (?, ?, ?)
    ''', [(doc_id, i, chunk_text) for i, (chunk_text, _) in enumerate(chunks)])
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

def get_all_chunks(topic: str) -> list[Document]:
    """Fetch all chunks from DB as LangChain Document objects with metadata."""
    db_file = Path("db") / topic / "metadata.db"
    if not db_file.exists():
        print(f"[Error] metadata.db not found for topic: {topic}")
        return []

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('''
        SELECT c.content, c.chunk_index, d.id, d.path, d.title
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY d.id, c.chunk_index
    ''')

    rows = cur.fetchall()
    return [
        Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "path": path,
                "title": title,
                "chunk_index": chunk_index,
            }
        )
        for content, chunk_index, doc_id, path, title in rows
    ]