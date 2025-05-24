import os
from pathlib import Path
import shutil
import sqlite3
import sys
from datetime import datetime
from data.jsonhandler import ensure_normalization_json, JSON_PATH

DB_PATH = Path("db/metadata.db")

def is_metadata_db_empty() -> bool:
    """Check if metadata.db exists and contains chunks."""
    if not DB_PATH.exists():
        return True
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0] == 0
    except sqlite3.OperationalError:
        return True

def backup_old_db():
    """Back up the existing metadata.db before overwriting."""
    if not DB_PATH.exists():
        print("[Warn] backup_old_db() called, but metadata.db does not exist.")
        return
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_path = DB_PATH.with_name(f"metadata_{timestamp}.db")
        shutil.move(DB_PATH, backup_path)
        print(f"[Backup] Old DB moved to: {backup_path}")
    except Exception as e:
        print(f"[Error] Failed to back up old DB: {e}")

def init_db(rebuild=False) -> sqlite3.Connection:
    """Initialize the SQLite database and schema."""
    if not JSON_PATH.exists() and not rebuild:
        print(f"[Error] Normalization map not found at {JSON_PATH}")
        print("[Hint] Run with --rebuild-db to generate it.")
        sys.exit(1)

    if rebuild:
        ensure_normalization_json(force=True)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True) # create db directory

    db_already_exists = DB_PATH.exists()

    if rebuild:
        if db_already_exists:
            try:
                backup_old_db()
                DB_PATH.unlink() # Might raise FileNotFoundError if backup moved it
                print("[Info] Deleted existing metadata.db")
            except FileNotFoundError:
                print("[Warn] Tried to delete metadata.db, but it was already missing.")
            except Exception as e:
                print(f"[Error] Unexpected error while deleting DB: {e}")
                sys.exit(1) # File is now gone
        else:
            print("[Info] No existing DB found — skipping backup and deletion.")
    
    conn = sqlite3.connect(DB_PATH)
    if db_already_exists:
            print(f"Loaded existing metadata.db")
    else:
            print(f"[Info] Creating new metadata.db")  
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

def insert_chunks(doc_id, chunks: list[tuple[str, dict]]):
    conn = init_db()
    cur = conn.cursor()
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