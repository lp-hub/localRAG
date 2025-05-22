import sqlite3
from data import init_db

def query_documents(filetype=None, date_after=None, skip_tags=None):
    conn = init_db()
    cur = conn.cursor()

    sql = "SELECT path, title, timestamp FROM documents WHERE 1=1"
    args = []

    if filetype:
        sql += " AND source_type = ?"
        args.append(filetype)
    if date_after:
        sql += " AND timestamp > ?"
        args.append(date_after)
    # skip_tags is hypothetical for later use (if you store tags)

    cur.execute(sql, args)
    return cur.fetchall()