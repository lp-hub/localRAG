# Multiple file types: .faiss, .db, .log, .json
# Recursive monitoring: /mnt/ramdisk/folder
# Atomic write via temp + os.replace()
# Per-file debouncing: avoids rapid-fire re-syncs
# Format-aware validation before overwriting disk
import os
import shutil
import time
import faiss
import sqlite3
import threading
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# pip install watchdog
SRC_DIR = "/mnt/ramdisk/db"
USER_DIR = os.path.expanduser(os.environ["USER_DIR"])
DST_DIR = os.path.join(USER_DIR, "db")
TEMP_DIR = os.path.join(DST_DIR, ".tmp_sync")

print(f"Watchdog resolved DST_DIR: {DST_DIR}")

# ========== Validation Logic ==========
def is_valid_faiss(path):
    try:
        _ = faiss.read_index(path)
        return True
    except:
        return False

def is_valid_sqlite(path):
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = conn.execute("PRAGMA integrity_check;")
        result = cur.fetchone()[0]
        conn.close()
        return result == "ok"
    except:
        return False

def is_valid_json(path):
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except:
        return False

# ========== Sync Logic ==========
def initial_sync():
    for root, _, files in os.walk(SRC_DIR):
        for f in files:
            full_path = os.path.join(root, f)
            sync_file_to_disk(full_path)

def has_file_changed(src, dst): # checks both size and mtime, avoiding slow hashing
    if not os.path.exists(dst):
        return True
    src_stat = os.stat(src)
    dst_stat = os.stat(dst)
    return src_stat.st_size != dst_stat.st_size or src_stat.st_mtime > dst_stat.st_mtime

def validate_file(path):
    if path.endswith(".faiss"):
        return is_valid_faiss(path)
    elif path.endswith(".db"):
        return is_valid_sqlite(path)
    elif path.endswith(".json"):
        return is_valid_json(path)
    elif path.endswith(".log"):
        return True
    else:
        print(f"⚠️ Ignored file type: {path}")
        return False
    
def sync_file_to_disk(src_path):
    abs_src = os.path.abspath(src_path)
    if abs_src.startswith(os.path.abspath(DST_DIR)) or ".tmp_sync" in abs_src:
        print(f"🚫 Skipping self-triggered or temp path: {abs_src}")
        return

    rel_path = os.path.relpath(src_path, SRC_DIR)
    tmp_path = os.path.join(TEMP_DIR, rel_path)
    dst_path = os.path.join(DST_DIR, rel_path)

    if not has_file_changed(src_path, dst_path):
        # print(f"⚖️ Skipping unchanged: {rel_path}") # prints into prompt => You: TODO
        return

    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    shutil.copy2(src_path, tmp_path)

    if not validate_file(tmp_path):
        print(f"❌ Validation failed for {rel_path}. Skipping sync.")
        os.remove(tmp_path)
        return

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    os.replace(tmp_path, dst_path)
    print(f"✅ Synced: {rel_path}")

# ========== Watchdog Handler ==========
class RagSyncHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_synced = {}
        self.min_interval = 5  # seconds per-file

    def _should_skip(self, path):
        return (
            path.endswith("~") or
            ".tmp_sync" in path or
            not path.startswith(SRC_DIR)
        )

    def on_any_event(self, event):
        if event.is_directory or self._should_skip(event.src_path):
            return

        rel_path = os.path.relpath(event.src_path, SRC_DIR)
        now = time.time()

        # Debounce rapid writes
        if rel_path in self.last_synced and now - self.last_synced[rel_path] < self.min_interval:
            # print(f"⏱️ Skipping rapid re-sync of {rel_path}") # prints into prompt => You: TODO
            return

        self.last_synced[rel_path] = now
        sync_file_to_disk(event.src_path)

# ========== Entry Point ==========
def start_watchdog(path=SRC_DIR):
    os.makedirs(DST_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"👁️ Watching: {path}")
    print(f"📤 Backing up to: {DST_DIR}")

    initial_sync()  # one-time sync

    observer = Observer()
    observer.schedule(RagSyncHandler(), path=path, recursive=True)
    observer.start()

    def monitor():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

if __name__ == "__main__":
    start_watchdog()