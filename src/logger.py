import logging
import os

from logging.handlers import RotatingFileHandler
from datetime import datetime

# === Config Paths ===
LOG_DIR = "logs"
ROTATING_LOG_FILE = "rag_errors.log"
MANUAL_LOG_FILE = "log.txt"
LOG_FILENAME = os.path.join(os.path.dirname(__file__), LOG_DIR, ROTATING_LOG_FILE)

# Create logs/ directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# === Rotating File Logger ===
handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, ROTATING_LOG_FILE),
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=3               # Keep 3 backup logs
)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger("RAG")
logger.setLevel(logging.WARNING)  # Only warnings and errors
logger.addHandler(handler)
logger.propagate = False # Avoid duplicate logs if root logger is used elsewhere


# === Manual Timestamped Log Function ===
def save_manual_log(message: str):
    """
    Write a single message to logs/log.txt with timestamp.

    Args:
        message (str): Message string to log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} - {message}\n"
    with open(os.path.join(LOG_DIR, MANUAL_LOG_FILE), "a", encoding="utf-8") as f:
        f.write(entry)


# === Unified Exception Logging Function ===
def log_exception(message: str, exception: Exception, context: str = None):
    """
    Log an exception both as rotating logger error and also in manual log.txt with timestamp.

    Args:
        message (str): General description of what failed.
        exception (Exception): The caught exception object.
        context (str): Optional specific info (e.g., file name, query, operation).
    """
    error_msg = f"{message}: {str(exception)}"
    if context:
        error_msg += f" | Context: {context}"

    # Log to rotating logger (rag_errors.log)
    logger.error(error_msg)

    # Also log manually with timestamp in plain .txt
    save_manual_log(error_msg)



