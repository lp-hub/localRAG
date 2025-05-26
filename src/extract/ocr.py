import fitz  # PyMuPDF
import io
import os
import pytesseract
import re
import subprocess
# PYTHONPATH=./src python scripts/ocr.py
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

DST_DIR = Path(os.getenv("DST_DIR")) # extracted text files
SRC_DIR = Path(os.getenv("SRC_DIR")) # original files => pdf, etc
OCRD_LOG=Path(os.getenv("OCRD_LOG")) # optically character recognised files list prevents overwriting
OCR_CANDIDATES=Path(os.getenv("OCR_CANDIDATES")) # list of files to be OCRed appends from DST_DIR

print("[.env] DST_DIR:", DST_DIR)
print("[.env] SRC_DIR:", SRC_DIR)
print("[.env] OCRD_LOG:", OCRD_LOG)
print("[.env] OCR_CANDIDATES:", OCR_CANDIDATES)
print("Searching...")
# ========================================================================
# ========================================================================
# ========================================================================
def detect_language_from_filename(file_path: Path) -> str:
    name = file_path.name.lower()

    lang_keywords = {
        # Cyrillic and Slavic
        "rus": "rus", "russian": "rus", "рос": "rus",
        "ukr": "ukr", "ukrainian": "ukr",
        # "bul": "bul", "bulgarian": "bul",
        # "srp": "srp", "serbian": "srp",
        # "srp_latn": "srp_latn",
        "bel": "bel", "belarusian": "bel",
        # "kaz": "kaz", "kazakh": "kaz",
        # "uzb": "uzb", "uzbek": "uzb",
        # "uzb_cyrl": "uzb_cyrl",
        # "kir": "kir", "kyrgyz": "kir",
        # "tgk": "tgk", "tajik": "tgk",
        # "tat": "tat", "tatar": "tat",
        # "mkd": "mkd", "macedonian": "mkd",

        # Western languages
        "eng": "eng", "english": "eng",
        # "deu": "deu", "ger": "deu", "german": "deu",
        # "fra": "fra", "fre": "fra", "french": "fra",
        # "ita": "ita", "italian": "ita",
        # "spa": "spa", "spanish": "spa",
        # "por": "por", "portuguese": "por",
        "pol": "pol", "polish": "pol", "polska": "pol",
        # "nld": "nld", "dutch": "nld",
        # "swe": "swe", "swedish": "swe",
        # "dan": "dan", "danish": "dan",
        "nor": "nor", "norwegian": "nor",
        # "fin": "fin", "finnish": "fin",

        # # Asian languages
        # "chi_sim": "chi_sim", "zh_cn": "chi_sim", "simplified": "chi_sim",
        # "chi_tra": "chi_tra", "zh_tw": "chi_tra", "traditional": "chi_tra",
        # "jpn": "jpn", "japanese": "jpn",
        # "kor": "kor", "korean": "kor",
        # "hin": "hin", "hindi": "hin",
        # "tam": "tam", "tamil": "tam",
        # "tel": "tel", "telugu": "tel",
        # "kan": "kan", "kannada": "kan",
        # "mal": "mal", "malayalam": "mal",
        # "mya": "mya", "burmese": "mya",
        # "tha": "tha", "thai": "tha",
        # "vie": "vie", "vietnamese": "vie",

        # Others
        # "ara": "ara", "arabic": "ara",
        # "heb": "heb", "hebrew": "heb",
        # "grc": "grc", "greek": "ell",
        # "ell": "ell", "modern_greek": "ell",
        # "amh": "amh", "ethiopic": "amh",
        # "ben": "ben", "bengali": "ben",
        # "guj": "guj", "gujarati": "guj",
        # "pan": "pan", "punjabi": "pan",
        # "urd": "urd", "urdu": "urd",
        # "syr": "syr", "syriac": "syr",
        # "san": "san", "sanskrit": "san",
        # "nep": "nep", "nepali": "nep",
    }

    for key, lang in lang_keywords.items():
        if key in name:
            return lang
    return "eng"  # default fallback


def ocr_image_file(file_path, lang="eng"):
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as e:
        print(f"[ERROR] OCR failed on image {file_path}: {e}")
        return ""


def ocr_pdf_file(file_path, lang="eng"):
    try:
        doc = fitz.open(file_path)
        return "\n\n".join(
            pytesseract.image_to_string(
                Image.open(io.BytesIO(page.get_pixmap(alpha=False).tobytes())),
                lang=lang
            ) for page in doc
        )
    except Exception as e:
        print(f"[ERROR] OCR failed on PDF {file_path}: {e}")
        return ""


def ocr_djvu_file(file_path, lang="eng"):
    png_path = file_path.with_suffix(".djvu.png")
    try:
        subprocess.run(["ddjvu", "-format=png", str(file_path), str(png_path)], check=True)
        return ocr_image_file(png_path, lang)
    except Exception as e:
        print(f"[ERROR] OCR failed on DjVu file {file_path}: {e}")
        return ""
    finally:
        if png_path.exists():
            png_path.unlink()


def ocr_file(file_path, lang=None):
    ext = file_path.suffix.lower()
    lang = lang or detect_language_from_filename(file_path)

    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return ocr_image_file(file_path, lang)
    elif ext == ".pdf":
        return ocr_pdf_file(file_path, lang)
    elif ext == ".djvu":
        return ocr_djvu_file(file_path, lang)
    else:
        print(f"[WARN] Unsupported file type for OCR: {file_path}")
        return ""
# ========================================================================
# ========================================================================
# ========================================================================     
def get_already_ocrd_stems(get_ocr_log):
    if not get_ocr_log.exists():
        print(f"[INFO] File not found: {get_ocr_log}")
        return set()
    return set(line.strip() for line in get_ocr_log.read_text(encoding="utf-8").splitlines())

def get_ocr_candidates_pending(get_ocr_candidates):
    if not get_ocr_candidates.exists():
        print(f"[INFO] File not found: {get_ocr_candidates}")
        return set()
    return set(line.strip() for line in get_ocr_candidates.read_text(encoding="utf-8").splitlines())

def strip_timestamp_and_txt(txt_stem: str) -> str:
# Remove timestamp and optional "_vX" or "_final", keep original filename with extension
    return re.sub(r"(_\d{8}_\d{6}|_v\d+|_final)?$", "", txt_stem)
# ========================================================================
# ========================================================================
# ======================================================================== 
def append_missing_candidates(dst_dir: Path, src_dir: Path, pending_path: Path):
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    dst_files = sorted(dst_dir.rglob("*.txt"))
    if not dst_files:
        print(f"[INFO] No .txt files in {dst_dir}.")
        return
    
    new_lines = []
    for txt_path in dst_files:
        # Consider empty .txt files only
        if txt_path.stat().st_size == 0 or txt_path.read_text(encoding="utf-8").strip() == "":
            full_stem = txt_path.stem  # e.g. "book.pdf_20250524_185119"
            base_filename = strip_timestamp_and_txt(full_stem)  # e.g. "book.pdf"
            
            # Find the matching source file in src_dir (allowing any extension)
            matches = list(src_dir.rglob(base_filename))
            if not matches:
                continue
            src_path = matches[0]

            line = f"{base_filename} | SRC: {src_path} | TXT: {txt_path} | SRC_EXISTS: {src_path.exists()}"
            new_lines.append(line)

            print(f"Found {len(new_lines)} candidates from empty txt files")

    if new_lines:
        with pending_path.open("w", encoding="utf-8") as f:  # overwrite!
            for line in new_lines:
                f.write(line + "\n")
        print(f"[INFO] Updated {len(new_lines)} OCR candidates in {pending_path}")
        print("Searchnig...")
    else:
        print(f"[INFO] No empty .txt files to add.")
# ========================================================================
# ========================================================================
# ========================================================================
def find_ocr_candidates() -> list[tuple[Path, Path, str]]:
    if not DST_DIR.exists() or not SRC_DIR.exists():
        print(f"[ERROR] DST_DIR or SRC_DIR missing: {DST_DIR} / {SRC_DIR}")
        return []

    OCR_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    OCRD_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Update pending list
    append_missing_candidates(DST_DIR, SRC_DIR, OCR_CANDIDATES)

    already_ocrd = get_already_ocrd_stems(OCRD_LOG)

    # Step 2: Find current empty .txt files
    empty_txt_files = {
        strip_timestamp_and_txt(f.stem): f
        for f in DST_DIR.rglob("*.txt")
        if f.stat().st_size == 0 or f.read_text(encoding="utf-8").strip() == ""
    }

    def matches(base_stem):
        return list(SRC_DIR.rglob(base_stem + ".*"))

    candidates = []

    for base_stem, txt_file in empty_txt_files.items():
        src_matches = matches(base_stem)
        if not src_matches:
            continue
        if base_stem not in already_ocrd:
            candidates.append((txt_file, src_matches[0], base_stem))

    # Step 3: Fallback if no fresh candidates
    if not candidates:
        pending_from_file = get_ocr_candidates_pending(OCR_CANDIDATES)
        if pending_from_file:
            print(f"[INFO] Using {len(pending_from_file)} existing OCR candidates from pending file.")
            for line in pending_from_file:
                parts = line.split("|")
                base = parts[0].strip()
                src_path = txt_path = None
                for p in parts:
                    if "SRC:" in p:
                        src_path = Path(p.split("SRC:")[1].strip())
                    elif "TXT:" in p:
                        txt_path = Path(p.split("TXT:")[1].strip())
                if src_path and txt_path:
                    candidates.append((txt_path, src_path, base))
        else:
            print("[INFO] No OCR candidates found.")
            return []

    # Step 4: Confirm with user
    print(f"[INFO] Ready to OCR {len(candidates)} files.")
    confirm = input("Proceed with OCR? [Y/N]: ").strip().lower()
    if confirm != "y":
        print("[INFO] OCR aborted by user.")
        return []

    return candidates
# ========================================================================
# ========================================================================
# ========================================================================
def perform_ocr_workflow(ocr_candidates: list[tuple[Path, Path, str]]):
    if not ocr_candidates:
        print("[INFO] No OCR work to perform.")
        return

    replaced, skipped = 0, 0
    for txt_file, src_file, base_stem in ocr_candidates:
        print(f"[OCR] Processing {src_file}")
        text = ocr_file(src_file)
        if text.strip():
            txt_file.write_text(text, encoding="utf-8")
            with OCRD_LOG.open("a", encoding="utf-8") as log_f:
                log_f.write(f"{base_stem}\n")
            print(f"[OCR] OCR successful → {txt_file}")
            replaced += 1
        else:
            print(f"[WARN] No text extracted from {src_file}")
            skipped += 1

    print(f"[OCR] Done. Replaced: {replaced}, Skipped: {skipped}")

if __name__ == "__main__":
    try:
        candidates = find_ocr_candidates()
        perform_ocr_workflow(candidates)
    except Exception as e:
        print(f"[FATAL] Exception: {e}")
'''
If ocr_candidates_pending.txt exists, load it.
    Scan DST_DIR for any empty .txt files.
    For each empty .txt, check if it's already listed in ocr_candidates_pending.txt. If not:
        Build corresponding SRC path and TXT path and append it to the file.
Then proceed as usual with filtering and OCR.
    Append file names to OCRD_LOG.
'''