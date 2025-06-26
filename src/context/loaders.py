import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import QName
from bs4 import BeautifulSoup
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader, TextLoader)
from langchain.schema import Document
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
from unstructured.partition.doc import partition_doc
from unstructured.partition.html import partition_html

# ========== .txt loader ==========
class SafeTextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding=None, autodetect_encoding=True)
        # super().__init__(file_path, encoding='utf-8', autodetect_encoding=False)

# ========== .doc loader (fallback using unstructured) ==========
class UnstructuredDocLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        elements = partition_doc(filename=self.file_path)
        return [Document(page_content=str(el)) for el in elements]

# ========== .rtf loader using striprtf ==========
class RTFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> list[Document]:
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = rtf_to_text(f.read())
        return [Document(page_content=content)]

# ========== .djvu loader using djvu.decode (basic) ==========
class DidjvuLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        if not shutil.which("djvutxt"):
            raise EnvironmentError("djvutxt is not installed. sudo apt install djvulibre-bin")

        djvu_path = Path(self.file_path)

        if not djvu_path.exists():
            raise FileNotFoundError(f"DjVu file not found: {self.file_path}")

        try:
            # Extract text using djvutxt
            result = subprocess.run(
                ["djvutxt", self.file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            full_text = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"djvutxt failed: {e.stderr}")

        return [Document(page_content=full_text)]

# ========== .chm loader using extract_chmlib ==========
class CHMLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self) -> list[Document]:
        extract_dir = Path("/tmp/chm_extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Extracting CHM file {self.file_path} to {extract_dir}")

        subprocess.run(["extract_chmLib", self.file_path, str(extract_dir)])
        text = ""
        for html_file in extract_dir.rglob("*.htm*"):
            with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                text += f.read() + "\n"
        print(f"[DEBUG] Finished extracting CHM, total length {len(text)} chars")
        return [Document(page_content=text)]

# Some .chm files can't be parsed well because they're binary-encoded archives.
#     Extract .chm manually:
# archmage mybook.chm output_dir/
# or:
#     7z x mybook.chm -ooutput_dir/
#     Then recursively process .html files from the extracted content.
    
# ========== .htm .html loader using unstructured ==========
class UnstructuredHTMLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            elements = partition_html(text=f.read())
        return [Document(page_content=el.text) for el in elements if el.text]

# ========== .mobi loader using ebooklib and bs4 ==========
# Class to fix Path vs str problem in UnstructuredEPubLoader
class FixedEPubLoader(UnstructuredEPubLoader):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(str(file_path), *args, **kwargs)     
# MOBI is not directly supported. Convert using Calibre CLI to EPUB before ingestion.
# ebook-convert input.mobi output.epub
class MOBILoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        if not shutil.which("ebook-convert"):
            raise EnvironmentError("'ebook-convert' not found. Please install Calibre CLI.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            epub_path = tmpdir_path / (self.file_path.stem + ".epub")

            try:
                subprocess.run(
                    ["ebook-convert", str(self.file_path), str(epub_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to convert MOBI to EPUB: {e}")

            if not epub_path.exists():
                raise FileNotFoundError(f"Conversion failed, EPUB not found at {epub_path}")
            return FixedEPubLoader(epub_path).load()

# ========== .pdf loader ==========
class PyPDFLoaderWithPassword(PyPDFLoader):
    def __init__(self, file_path, password=None):
        super().__init__(file_path)
        self.password = password

    def load(self) -> list[Document]:
        reader = PdfReader(self.file_path, password=self.password)
        texts = [page.extract_text() or "" for page in reader.pages]
        return [Document(page_content="\n".join(texts))]

# ========== .xml Blogspot loader ==========
class BlogspotXMLLoader:
    def __init__(self, file_path, tags_filter: list[str] = None):
        self.file_path = file_path
        self.tags_filter = tags_filter  # optional tag filtering

    @staticmethod
    def is_blogspot_export(file_path: str) -> bool:
        # Strip namespace and return local tag name.
        def localname(tag: str) -> str:    
            if tag.startswith("{"):
                return tag.split("}", 1)[1]
            return tag

        try:
            for event, elem in ET.iterparse(file_path, events=("start",)):
                if localname(elem.tag) == "feed":
                    # Blogger-specific hints
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore").lower()
                    if "schemas.google.com/blogger" in content or "www.blogger.com" in content:
                        return True
                    return False
                break
        except ET.ParseError:
            return False
        return False

    def load(self) -> list[Document]:
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        ns = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        documents = []
        for entry in root.findall("atom:entry", ns):
            kind = entry.find("atom:category", ns)
            categories = entry.findall("atom:category", ns)
            tags = [cat.attrib.get("term", "") for cat in categories if cat.attrib.get("term")]
            # --- Normalize tags ---
            normalized_tags = {tag.strip().lower() for tag in tags}
            normalized_filter = {tag.strip().lower() for tag in self.tags_filter}

            # --- Apply tag filter if defined ---
            if self.tags_filter:
                if not normalized_tags & normalized_filter:
                    continue
            # --- Ensure this is a real blog post ---
            if not any(tag.endswith("#post") for tag in tags):
                print(f"[MATCH] Tags: {tags} → matched: {normalized_tags & normalized_filter}")
                continue

            # if kind is not None and kind.attrib.get("term", "").endswith("#post"):
            title_el = entry.find("atom:title", ns)
            content_el = entry.find("atom:content", ns)
            pub_date_el = entry.find("atom:published", ns)
            # --- Advantage of HTML in getting images and other data ---
            title = title_el.text if title_el is not None else ""
            content = content_el.text if content_el is not None else ""
            pub_date = pub_date_el.text if pub_date_el is not None else ""
            full_text = f"{title}\n{pub_date}\n\n{content}".strip()
            # --- BeautifulSoup removes links to images and videos ---
            # raw_html = content_el.text if content_el is not None else "" # BeautifulSoup
            # clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n\n")
            # full_text = f"{title}\n{pub_date}\n\n{clean_text}".strip() # BeautifulSoup
            if full_text:
                documents.append(Document(page_content=full_text))
        return documents
    
# ========== .xml WordPress loader ==========
class WordPressXMLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def is_wordpress_export(file_path: str) -> bool:
        try:
            for event, elem in ET.iterparse(file_path, events=("start",)):
                if elem.tag == "rss" and "wordpress.org/export" in str(elem.attrib.get("xmlns:wp", "")):
                    return True
                if elem.tag.endswith("rss"):
                    # Check for WordPress namespace
                    for k, v in elem.attrib.items():
                        if "wordpress.org/export" in v:
                            return True
                break  # We only need the root
        except ET.ParseError:
            return False
        return False

    def load(self) -> list[Document]:
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        ns = {
            "wp": "http://wordpress.org/export/1.2/",
            "content": "http://purl.org/rss/1.0/modules/content/",
            "dc": "http://purl.org/dc/elements/1.1/"
        }

        documents = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            content_el = item.find("content:encoded", ns)
            pub_date_el = item.find("pubDate")

            title = title_el.text if title_el is not None else ""
            content = content_el.text if content_el is not None else ""
            pub_date = pub_date_el.text if pub_date_el is not None else ""

            full_text = f"{title}\n{pub_date}\n\n{content}".strip()
            if full_text:
                documents.append(Document(page_content=full_text))

        return documents

# ========== Loader Dispatcher ==========
def detect_and_load_text(file_path: str, pdf_password: str = None) -> list[Document] | None:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoaderWithPassword(file_path, password=pdf_password)

    elif ext == ".xml":
        if WordPressXMLLoader.is_wordpress_export(file_path):
            loader = WordPressXMLLoader(file_path)
        elif BlogspotXMLLoader.is_blogspot_export(file_path):
            raw_tags = os.getenv("TAGS", "")
            tags_filter = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
            loader = BlogspotXMLLoader(file_path, tags_filter=tags_filter)
        else:
            print(f"[INFO] .xml file not recognized as WordPress or Blogspot export: {file_path}")
            return []

    else:
        loader_map = {
        # ".pdf": PyPDFLoaderWithPassword, # PyPDFLoader replaced to fix pypdf/_encryption.py
        ".md": UnstructuredMarkdownLoader,
        ".epub": FixedEPubLoader,  # UnstructuredEPubLoader replaced to globally fix .epub loading
        ".mobi": MOBILoader,  # custom MOBI loader using Calibre conversion
        ".chm": CHMLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredDocLoader,
        ".rtf": RTFLoader,
        ".txt": SafeTextLoader,
        ".djvu": DidjvuLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        }
    
        loader_cls = loader_map.get(ext)
        if loader_cls is None:
            return None
        loader = loader_cls(file_path)

    try:
        return loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []