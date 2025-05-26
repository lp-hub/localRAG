# localRAGOCR

## Free, local, open-source RAG with SQLite & FAISS

- Created & tested with Python 3.12, llama-cpp-python, LangChain, 
- FAISS, and Gradio. Works offline on Ubuntu with NVIDIA GPU (for
- CUDA acceleration) and GGUF model. Includes OCR scripts.

#### Set up:

- 1. Download or clone this repository.
======================================================================

```
git clone https://github.com/lp-hub/localRAGOCR.git && cd localRAGOCR
```

- 2. Install GCC / build tools
======================================================================
```
sudo apt update

sudo apt install python3 python3.12-venv build-essential cmake sqlite3

sudo apt install calibre djvulibre-bin libchm-bin pandoc tesseract-ocr-all
```

- 3. Create and activate virtual environment
======================================================================
```
cd /../localRAGOCR && python3.12 -m venv venv # to create venv dir

source venv/bin/activate # (venv) USER@PC:/../localRAGOCR$

deactivate # after usig RAG
```

- 4. Install Python dependencies
======================================================================
```
pip install --upgrade pip && pip3 install striprtf

pip install faiss-cpu ftfy gradio langchain langchain-community langchain-huggingface pathspec pillow pymupdf pypandoc pypdf pyrtf-ng pyspellchecker pytesseract python-docx python-dotenv rapidfuzz sentence-transformers sqlite-utils symspellpy tiktoken unstructured
```

- 5. Install llama-cpp-python with CUDA support
======================================================================
```
pip uninstall -y llama-cpp-python

CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir --force-reinstall llama-cpp-python
```

- 6. Download the GGUF model
======================================================================
```
mkdir -p models && wget https://huggingface.co/mradermacher/LLama-3-8b-Uncensored-GGUF/resolve/main/LLama-3-8b-Uncensored.Q8_0.gguf -O models/Llama-3-8B-Uncensored.Q8_0.gguf
```

- 7. Add your documents
======================================================================
```
Place .pdf, .txt, .md, .epub, etc., into your files/ folder.
Supported file types are automatically handled by the loader.
```

- 8. Create and onfigure .env
======================================================================
```
DATA_DIR=/files/ DB_DIR=/faiss_db/ MODEL_PATH=/AI_model.gguf
```

#### Usage
```
1. Run the CLI interface

python3 src/main.py --rebuild-db # use --rebuild-db first time or to make new db

First run will embed and index documents.
You'll get an interactive prompt (You:) for local Q&A with sources.
Type in your question and wait for the model response.

2. (Optional) Start the Gradio Web UI

python webui.py

You will see something like:
Web UI running at http://192.168.X.X:7860
Open the IP in your browser for a simple web-based interface.
```
#### Notes

Your computer may not be powerful enough to run some models.

localRAGOCR
├── db
│   └── normalization_map.template.json
├── faiss_db
├── help
│   ├── docstore.txt
│   ├── learning.txt
│   ├── LLama-3-8b-Uncensored.txt
│   ├── models.txt
│   └── SQLite.txt
├── logs
├── scripts
├── src
│   ├── data
│   │   ├── ui
│   │   │   ├── admin.py
│   │   │   ├── filtering_cli.py
│   │   │   └── ui.py
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── filter.py
│   │   └── jsonhandler.py
│   ├── extract
│   │   ├── extractor.py
│   │   ├── ocr.py
│   │   ├── ocr2map.py
│   │   └── ocrerrors.py
│   ├── ingest
│   │   ├── chunker.py
│   │   └── formatter.py
│   ├── know
│   │   ├── provenance.py
│   │   ├── retriever.py
│   │   └── store.py
│   ├── config.template.py
│   ├── llm.py
│   ├── logger.py
│   ├── main.py
│   └── webui.py
├── venv
├── .env_template.txt
├── .gitignore
├── README.md
└── requirements.txt
