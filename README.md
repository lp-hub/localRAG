# localRAG

## Free, local, open-source RAG with SQLite & FAISS

- Created & tested with Python 3.12, llama.cpp, LangChain, 
- FAISS, and Gradio. Works offline on Ubuntu with NVIDIA GPU (for
- CUDA acceleration) and GGUF model. OCR scripts available.

#### Set up:

- 1. Download or clone this repository.
======================================================================

```
git clone https://github.com/lp-hub/localRAG.git && cd localRAG
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
cd /../localRAG && python3.12 -m venv venv # to create venv dir

source venv/bin/activate # (venv) USER@PC:/../localRAG$

deactivate # after usig RAG
```

- 4. Install Python dependencies
======================================================================
```
pip install --upgrade pip && pip3 install striprtf

pip install faiss-cpu ftfy gradio langchain langchain-community langchain-huggingface pathspec pillow pymupdf pypandoc pypdf pyrtf-ng pyspellchecker pytesseract python-docx python-dotenv rapidfuzz sentence-transformers sqlite-utils symspellpy tiktoken unstructured
```

- 5. Build and install llama-server with CUDA support
======================================================================
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_BUILD_SERVER=ON -DGGML_CUDA=ON
make -j$(nproc)
./build/bin/llama-server \ -m ./models/llama-3.gguf \--port 8080 \ --ctx-size 4096 \ --n-gpu-layers 35 \ --host 127.0.0.1 \ --mlock \ --no-mmap

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
DATA_DIR=/files/ DB_DIR=/db/ MODEL_PATH=/AI_model.gguf
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

localRAG
├── db
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
├── .gitignore
├── README.md
├── requirements.txt
├── template.env.txt
└── templatenormalization_map.json
