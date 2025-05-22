# localRAG

## Free, local, open-source RAG with SQLite & FAISS

Created & tested with Python 3.12, llama-cpp-python, LangChain, FAISS, and Gradio. Works offline on Ubuntu with NVIDIA GPU (for CUDA acceleration) and GGUF model

#### Set up:

1. Download or clone this repository.

git clone https://github.com/lp-hub/localRAG.git && cd localRAG

2. Install GCC / build tools

sudo apt update

sudo apt install python3 python3.12-venv build-essential cmake

https://visualstudio.microsoft.com/visual-cpp-build-tools/ for Windows, whatever...

3. Create and activate virtual environment

cd /../localRAG && python3.12 -m venv venv # to create venv dir

source venv/bin/activate # (venv) USER@PC:/../localRAG$

deactivate # after usig RAG

4. Install Python dependencies

pip install --upgrade pip

pip install faiss-cpu gradio langchain langchain-community langchain-huggingface pathspec pymupdf pypdf sentence-transformers sqlite-utils tiktoken

5. Install llama-cpp-python with CUDA support

pip uninstall -y llama-cpp-python

CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir --force-reinstall llama-cpp-python

6. Download the GGUF model

mkdir -p models && wget https://huggingface.co/mradermacher/LLama-3-8b-Uncensored-GGUF/resolve/main/LLama-3-8b-Uncensored.Q8_0.gguf -O models/Llama-3-8B-Uncensored.Q8_0.gguf

7. Add your documents

Place .pdf, .txt, .md, .epub, etc., into your data/ folder.
Supported file types are automatically handled by the loader.

8. Rename config.template.py to config.py and configure

DATA_DIR = '/directory_with_files'
DB_DIR = "/faiss_db/"
MODEL_PATH = "/AI_model.gguf"

#### Usage

1. Run the CLI interface

python3 src/main.py --rebuild-db # use --rebuild-db first time or to make new db from new files

First run will embed and index documents.
You'll get an interactive prompt (You:) for local Q&A with sources.
Type in your question and wait for the model response.

2. (Optional) Start the Gradio Web UI

python webui.py

You will see something like:
Web UI running at http://192.168.X.X:7860
Open the IP in your browser for a simple web-based interface.

#### Notes

Your computer may not be powerful enough to run some models.

localRAG
├── db
├── faiss_db
├── help
│   ├── learning.txt
│   ├── LLama-3-8b-Uncensored.txt
│   └── SQLite.txt
├── logs
├── scripts
├── src
│   ├── __pycache__
│   ├── db
│   │   ├── __pycache__
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── db.py
│   │   ├── filtering_cli.py
│   │   ├── provenance.py
│   │   └── ui.py
│   ├── chunker.py
│   ├── config.template.py
│   ├── formatter.py
│   ├── llm.py
│   ├── logger.py
│   ├── main.py
│   ├── retriever.py
│   ├── store.py
│   └── webui.py
├── venv
├── .gitignore
├── README.md
└── requirements.txt
