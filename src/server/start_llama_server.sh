#!/bin/bash

# === CONFIGURATION ===
LLAMA_BIN="/llama.cpp/build/bin/llama-server"
MODEL_PATH="/models/LLama-3-8b-Uncensored.Q8_0.gguf"
PORT=8080
CTX_SIZE=4096
GPU_LAYERS=36
HOST="127.0.0.1"
EXTRA_FLAGS="--mlock --no-mmap"

# === VALIDATION ===
if [ ! -f "$LLAMA_BIN" ]; then
  echo "[Error] llama-server binary not found at: $LLAMA_BIN"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[Error] Model file not found at: $MODEL_PATH"
  exit 1
fi

# === LAUNCH ===
echo "[Info] Starting llama-server..."
echo "[Info] Model: $MODEL_PATH"
echo "[Info] Port: $PORT, GPU Layers: $GPU_LAYERS, Context: $CTX_SIZE"

"$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" \
  --host "$HOST" \
  $EXTRA_FLAGS

#chmod +x start_llama_server.sh
#PORT=8081 ./start_llama_server.sh
#./start_llama_server.sh