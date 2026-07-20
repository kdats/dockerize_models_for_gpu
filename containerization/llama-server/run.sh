#!/bin/bash
# FALLBACK DEMO — Use this if the primary Python GPU demo has issues
# Requirements: Docker + NVIDIA Container Toolkit on the host

MODEL_DIR="$(pwd)/../models"
MODEL_FILE="$MODEL_DIR/SmolLM2-360M-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true"

# Step 1: Download model if not already present
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading SmolLM2 360M model (~220MB)..."
    curl -L "$MODEL_URL" -o "$MODEL_FILE"
fi

# Step 2: Start the llama.cpp inference server (GPU)
echo "🚀 Starting llama.cpp server with GPU..."
docker run -d \
    --name llama-server \
    --gpus all \
    -p 8080:8080 \
    -v "$MODEL_DIR":/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 999

# Step 3: Wait for server to be ready
echo "Waiting for server to start (10 seconds)..."
sleep 10

# Step 4: Send a prompt and show the response
echo ""
echo "=== 🤖 Querying the model via API ==="
curl -s http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
          "prompt": "<|im_start|>user\nExplain why GPU computing matters in HPC in one short sentence.<|im_end|>\n<|im_start|>assistant\n",
          "max_tokens": 100,
          "stop": ["<|im_end|>"]
        }' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"🤖 AI: {r['choices'][0]['text'].strip()}\")"

# Cleanup
echo ""
echo "=== ✅ Done. Stopping server... ==="
docker stop llama-server && docker rm llama-server
