#!/bin/bash
# ==============================================================================
# ONE-CLICK GCP SETUP SCRIPT for GPU AI DEMO
# ==============================================================================
# IMPORTANT PREREQUISITE:
# You MUST have created the VM with a 50GB Boot Disk, OR increased the disk
# size to 50GB in the GCP console before running this script.
# AI models require at least 20-30GB of free space to build and run.
# ==============================================================================

echo "🚀 [1/4] Expanding partitions to use all available disk space..."
sudo growpart /dev/sda 1 2>/dev/null || true
sudo resize2fs /dev/sda1 2>/dev/null || true
sudo growpart /dev/nvme0n1 1 2>/dev/null || true
sudo resize2fs /dev/nvme0n1p1 2>/dev/null || true

echo "🚀 [2/4] Updating package lists and installing GPU Drivers..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq nvidia-driver-535

echo "🚀 [3/4] Installing Docker and NVIDIA Container Toolkit..."
sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "🚀 [4/4] Creating App Files and Building the AI Container..."
mkdir -p ~/demo && cd ~/demo

# Create Dockerfile
cat << 'EOF' > Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip3 install llama-cpp-python huggingface_hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir
RUN python3 -c "import huggingface_hub; huggingface_hub.hf_hub_download(repo_id='bartowski/SmolLM2-360M-Instruct-GGUF', filename='SmolLM2-360M-Instruct-Q4_K_M.gguf', local_dir='/models')"
COPY app.py .
CMD ["python3", "app.py"]
EOF

# Create app.py
cat << 'EOF' > app.py
from llama_cpp import Llama
print("\n🚀 INITIALIZING AI MODEL...")
llm = Llama(model_path="/models/SmolLM2-360M-Instruct-Q4_K_M.gguf", n_gpu_layers=-1, verbose=False)
print("✅ Model loaded successfully into GPU memory.\n")
prompt = "Explain why GPUs are better than CPUs for AI in one short sentence."
print(f"👤 USER: {prompt}\n🤖 AI: ", end="", flush=True)
output = llm(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", max_tokens=100, stop=["<|im_end|>"], echo=False, stream=True)
for chunk in output: print(chunk["choices"][0]["text"], end="", flush=True)
print("\n\n✅ Container execution completed perfectly.")
EOF

# Build and Run
echo "🔨 Building the Docker Image (This will take ~2 minutes)..."
sudo docker build -t gpu-demo:v1 .

echo "✅ Build Complete! Running the AI..."
sudo docker run --rm --gpus all gpu-demo:v1
