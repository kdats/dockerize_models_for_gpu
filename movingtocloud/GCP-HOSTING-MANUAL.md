# How to Host the AI API Server on GCP (Step-by-Step Manual)

This manual shows you how to run the `llama-server` container on a GCP Cloud VM and open it up to the internet, so you can send queries to it from your local laptop (or a web app). This proves you can host an AI microservice in the cloud.

## Step 1: Create the VM & Open the Firewall

Open your local terminal and run these commands to create a T4 GPU VM and open port 8080 to the internet.

```bash
# 1. Create the VM (we add the 'ai-server' tag to identify it for the firewall)
gcloud compute instances create ai-hosting-vm \
    --machine-type=n1-standard-4 \
    --zone=us-central1-a \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --tags=ai-server

# 2. Open Port 8080 on GCP Firewall so your laptop can reach it
gcloud compute firewall-rules create allow-ai-port \
    --allow tcp:8080 \
    --target-tags=ai-server \
    --description="Allow incoming traffic to llama-server API"
```

## Step 2: SSH into the VM & Install Prerequisites

Connect to the VM:
```bash
gcloud compute ssh ai-hosting-vm --zone=us-central1-a
```

Once inside the VM, run this block to install the NVIDIA drivers and Docker:
```bash
# Install Drivers
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot  # The SSH session will disconnect. Reconnect after 30 seconds.
```

Reconnect (`gcloud compute ssh ai-hosting-vm`), then install Docker:
```bash
# Install Docker & NVIDIA Toolkit
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 3: Start the Hosted AI Server

Still inside the GCP VM, run the following command. It will download the model directly on the cloud VM and spin up the server container as a background daemon (`-d`).

```bash
# Download the model to a local folder on the VM
mkdir -p models
curl -L "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true" -o models/SmolLM2-360M-Instruct-Q4_K_M.gguf

# Run the container (bind to port 8080)
sudo docker run -d \
    --name my-hosted-ai \
    --gpus all \
    -p 8080:8080 \
    -v $(pwd)/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 999
```

Your API is now live! Type `exit` to leave the GCP VM and go back to your local laptop terminal.

## Step 4: Test the Hosted API from your Laptop!

First, get the public IP address of your GCP VM:
```bash
gcloud compute instances list --filter="name=ai-hosting-vm"
# Look for the EXTERNAL_IP column. (e.g., 34.123.45.67)
```

Now, from your laptop (or anywhere in the world), run this `curl` command using that Public IP:

```bash
# Replace YOUR_VM_PUBLIC_IP with the actual IP
curl -s http://YOUR_VM_PUBLIC_IP:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
          "prompt": "<|im_start|>user\nWhat is Cloud Computing?<|im_end|>\n<|im_start|>assistant\n",
          "max_tokens": 100,
          "stream": true
        }'
```

Because we added `"stream": true`, you will see the cloud server stream the AI response back to your laptop in real-time chunks over the internet!

## Step 5: Cleanup

When the demo is over, delete the VM to avoid billing:
```bash
gcloud compute instances delete ai-hosting-vm --zone=us-central1-a --quiet
gcloud compute firewall-rules delete allow-ai-port --quiet
```
