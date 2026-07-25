# CUDA to Cloud & HPC: Master Reference & Teaching Handbook

> **Programme Title:** CUDA to Cloud — Accelerating AI through Supercomputing and Advanced Computing  
> **Target Audience:** University Faculty, Department Heads, Researchers, and Computer Science Students  
> **Document Type:** Master Reference Manual & Step-by-Step Live Session Handbook  
> **Scope:** Local Containerization (Docker), GPU Acceleration (NVIDIA Container Toolkit), Cloud Infrastructure Migration (Google Cloud Platform via Web Console), Multi-Container Orchestration (Docker Compose), and Supercomputing HPC Execution (Apptainer/Singularity).

---

# Table of Contents
1. [Executive Summary & Core Narrative](#1-executive-summary--core-narrative)
2. [Workload Portability & Containerization Theory](#2-workload-portability--containerization-theory)
3. [Deep Dive into Docker Architecture & Recipes](#3-deep-dive-into-docker-architecture--recipes)
4. [Multi-Container Orchestration: Docker Compose](#4-multi-container-orchestration-docker-compose)
5. [GPU Hardware Acceleration & NVIDIA Container Toolkit](#5-gpu-hardware-acceleration--nvidia-container-toolkit)
6. [Local Workstation Build, Execution & Image Transfer](#6-local-workstation-build-execution--image-transfer)
7. [Cloud Infrastructure Concepts & GCP Selection](#7-cloud-infrastructure-concepts--gcp-selection)
8. [Step-by-Step Live Session Walkthrough (Pure Web UI Path)](#8-step-by-step-live-session-walkthrough-pure-web-ui-path)
9. [HPC Supercomputing & Apptainer/Singularity for Academia](#9-hpc-supercomputing--apptainersingularity-for-academia)
10. [Faculty Presentation Prompts & Educational Use Cases](#10-faculty-presentation-prompts--educational-use-cases)
11. [Emergency Cheatsheet & Diagnostic Troubleshooting](#11-emergency-cheatsheet--diagnostic-troubleshooting)
12. [Audience Q&A & Presenter Mastery](#12-audience-qa--presenter-mastery)

---

# 1. Executive Summary & Core Narrative

### 1.1 The Challenge: The "Works on My Machine" Paradigm in AI
In academic research and industrial software engineering, developing Artificial Intelligence (AI) and Machine Learning (ML) applications on a local developer workstation is only the first step. The moment an engineer attempts to deploy that workload to a team member's machine, a remote server, a cloud Virtual Machine (VM), or a university High-Performance Computing (HPC) cluster, they encounter severe environment divergence:
- Incompatible CUDA driver versions installed on the host.
- Python package dependency hell (differing `torch`, `transformers`, or `llama-cpp-python` wheels).
- Missing Linux system shared libraries (`libgomp`, `libcuda.so`).
- Differing hardware capabilities and PATH configurations.

### 1.2 The Solution: The Single Workload Migration Path
The core methodology taught in this programme is the **Single Workload Migration Path**. Rather than treating local development, cloud computing, and supercomputing as entirely separate disciplines with different tools, we construct **one single containerized GPU workload** and move it across three distinct tiers:

```text
+-----------------------------------------------------------------------------------+
|                            THE CENTRAL LEARNING PATH                              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ 1. Local Workstation ]  ==> Package Python + CUDA model inside Docker          |
|                                                                                   |
|  [ 2. Cloud GPU VM (GCP) ] ==> Deploy & Stream inference via GCP Web Console      |
|                                                                                   |
|  [ 3. HPC Supercomputer ]  ==> Convert to Apptainer (.sif) for rootless execution |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

By maintaining a single unified workload (language model text inference), learners focus on **portability engineering** rather than memorizing redundant cloud APIs or framework syntax.

---

# 2. Workload Portability & Containerization Theory

### 2.1 What Exactly is a "Workload"?
A workload is not just a Python script, nor is it just a model weights file. A workload is the **complete execution context** required to produce deterministic, correct computational results:

$$\text{Workload} = \text{Application Code} + \text{Runtime Dependencies} + \text{Hardware Bindings} + \text{Model Tensors} + \text{Environment Config}$$

1. **Application Code**: The Python script (`app.py`), API wrapper, or HTTP server logic.
2. **Runtime Dependencies**: Python binaries, PyTorch, C++ libraries, `llama.cpp` bindings.
3. **Hardware Bindings**: User-space CUDA libraries (`libcuda.so`, `libcudart.so`), NVIDIA Driver hooks.
4. **Model Tensors**: The quantized neural network weights (e.g., `SmolLM2-360M-Instruct-Q4_K_M.gguf`).
5. **Environment Config**: Port bindings, environment variables (`CUDA_VISIBLE_DEVICES`), thread counts.

### 2.2 Virtual Machines vs. Containers: Technical Deep-Dive

```text
+-----------------------------------+        +-----------------------------------+
|   App A (PyTorch) | App B (vLLM)  |        |   App A (PyTorch) | App B (vLLM)  |
+-------------------+---------------+        +-------------------+---------------+
|   Bins / Libs     |  Bins / Libs  |        |   User-Space Libs | User-Space Libs|
+-------------------+---------------+        +-------------------+---------------+
|   Guest OS Kernel | Guest OS Kernel|       |          Docker Daemon            |
+-------------------+---------------+        +-----------------------------------+
|            Hypervisor             |        |          Host OS Kernel           |
+-----------------------------------+        +-----------------------------------+
|        Physical Hardware          |        |        Physical Hardware          |
+-----------------------------------+        +-----------------------------------+
        VIRTUAL MACHINES (VMs)                           CONTAINERS
```

### The Docker Deamon

If Docker were a restaurant, the **Docker daemon** is the **head chef** working tirelessly in the kitchen. It is the invisible background program that actually does all the heavy lifting to make your containers run.

### The Restaurant Analogy

When you use Docker, there are usually two main parts working together:

* **The Docker Client (The Waiter):** This is the part you interact with. When you type a command into your computer like `"docker run"` to start a container, you are placing an order with the waiter.
* **The Docker Daemon (The Head Chef):** The waiter doesn't actually cook your food; they just take your order to the kitchen. The Docker daemon receives your command and gets to work. It fetches the ingredients (downloads the software), prepares the meal (builds the container), and manages the kitchen (makes sure the container keeps running securely).

Because the daemon is the brain controlling how containers are built and run, you have to tell *it* about your shiny new NVIDIA Toolkit so it knows to use it!


| Architectural Feature | Virtual Machines (VMs) | Containers (Docker / Apptainer) |
| :--- | :--- | :--- |
| **Abstraction Layer** | Hardware-level virtualization | Operating System (Kernel) user-space isolation |
| **Guest OS** | Full Linux/Windows OS kernel in each VM | **No Guest Kernel** (shares host OS kernel) |
| **Startup Overhead** | Minutes (boots BIOS, kernel, systemd) | **Milliseconds to Seconds** (spawns Linux process) |
| **Memory Footprint** | Gigabytes per VM (kernel memory reservation) | Megabytes (only application process memory) |
| **Disk Image Size** | 10 GB to 100 GB (complete OS disk) | 100 MB to 5 GB (only user-space files) |
| **GPU Hardware Access** | Requires PCIe Passthrough / Virtual GPU drivers | Direct device mounting via Container Runtime Hooks |
| **HPC Suitability** | Low (cannot share supercomputer hardware easily)| **Extremely High** (native Linux process execution) |

### 2.3 User Space vs. Kernel Space
Understanding why containers are lightweight requires understanding Linux process architecture:
- **Kernel Space**: Controls physical hardware (CPU scheduling, RAM management, disk I/O, NVIDIA PCIe drivers). The host Linux kernel is shared by all processes.
- **User Space**: Holds user binaries, shared C libraries (`glibc`), Python interpreters, and user code.

A container is simply a isolated set of user-space processes running directly on the host Linux kernel using two core Linux kernel features:
1. **Namespaces**: Provides isolation (PID namespace hides other processes; NET namespace isolates network ports; MOUNT namespace isolates directory trees).
2. **Control Groups (cgroups)**: Enforces resource boundaries (limits max CPU cores, RAM gigabytes, and GPU access).

---

# 3. Deep Dive into Docker Architecture & Recipes

https://docs.docker.com/desktop/setup/install/windows-install/ 

### 3.1 Anatomy of a Production-Grade Dockerfile
A `Dockerfile` is an executable specification for constructing a read-only container image.

```dockerfile
# ==============================================================================
# Step 1: Base Image (CUDA Runtime Baseline)
# ==============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ==============================================================================
# Step 2: System Dependencies & Environment Setup
# ==============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -yq --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Step 3: Set Application Directory
# ==============================================================================
WORKDIR /app

# ==============================================================================
# Step 4: Install Python Dependencies (Using Pre-compiled CUDA Wheels)
# ==============================================================================
RUN pip3 install --no-cache-dir \
    llama-cpp-python \
    huggingface_hub \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# ==============================================================================
# Step 5: Download Model Weights (Container Bake-In Pattern)
# ==============================================================================
RUN python3 -c "import huggingface_hub; huggingface_hub.hf_hub_download(repo_id='bartowski/SmolLM2-360M-Instruct-GGUF', filename='SmolLM2-360M-Instruct-Q4_K_M.gguf', local_dir='/models')"

# ==============================================================================
# Step 6: Application Code & Startup Command
# ==============================================================================
COPY app.py .

EXPOSE 8080

CMD ["python3", "app.py"]
```

### 3.2 Detailed Instruction Breakdown
- `FROM`: Specifies the foundational container image. Always use official, minimal base images (e.g., `nvidia/cuda:12.1.1-runtime-ubuntu22.04`).
- `ENV`: Sets environment variables inside the container environment during both build and runtime phases.
- `RUN`: Executes bash commands during the build phase to create a new read-only image layer. Always chain `apt-get update && apt-get install` and clear cache (`rm -rf /var/lib/apt/lists/*`) in a single line to keep layer sizes small.
- `WORKDIR`: Sets the active directory context for all subsequent `RUN`, `COPY`, `CMD`, and `ENTRYPOINT` instructions.
- `COPY`: Transfers files from the host machine's **Build Context** into the image.
- `EXPOSE`: Documents which network ports the application listens on (informational; does not publish the port by itself).
- `CMD`: Provides default arguments for executing the container. Can be overridden at runtime.

### 3.3 Image Layer Caching & Context Optimization
Docker builds images using a layered filesystem. Each instruction in a `Dockerfile` creates a new immutable layer.

```text
[ Layer 1: Base Ubuntu 22.04 + CUDA Runtime ] ------> Cached (Rarely changes)
[ Layer 2: System Packages (python3, pip)  ] ------> Cached (Rarely changes)
[ Layer 3: Pip Install llama-cpp-python     ] ------> Cached (Rarely changes)
[ Layer 4: Model Download (360M GGUF)       ] ------> Cached (Large file)
[ Layer 5: COPY app.py .                    ] ------> Rebuilt whenever app.py is modified!
```

#### Rule of Layer Ordering:
Always place stable, infrequently changed commands (installing packages, downloading large model weights) **near the top** of the `Dockerfile`, and frequently changing application source code (`COPY app.py .`) **at the very bottom**. This ensures that editing a line of Python code rebuilds only the final 5 KB layer in seconds rather than redownloading gigabytes of CUDA packages!

### 3.4 The Build Context (`.`) & `.dockerignore`
When you execute `docker build -t gpu-demo:v1 .`, the final dot (`.`) tells Docker to package the entire current working directory into a tarball and send it to the Docker daemon.

If your project directory contains large training logs, `.git` histories, temporary virtual environments, or old tarballs, sending them to the daemon will dramatically slow down builds. 

Create a `.dockerignore` file in the root directory:
```text
.git
.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
env/
models/*.bin
results/
*.tar
*.sif
```

---

# 4. Multi-Container Orchestration: Docker Compose

### 4.1 Why Docker Compose?
While a single container is sufficient for running an isolated AI script, real-world cloud applications consist of **multiple interacting microservices**. For example, an AI product typically requires:
1. **Frontend Container**: Web UI (React / Next.js) for user prompts.
2. **Backend AI Container**: High-performance inference engine (`llama-server`).
3. **Database Container**: Vector database (Qdrant / ChromaDB) or PostgreSQL for user chat history.

**Docker Compose** is a tool for defining and running multi-container Docker applications using a single YAML configuration file (`docker-compose.yml`).

### 4.2 Production Microservice Architecture

```text
                                 [ Browser Client ]
                                         │
                                         ▼ (Port 3000)
                        ┌─────────────────────────────────┐
                        │      Frontend Service (UI)      │
                        │    (Next.js / Node.js Container)│
                        └────────────────┬────────────────┘
                                         │
                                         ▼ (Internal Network: http://ai-engine:8080)
                        ┌─────────────────────────────────┐
                        │      Backend AI Service         │
                        │   (llama.cpp CUDA Container)    │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │       NVIDIA GPU Hardware       │
                        └─────────────────────────────────┘
```

### 4.3 `docker-compose.yml` Reference Implementation
Below is a complete reference file demonstrating how multi-container applications access GPU hardware declaratively:

```yaml
version: '3.8'

services:
  # ============================================================================
  # Service 1: GPU AI Inference Server
  # ============================================================================
  ai-engine:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    container_name: ai-backend-engine
    restart: always
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    command: >
      -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf
      --host 0.0.0.0
      --port 8080
      --n-gpu-layers 999
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # ============================================================================
  # Service 2: Web User Interface
  # ============================================================================
  web-ui:
    image: nginx:alpine
    container_name: ai-frontend-ui
    restart: always
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
    depends_on:
      - ai-engine
```

### 4.4 Essential Docker Compose Commands
- **Start all microservices in background**:
  ```bash
  docker compose up -d
  ```
- **Inspect running services and status**:
  ```bash
  docker compose ps
  ```
- **View aggregated live logs**:
  ```bash
  docker compose logs -f
  ```
- **Stop and dismantle all containers and networks**:
  ```bash
  docker compose down
  ```

---

# 5. GPU Hardware Acceleration & NVIDIA Container Toolkit

### 5.1 Why Standard Containers Cannot Access GPUs
Linux containers achieve isolation by masking hardware devices. Inside a standard container, `/dev/` contains only virtual devices like `/dev/null`, `/dev/random`, and `/dev/pts`.

Physical GPUs sit on the PCIe bus and require character device files (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and kernel module drivers (`nvidia.ko`). Without explicit device passthrough, user applications inside a container fail with errors such as:
```text
CUDA driver version is insufficient for CUDA runtime version
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

### 5.2 NVIDIA Container Toolkit Architecture
The **NVIDIA Container Toolkit** (`nvidia-container-toolkit`) modifies the container runtime lifecycle. It hooks into `containerd` or `dockerd` to automatically inject physical GPU devices and user-space libraries when a container starts.

```text
[ docker run --gpus all ]
          │
          ▼
[ Docker Engine / Containerd Runtime ]
          │
          ▼
[ nvidia-container-toolkit (OCI Prestart Hook) ]
          │
          ├── 1. Discovers host GPU hardware (/dev/nvidia0, /dev/nvidia-uvm)
          ├── 2. Mounts host NVIDIA user-space drivers (libcuda.so) into container /usr/lib
          └── 3. Passes control to container entrypoint
          │
          ▼
[ Running Container Process (PyTorch / CUDA App) ]
```
Technical documentation can definitely get bogged down in jargon! Here is a simple, straightforward breakdown of what the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.4/arch-overview.html) architecture is actually doing.

### The Big Picture

Normally, software "containers" (like Docker) are designed to be isolated little bubbles. They pack up an application so it can run anywhere, but because they are isolated, they don't naturally know how to talk to your computer's specialized hardware—specifically, your NVIDIA graphics card (GPU).

The **NVIDIA Container Toolkit** is essentially an adapter. It bridges the gap so that the software inside your isolated container can access your powerful NVIDIA GPU to do heavy lifting (like AI processing or complex math).

### The Analogy

Imagine a software container is a **standard rental car**, and your GPU is a **supercharged engine**.

* Normally, the rental agency just gives you a standard car, and you drive off.
* But if your project requires that supercharged engine, the standard rental agency doesn't know how to install it.
* The **NVIDIA Toolkit** acts as a specialized mechanic. Right before you drive the car off the lot, the mechanic steps in, wires up the supercharged engine to your rental car, and makes sure the car knows how to use it.

### The Components, Simplified

The page breaks down a few different pieces that make this happen. Here is what they actually do:

* **The Runtime:** Think of this as the manager. When you ask your computer to start a container, this manager intercepts the request and says, *"Hold on, this application needs a GPU. I need to call our specialized mechanic before we let it start."*
* **The Hook:** This is the mechanic. Right before the container actually boots up, this script jumps in to connect the specific GPU hardware to the container.
* **The Library and CLI (Command Line Interface):** These are the physical tools, manuals, and wrenches the mechanic uses to actually make the connection work with your computer's operating system.

### The Bottom Line for You
All you have to do is install the main package (`nvidia-container-toolkit`) and tell your container program (like Docker) to use it. Once you do that, the toolkit handles all the complicated mechanics behind the scenes for you.

### 5.3 Installing the NVIDIA Toolkit (On Host System)
Run these commands on any Ubuntu host equipped with NVIDIA hardware:

```bash
# 1. Add NVIDIA Repository GPG Key & Package Lists
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker Runtime to use NVIDIA Toolkit Hook
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker Daemon
sudo systemctl restart docker
```

### 5.4 Demystifying `--gpus all` vs Host Drivers
A critical distinction for developers and system administrators:

- **Host Operating System Must Have**:
  1. Physical NVIDIA GPU hardware plugged into PCIe slot.
  2. NVIDIA Linux Kernel Driver (`nvidia-driver-535` or higher).
  3. `nvidia-container-toolkit` package installed.

- **Container Does NOT Need**:
  1. Kernel drivers installed inside the container image.
  2. Full CUDA Toolkit installation (only runtime libraries are required).

When you run:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
The container uses its internal `nvidia-smi` binary, communicates through mounted `/dev/nvidia0` device nodes, and talks to the **host's kernel driver**.

---

# 6. Local Workstation Build, Execution & Image Transfer

Before moving an AI container to the cloud or supercomputer, you must build and validate it locally on your development machine.

### 6.1 Local Application Code Setup (`app.py`)
Below is the minimal Python inference application using `llama-cpp-python`. Create this file inside your local `containerization/` directory:

```python
# app.py - GPU-Accelerated LLM Inference
from llama_cpp import Llama
import sys

print("\n🚀 INITIALIZING GPU-ACCELERATED MODEL...")
# n_gpu_layers=-1 offloads ALL neural network layers to GPU memory
llm = Llama(
    model_path="/models/SmolLM2-360M-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    verbose=False
)
print("✅ Model successfully loaded into GPU memory.\n")

prompt = "Explain why GPUs are superior to CPUs for AI in one short sentence."
print(f"👤 USER PROMPT: {prompt}\n🤖 AI RESPONSE: ", end="", flush=True)

output = llm(
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=100,
    stop=["<|im_end|>"],
    stream=True
)

for chunk in output:
    print(chunk["choices"][0]["text"], end="", flush=True)

print("\n\n✅ Execution finished cleanly.")
```

### 6.2 Building the Local Docker Image
Open your terminal in the directory containing `Dockerfile` and `app.py`:

```bash
# 1. Build the local image and tag it as 'gpu-demo:v1'
docker build -t gpu-demo:v1 .

# 2. Verify the built image in local storage
docker image ls gpu-demo
```

### 6.3 Local GPU Execution & Bind Mount Pattern
Run the container on your local machine using `--gpus all`. We mount a local `models/` directory into the container so large model weight files do not need to be duplicated inside the image tarball:

```bash
# Verify local GPU visibility inside container
docker run --rm --gpus all gpu-demo:v1 nvidia-smi

# Execute the AI application with local model bind mount
docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  gpu-demo:v1
```

### 6.4 Image Transport Strategy 1: Container Registry Push (`docker push`)
To transfer the image over the internet to a cloud VM or remote host via a public or private registry (Docker Hub / GCP Artifact Registry):

```bash
# 1. Log in to Docker Hub (or GCP Artifact Registry)
docker login

# 2. Tag the image with your registry username
docker tag gpu-demo:v1 your-dockerhub-username/gpu-demo:v1

# 3. Push the image layers to the remote registry
docker push your-dockerhub-username/gpu-demo:v1
```

### 6.5 Image Transport Strategy 2: Offline Tarball Archive (`docker save` / `docker load`)
**Essential for University Sessions**: When presenting in lecture halls or auditoriums with poor internet bandwidth, uploading multi-gigabyte container images to a cloud registry is unviable. Use offline tarball export instead:

```bash
# 1. Export the container image into a portable .tar file on your laptop
docker save -o gpu-demo.tar gpu-demo:v1

# 2. Copy 'gpu-demo.tar' to a USB drive or transfer via local network

# 3. Import the container image on the target machine (Cloud VM or target laptop)
docker load -i gpu-demo.tar
```
*Note: The external model weights file (`SmolLM2-360M-Instruct-Q4_K_M.gguf`) remains outside the image and is transferred alongside the `.tar` file.*

---

# 7. Cloud Infrastructure Concepts & GCP Selection

### 7.1 Overview of Cloud Compute Architecture
Moving a containerized AI workload to Google Cloud Platform (GCP) involves renting a remote physical server sliced into a Virtual Machine (VM). GCP manages the hardware datacenter, power, cooling, and network infrastructure, allowing developers to provision compute on-demand.

### 7.2 Deep Dive into GCP Machine Families
Google Compute Engine categorizes VMs into distinct families optimized for different workloads:

```text
                                 GCP MACHINE FAMILIES
                                          │
       ┌──────────────────────────┬───────┴──────────────────┬──────────────────────────┐
       ▼                          ▼                          ▼                          ▼
[ N1 General Purpose ]    [ G2 Accelerator ]        [ A2 / A3 Heavy AI ]       [ E2 Cost-Optimized ]
- Flexible CPU/RAM        - NVIDIA L4 GPUs          - NVIDIA A100 / H100       - Shared-core CPU
- NVIDIA T4 Support       - High Inference          - Massive Training         - NO GPU SUPPORT
- Ideal for Demos         - Production Ready        - Enterprise Budget        - Light Web Apps
```

#### Detailed Family Comparison:

1. **N1 Series (First Generation General Purpose)**:
   - **Characteristics**: Highly customizable CPU-to-memory ratios. Supports attaching NVIDIA T4, V100, and P100 GPUs.
   - **Why We Select It**: The **N1 + NVIDIA T4 GPU** combination is available in almost every GCP region, supports 16GB VRAM, and is the most cost-effective tier for academic demonstrations and prototype inference.

2. **G2 Series (Next-Gen AI Inference)**:
   - **Characteristics**: Pre-bundled with NVIDIA L4 GPUs (24GB VRAM, Ada Lovelace architecture).
   - **Use Case**: Production-grade LLM inference and video processing (up to 2.5x higher performance than T4).

3. **A2 & A3 Series (High-Performance Supercomputing)**:
   - **Characteristics**: Bundled with NVIDIA A100 (40GB/80GB VRAM) or H100 (80GB VRAM) GPUs with NVLink interconnects.
   - **Use Case**: Distributed multi-node training of multi-billion parameter foundation models. High hourly cost ($3.50 to $30+/hour).

4. **E2 & N2 Series (General Purpose CPU Only)**:
   - **Characteristics**: E2 uses shared-core CPUs for cost savings. N2 uses modern Intel Ice Lake CPUs.
   - **Important Note**: **E2 instances do NOT support GPU attachments.** If you select E2 by mistake, the GCP Console will block GPU additions.

### 7.3 Hardware Memory Engineering: VRAM vs System RAM vs Disk

A major point of failure when deploying cloud AI workloads is confusing the three distinct memory layers:

$$\text{Total Memory Infrastructure} = \text{Disk Space (Storage)} + \text{System RAM (Host)} + \text{VRAM (GPU Memory)}$$

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ DISK STORAGE (Boot Disk: 50 GB)                                         │
│ Stores Linux OS, Docker layers, PyTorch libraries, & GGUF model files.  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Loads model into memory at startup
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SYSTEM RAM (Host Memory: 15 GB - n1-standard-4)                         │
│ Runs host OS, Docker daemon, & buffers model tensors during load.       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Offloads tensor operations to GPU
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ GPU VRAM (Accelerator Memory: 16 GB - NVIDIA T4)                        │
│ Holds active model weights & KV-cache during real-time LLM inference.   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Memory Allocation Rules of Thumb:
1. **Boot Disk Sizing**: Default GCP boot disk size is 10 GB. This will **fail** when installing CUDA drivers and pulling multi-gigabyte models. Always expand boot disk to **at least 50 GB**.
2. **System RAM Allocation**: System RAM should be $\ge 2\times$ the size of the neural network model file so the OS can buffer the load process without triggering Out-Of-Memory (OOM) kills.
3. **VRAM Bottleneck**: If a model requires 12 GB of VRAM, an NVIDIA T4 (16 GB VRAM) runs it perfectly in hardware. If the model exceeds VRAM, execution either crashes with `CUDA out of memory` or falls back to CPU execution, slowing token generation by 50x.

---

# 8. Step-by-Step Live Session Walkthrough (Pure Web UI Path)

*(This guide uses **100% Google Cloud Console Web UI**. Zero local `gcloud` terminal setup required!)*

---

### Step 1: Provision the GPU Virtual Machine in Google Cloud Console

1. Open your web browser and navigate to **[Google Cloud Console](https://console.cloud.google.com/)**.
2. Log in with your account credentials.
3. At the top of the window, select your active **GCP Project** from the dropdown menu.
4. Click the **Navigation Menu** (**☰** icon in top-left corner).
5. Hover over **Compute Engine** and click **VM instances**.

```text
[ GCP Console Top Bar ] ──> Select Project
  │
  ├──> [ Navigation Menu (☰) ]
         │
         └──> Compute Engine
                │
                └──> VM instances ──> Click [+ CREATE INSTANCE]
```

6. Click the **+ CREATE INSTANCE** button at the top of the VM instances page.

7. **Basic Details Configuration**:
   - **Name**: `ai-hosting-vm`
   - **Region**: `us-central1 (Iowa)`
   - **Zone**: `us-central1-a`

8. **Machine Configuration & Adding GPU**:
   - In the **Machine configuration** block, click on the **GPUs** tab (located next to General Purpose, Compute-optimized).
   - **GPU Type**: Select **NVIDIA T4**.
   - **Number of GPUs**: Select **1**.
   - **Machine Type**: Select **n1-standard-4** (4 vCPU, 15 GB RAM).
   - *(Note: GCP Console automatically sets "On host maintenance" to "Terminate" because GPU instances do not support live migration).*

9. **Boot Disk Configuration (Crucial Step)**:
   - Scroll down to the **Boot disk** section.
   - Click the **Change** button.
   - Set **Operating system**: **Ubuntu**.
   - Set **Version**: **Ubuntu 22.04 LTS (x86/64, amd64)**.
   - Set **Boot disk type**: **Balanced persistent disk**.
   - Set **Size (GB)**: Type `50` (Do not leave at default 10 GB!).
   - Click **Select**.

10. **Firewall Settings**:
    - Under the **Firewall** section on the main creation page, check the box for **Allow HTTP traffic**.

11. **Network Tag Configuration**:
    - Scroll down and click **Advanced options** to expand settings.
    - Click **Networking** to expand network options.
    - In the **Network tags** input box, type `ai-server` and press **Enter**.
    - *(This tag connects our VM directly to the custom firewall rule we create in Step 2).*

12. **Deploy**:
    - Scroll to the bottom and click **Create**.
    - Wait ~60 seconds until a green checkmark (**✓**) appears next to `ai-hosting-vm` in the list.

---

### Step 2: Create a Custom VPC Firewall Rule for API Port 8080

By default, GCP blocks all external internet traffic into a VM except SSH (Port 22) and standard HTTP (Port 80). We must open Port 8080 for our AI server.

1. Click the **Navigation Menu** (**☰**) in top-left corner.
2. Scroll down to **VPC network** and click **Firewall**.
3. At the top of the Firewall rules page, click **+ CREATE FIREWALL RULE**.

```text
[ Navigation Menu (☰) ] ──> VPC network ──> Firewall ──> [+ CREATE FIREWALL RULE]
```

4. Configure the following fields:
   - **Name**: `allow-ai-port`
   - **Network**: Leave as `default`
   - **Priority**: Leave as `1000`
   - **Direction of traffic**: `Ingress` (incoming)
   - **Action on match**: `Allow`
   - **Targets**: Select **Specified target tags** from the dropdown menu.
   - **Target tags**: Type `ai-server` and press **Enter** (matches the tag assigned to VM).
   - **Source filter**: Select **IPv4 ranges**.
   - **Source IPv4 ranges**: Type `0.0.0.0/0` (allows incoming traffic from any IP).
   - **Protocols and ports**: Select **Specified protocols and ports**.
   - Check the **TCP** checkbox, and type `8080` in the text field next to it.
5. Click **Create** at the bottom.

---

### Step 3: Connect via In-Browser SSH and Run Master Setup Script

1. Return to **Compute Engine** > **VM instances** in the GCP Console.
2. Find `ai-hosting-vm` in the table.
3. In the **Connect** column on the far right, click the **SSH** button.
4. A browser window opens containing a web terminal connected to your cloud VM.

```text
[ Compute Engine ] ──> [ VM instances ] ──> Find 'ai-hosting-vm' ──> Click [ SSH ]
                                                                       │
                                                                       ▼
                                                    (Opens Browser Terminal Window)
```

5. In the SSH browser terminal, create the automated setup script file:
   ```bash
   nano GCP-MASTER-SETUP.sh
   ```

6. Copy the complete script below and paste it directly into the SSH terminal window (Right-click > **Paste**, or `Ctrl+V`):

```bash
#!/bin/bash
# ==============================================================================
# ONE-CLICK GCP SETUP SCRIPT FOR GPU AI DEMO
# ==============================================================================
set -e

echo "🚀 [1/4] Expanding partition to utilize full 50GB disk..."
sudo growpart /dev/sda 1 2>/dev/null || true
sudo resize2fs /dev/sda1 2>/dev/null || true

echo "🚀 [2/4] Installing NVIDIA GPU Drivers..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq ubuntu-drivers-common
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq nvidia-driver-535

echo "🚀 [3/4] Installing Docker and NVIDIA Container Toolkit..."
sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "🚀 [4/4] Starting Hosted llama-server AI Container on Port 8080..."
mkdir -p ~/models
curl -L "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true" -o ~/models/SmolLM2-360M-Instruct-Q4_K_M.gguf

sudo docker run -d \
    --name hosted-ai-api \
    --restart always \
    --gpus all \
    -p 8080:8080 \
    -v ~/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 999

echo "✅ SUCCESS! AI Server is running live on GPU port 8080."
```

7. Save and exit `nano`: Press `Ctrl+O`, hit `Enter`, then press `Ctrl+X`.

8. Make the script executable and run it:
   ```bash
   chmod +x GCP-MASTER-SETUP.sh
   ./GCP-MASTER-SETUP.sh
   ```
9. Wait ~3–4 minutes for driver setup and container launch to complete.

---

### Step 4: Test the Live Cloud API from Your Local Laptop

1. Look at your GCP Console **VM instances** page and copy the **External IP** assigned to `ai-hosting-vm` (e.g., `34.123.45.67`).

2. Open the terminal on your **local laptop** (or any computer connected to the internet) and execute the streaming `curl` request:

```bash
# Replace 34.123.45.67 with your VM's actual External IP
curl http://34.123.45.67:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
          "prompt": "<|im_start|>user\nWhat is Cloud Computing?<|im_end|>\n<|im_start|>assistant\n",
          "max_tokens": 100,
          "stream": true
        }'
```

3. **Observe Output**: Because `"stream": true` is passed, you will see real-time streaming text tokens returned from your cloud GPU server directly to your laptop screen!

---

### Step 5: Resource Teardown (Prevent Cloud Billing)

Once the demonstration is finished, delete the resources immediately to prevent unwanted charges:

1. **Delete VM**: Go to **Compute Engine** > **VM instances**, check box next to `ai-hosting-vm`, click **Delete** (trash icon) at top.
2. **Delete Firewall Rule**: Go to **VPC network** > **Firewall**, check box next to `allow-ai-port`, click **Delete**.

---

# 9. HPC Supercomputing & Apptainer/Singularity for Academia

### 9.1 Why University HPC Clusters Forbid Docker
A common point of frustration for university researchers is discovering that campus supercomputing clusters refuse to install Docker.

```text
                              THE DOCKER SECURITY PROBLEM ON HPC
                              
  [ Unprivileged Student User ] ────> Runs: docker run -v /:/host_root ubuntu
                                                     │
                                                     ▼
  [ Docker Daemon (Runs as ROOT) ] ──> Mounts physical host root filesystem
                                                     │
                                                     ▼
  [ Result: Student obtains FULL ROOT CONTROL over campus supercomputer physical host! ]
```

#### Security Risks of Docker in Shared Academic Environments:
1. **Root Daemon Execution**: Docker commands communicate with `dockerd`, which executes as `root`.
2. **Privilege Escalation**: A user who can execute `docker run` can mount host system directories (`-v /etc:/host_etc`) and overwrite host passwords, gaining full root administrative control over the physical node.
3. **Setuid / Capabilities**: Docker containers can grant internal Linux capabilities that bypass multi-tenant security isolation.

### 9.2 The Apptainer (Singularity) Solution
**Apptainer** (formerly Singularity) was built specifically for High-Performance Computing (HPC) and academic environments.

```text
+-----------------------------------------------------------------------------------+
|                        APPTAINER (SINGULARITY) FOR HPC                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. Rootless Execution  ──> Runs strictly under the student's unprivileged UID.   |
|                                                                                   |
|  2. Immutable File (.sif)─> Entire container compiled into ONE read-only file.   |
|                                                                                   |
|  3. Native Host Access  ──> Direct access to HPC Infiniband & GPUs via --nv flag. |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

#### Key Differences Between Docker and Apptainer:

| Feature | Docker | Apptainer (Singularity) |
| :--- | :--- | :--- |
| **Primary Audience** | Cloud microservices, enterprise web apps | Supercomputing, university research labs |
| **Security Model** | Root daemon (`dockerd` root privileges required) | **Rootless / Unprivileged** (runs as current user) |
| **Image Format** | Layered tar archives stored in daemon cache | **Single `.sif` file** (Singularity Image Format) |
| **User Mapping** | User inside container can be `root` | User inside container is **identical to host user ID** |
| **File System Binding** | Isolated virtual mounts (`/app`) | Automatically mounts host `$HOME` and `$CWD` |
| **GPU Flag** | `--gpus all` | `--nv` (NVIDIA) or `--rocm` (AMD) |

### 9.3 Apptainer Recipe (`apptainer.def`) Reference Implementation

```apptainer
Bootstrap: docker
From: nvidia/cuda:12.1.1-runtime-ubuntu22.04

%post
    # Run setup commands during image compilation
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
        ca-certificates
    rm -rf /var/lib/apt/lists/*

    pip3 install --no-cache-dir \
        llama-cpp-python \
        huggingface_hub \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

%environment
    # Set runtime environment variables
    export LC_ALL=C
    export PATH="/usr/local/bin:$PATH"

%runscript
    # Default action executed when running .sif image
    exec python3 -c "print('🚀 Hello from Apptainer HPC Container with GPU!')"
```

### 9.4 Building and Running Apptainer Images

1. **Build `.sif` file on local workstation or build node**:
   ```bash
   apptainer build gpu-demo.sif apptainer.def
   ```
   *(Or pull directly from Docker Hub)*:
   ```bash
   apptainer pull gpu-demo.sif docker://ubuntu:22.04
   ```

2. **Execute Apptainer container on HPC Cluster with GPU**:
   ```bash
   apptainer run --nv \
     --bind "$PWD/models:/models:ro" \
     gpu-demo.sif
   ```

- `--nv`: Automatically binds host HPC NVIDIA GPU libraries and `/dev/nvidia*` devices into the container.
- `--bind`: Mounts host directories into the container filesystem read-only (`:ro`).

---

# 10. Faculty Presentation Prompts & Educational Use Cases

When presenting to university leadership, faculty heads, and research directors, use these strategic narrative points:

### 10.1 Paradigm Shift: Reproducible Scientific Research
> **Speaker Prompt:**  
> *"In traditional academic research, a graduate student publishes a paper with complex PyTorch scripts. When a peer reviewer attempts to reproduce the study two years later, new software updates break the environment. By submitting a compiled **Apptainer `.sif` container** alongside research papers, universities ensure 100% computational reproducibility. The exact experiment can be rerun 10 years later producing identical results down to the floating-point calculation."*

### 10.2 Institutional Cost Savings & "Cloud Bursting"
> **Speaker Prompt:**  
> *"University supercomputers experience heavy queue congestion during semester assignment deadlines or conference submission dates. By standardizing student workloads into portable containers, universities implement **Cloud Bursting**. During normal operation, student containers execute free of charge on campus HPC clusters. During peak deadline spikes, workloads dynamically bursting onto Google Cloud or AWS GPU instances without modifying code."*

### 10.3 Multi-Tenant Isolation in Student Labs
> **Speaker Prompt:**  
> *"Teaching AI in computer labs with shared GPU hardware is difficult because students install conflicting CUDA drivers and Python versions. Containerization enforces sandbox isolation. Every student gets a pristine container environment without risk of corrupting shared operating system files or accessing other students' files."*

---

# 11. Emergency Cheatsheet & Diagnostic Troubleshooting

### 11.1 Issue: GCP VM Disk Full (`No space left on device`)
**Symptom**: During `docker build` or model download, setup fails with disk write errors even though a 50GB boot disk was created.  
**Root Cause**: GCP images format the primary partition to 10GB by default, leaving 40GB unallocated.  
**Fix**: Execute partition expansion commands inside the VM terminal:
```bash
# Expand partition 1 on primary disk
sudo growpart /dev/sda 1 2>/dev/null || true
# Resize ext4 filesystem to consume full disk
sudo resize2fs /dev/sda1 2>/dev/null || true

# Verify disk space (Look for /dev/sda1 showing ~50G)
df -h /
```

### 11.2 Issue: Container Cannot See GPU (`NVIDIA driver mismatch`)
**Symptom**: `docker run --gpus all` returns: `could not select device driver "" with capabilities: [[gpu]]`.  
**Root Cause**: The host NVIDIA Container Toolkit is missing runtime configuration or Docker daemon was not restarted.  
**Fix**:
```bash
# Re-configure NVIDIA container runtime hook
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service
sudo systemctl restart docker

# Test baseline GPU passthrough
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 11.3 Issue: Port 8080 Unreachable from Laptop (`Connection Refused / Timed Out`)
**Symptom**: `curl http://<VM_EXTERNAL_IP>:8080/v1/completions` hangs or returns Connection Timed Out.  
**Diagnostic Checklist**:
1. **Check Container Status**: Inside VM SSH, run `sudo docker ps`. Verify container is running and shows `0.0.0.0:8080->8080/tcp`.
2. **Check In-VM Port Listening**: Run `netstat -tuln | grep 8080` or `curl http://localhost:8080/v1/models` inside VM SSH. If localhost works inside VM but fails externally, the issue is GCP Firewall.
3. **Verify GCP Firewall Rule**:
   - Go to GCP Console > **VPC network** > **Firewall**.
   - Ensure rule `allow-ai-port` exists, direction is `Ingress`, target tag is `ai-server`, source is `0.0.0.0/0`, and port is `tcp:8080`.
   - Go to **Compute Engine** > **VM instances** > Click `ai-hosting-vm`. Verify under **Network tags** that tag `ai-server` is assigned to VM.

### 11.4 Key Diagnostic Commands Summary Table

| Task | Command |
| :--- | :--- |
| **Inspect Host GPUs** | `nvidia-smi` |
| **Inspect PCI GPU Hardware** | `lspci \| grep -i nvidia` |
| **Check Docker Service Status** | `sudo systemctl status docker` |
| **View Active Containers** | `sudo docker ps -a` |
| **Stream Live Container Logs** | `sudo docker logs -f --tail 100 <container_name>` |
| **Execute Interactive Shell in Container**| `sudo docker exec -it <container_name> /bin/bash` |
| **Inspect Kernel & Hardware Logs** | `dmesg \| grep -i nvidia` |
| **Check Port Binding & Network Listeners**| `sudo ss -tulpn \| grep 8080` |

---

# 12. Audience Q&A & Presenter Mastery

Below are the top 12 questions students, researchers, and faculty will ask during your presentation. Click any question during your presentation to reveal the answer!

<details>
<summary><b>❓ Q1: "If containers share the host kernel, can I run a Windows container on a Linux host (or vice versa)?"</b></summary>
<br>

> **Answer**: No. Containers rely directly on the host OS kernel's system calls. A Linux container requires a Linux kernel. When running Docker Desktop on Windows or Mac, Docker silently runs a lightweight Linux Virtual Machine (via WSL2 or HyperKit) under the hood to supply that kernel.
</details>

<br>

<details>
<summary><b>❓ Q2: "Does running Python inside a container add performance overhead compared to bare metal?"</b></summary>
<br>

> **Answer**: Virtually zero. Unlike Virtual Machines, which translate hardware instructions through a hypervisor, containerized processes execute directly as native processes on the host CPU and GPU. Performance is 99%+ identical to bare metal.
</details>

<br>

<details>
<summary><b>❓ Q3: "What happens if two containers try to use the same port (e.g., Port 8080) on the host machine?"</b></summary>
<br>

> **Answer**: Inside their isolated network namespaces, both containers can listen on port 8080 internally. However, when publishing ports to the host (`-p host:container`), each host port can only be bound once. You would map Container A to `-p 8080:8080` and Container B to `-p 8081:8080`.
</details>

<br>

<details>
<summary><b>❓ Q4: "Why is my AI Docker image 6 GB to 8 GB when my Python script is only 5 KB?"</b></summary>
<br>

> **Answer**: The Python script is tiny, but the container image includes the complete user-space operating system baseline (Ubuntu), the Python interpreter, heavy CUDA runtime C++ dynamic libraries (`libcuda.so`, `libcudnn.so`), PyTorch/LLM binaries, and baked-in model weights.
</details>

<br>

<details>
<summary><b>❓ Q5: "I edited one line of Python code, and Docker started redownloading all 4 GB of CUDA packages again. Why?"</b></summary>
<br>

> **Answer**: Docker builds images in sequential read-only layers. If you place `COPY app.py .` near the top of your `Dockerfile` above `RUN pip install`, modifying `app.py` invalidates the build cache for that line and every line below it. **Rule of thumb:** Always put stable dependencies near the top, and your code at the bottom.
</details>

<br>

<details>
<summary><b>❓ Q6: "What is the difference between `RUN` and `CMD` in a Dockerfile?"</b></summary>
<br>

> **Answer**: `RUN` executes commands **during the build phase** to construct the image layers (e.g., installing packages). `CMD` defines the default command executed **at runtime** when the container actually starts up.
</details>

<br>

<details>
<summary><b>❓ Q7: "Do I need to install CUDA Toolkit inside the container if NVIDIA drivers are installed on the host?"</b></summary>
<br>

> **Answer**: You do NOT need kernel drivers inside the container, but the application inside the container still needs user-space CUDA runtime libraries (like `libcuda.so`). That is why we start from base images like `nvidia/cuda:12.1.1-runtime-ubuntu22.04`.
</details>

<br>

<details>
<summary><b>❓ Q8: "What does `--gpus all` actually do when I run `docker run`?"</b></summary>
<br>

> **Answer**: It tells the NVIDIA Container Toolkit hook to mount physical host GPU device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) into the container's user space process tree. Without `--gpus all`, the container is completely blind to host GPUs.
</details>

<br>

<details>
<summary><b>❓ Q9: "Can two containers share the same GPU at the same time?"</b></summary>
<br>

> **Answer**: Yes! NVIDIA GPUs support time-slicing and multi-process service (MPS). Multiple containers can submit CUDA kernels to the same GPU simultaneously, provided the total VRAM usage of both containers does not exceed the GPU's memory limit.
</details>

<br>

<details>
<summary><b>❓ Q10: "My cloud VM has 50 GB disk space and 15 GB System RAM. Why did my model crash with 'Out of Memory'?"</b></summary>
<br>

> **Answer**: You ran out of **VRAM (GPU Memory)**, not disk space or system RAM. Neural network model weights must fit inside the physical GPU's VRAM (e.g., 16 GB on an NVIDIA T4). Disk space stores files, System RAM runs OS processes, but GPU VRAM executes model tensors.
</details>

<br>

<details>
<summary><b>❓ Q11: "What is Quantization (e.g., GGUF Q4_K_M) and why do we use it?"</b></summary>
<br>

> **Answer**: Quantization compresses floating-point model weights from 16-bit precision (`FP16`) down to 4-bit integers (`INT4`). This reduces the model's VRAM footprint by ~75% with minimal loss in accuracy, allowing us to run AI models on standard cost-effective GPUs like the T4.
</details>

<br>

<details>
<summary><b>❓ Q12: "When should I use a single `docker run` command versus Docker Compose?"</b></summary>
<br>

> **Answer**: Use `docker run` for single, standalone tasks. Use **Docker Compose** when your application relies on multiple connected microservices—such as a React Frontend container communicating with a Python AI Backend container and a Vector Database container.
</details>
