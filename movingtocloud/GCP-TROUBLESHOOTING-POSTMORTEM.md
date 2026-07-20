# GCP L4 GPU AI Deployment: Exhaustive Troubleshooting Post-Mortem

This document serves as an exhaustive record of every trap, bug, and "gotcha" encountered while attempting to deploy a Dockerized AI model on a Google Cloud Platform (GCP) L4 GPU instance, and exactly how we solved them.

---

## 1. The Disk Space Trap (No Space Left on Device)
**The Problem:**
By default, GCP provisions new VMs with a tiny **10 GB boot disk**. AI models (even small ones like SmolLM) and Docker base images (like `nvidia/cuda`) require significant space. Attempting to build the Docker image resulted in a fatal `No space left on device` error halfway through the build.

**The Solution:**
1. **Always provision at least 50 GB** when creating the VM (`--boot-disk-size=50GB`).
2. If you already created it, you can increase the size in the GCP Console, but Linux does not automatically recognize the new space while running. You must run:
   ```bash
   sudo growpart /dev/sda 1
   sudo resize2fs /dev/sda1
   ```

---

## 2. The Debian vs. Ubuntu Trap (DKMS Compilation Failure)
**The Problem:**
GCP's default OS is often **Debian 13**. We attempted to use Google's official automated script (`install_gpu_driver.py`) to install the NVIDIA drivers. The script failed with:
`ERROR: An error occurred while performing the step: "Building kernel modules".`
Debian 13 uses custom cloud kernels (`deb13-cloud-amd64`) but the Debian repositories do not always host the matching `linux-headers` required for DKMS to compile the NVIDIA driver from scratch.

**The Solution:**
1. **Never use Debian for this.** Always create the VM using **Ubuntu 22.04 LTS** (`ubuntu-2204-lts`).
2. **Never compile from scratch.** Bypassing Google's python compilation script entirely and using Canonical's official, pre-compiled Ubuntu binaries guarantees success and installs in seconds:
   ```bash
   sudo apt-get update
   sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq nvidia-driver-535
   sudo modprobe nvidia
   ```

---

## 3. The GPG Keyring Trap (Interactive Freezes)
**The Problem:**
While installing the NVIDIA Container Toolkit, the command to import the GPG key (`gpg --dearmor`) silently crashed or froze. This happened because a previous broken attempt left an empty keyring file on the disk, and `gpg` was secretly waiting for the user to press "Y" to overwrite it, but the prompt was hidden.

**The Solution:**
Always force non-interactive mode and explicit overwrites in setup scripts:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

---

## 4. The Docker Build Freeze (The MTU Bug)
**The Problem:**
During `docker build`, the process completely froze during the Hugging Face model download (around the `WARNING: Running pip as root` stage). It sat at 0% for 10+ minutes without throwing an error.

**The Cause:**
This is a notorious GCP network bug. GCP's Virtual Private Cloud (VPC) uses a Maximum Transmission Unit (MTU) of **1460 bytes**. Docker's internal bridge network defaults to **1500 bytes**. When downloading large files (like a 250MB AI model), packets exceed 1460 bytes and are silently dropped by the GCP network, causing an infinite freeze.

**The Solution:**
Tell Docker to bypass its internal bridge and use the GCP host's network directly during the build:
```bash
sudo docker build --network=host -t gpu-demo:v1 .
```

---

## 5. Billing and Instance Management (Stop vs. Delete)
**The Problem:**
GPU instances (like the L4) cost roughly $0.70/hour. Leaving them running when not in use results in massive unexpected bills. However, deleting the instance means losing all the hard work, Docker images, and driver configurations.

**The Solution:**
- **STOP the instance:** This pauses compute billing (CPU/RAM/GPU). You only pay a few pennies a day for the 50GB persistent disk. Your files, Docker images, and drivers are perfectly preserved.
- **When restarting:** You do not need to run any setup scripts again. Simply SSH in and run:
  ```bash
  sudo docker run --rm --gpus all gpu-demo:v1
  ```

---

## 6. The Final, Bulletproof Setup Script
Combining all these lessons learned, here is the flawless, zero-interaction master script for a fresh **Ubuntu 22.04 LTS (50GB Disk)** instance:

```bash
#!/bin/bash
echo "🚀 [1/3] Expanding partitions (just in case)..."
sudo growpart /dev/sda 1 2>/dev/null || true
sudo resize2fs /dev/sda1 2>/dev/null || true

echo "🚀 [2/3] Installing pre-compiled NVIDIA Drivers & Docker..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq nvidia-driver-535 docker.io curl gnupg
sudo modprobe nvidia

echo "🚀 [3/3] Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
