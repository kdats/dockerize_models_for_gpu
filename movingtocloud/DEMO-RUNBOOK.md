# Cloud Session Demo Runbook (Live GCP)

**Objective**: Demonstrate that the exact same Docker container built locally runs identically on a Cloud GPU VM.

## Prerequisites Before the Session
- You must have the `gcloud` CLI installed locally and authenticated (`gcloud auth login`).
- You must have a Docker Hub account and be logged in locally (`docker login`).

---

## Step 1: Push Local Image to Registry

*Do this live. It shows how the container leaves your laptop.*

```bash
# 1. Tag the image with your Docker Hub username
docker tag gpu-demo:v1 <your-dockerhub-username>/gpu-demo:v1

# 2. Push the image to the public registry
docker push <your-dockerhub-username>/gpu-demo:v1
```

## Step 2: Spin up a GPU VM in GCP

*Run this from your local terminal. This provisions a VM with a T4 GPU.*

```bash
gcloud compute instances create gpu-demo-vm \
    --machine-type=n1-standard-4 \
    --zone=us-central1-a \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB
```

## Step 3: Connect and Setup

```bash
# Connect to the VM (gcloud handles SSH keys automatically!)
gcloud compute ssh gpu-demo-vm --zone=us-central1-a
```

*(Optional Pro-Tip: To avoid making the faculty watch NVIDIA drivers install for 4 minutes during the live demo, you can provision this VM before the session, install the drivers, and leave the VM in a "Stopped" state. Then just start it live.)*

If starting fresh, run these inside the VM:
```bash
# 1. Install NVIDIA Drivers
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot  # Reconnect after reboot using gcloud compute ssh again

# 2. Verify GPU is attached
nvidia-smi

# 3. Install Docker & NVIDIA Container Toolkit
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 4: The Grand Finale (Run the Container)

Now, run the exact same command you ran on your local machine, but on the cloud VM:

```bash
sudo docker run --rm --gpus all <your-dockerhub-username>/gpu-demo:v1
```

**Teaching Point to the Faculty:**
> *"Notice the output. It is identical to what ran on my local laptop. I didn't change the code, I didn't worry about CUDA versions on this cloud VM, and I didn't reinstall Python packages. The container is the boundary."*

## Step 5: Clean Up (Crucial to avoid billing!)

Back on your local machine terminal:
```bash
gcloud compute instances delete gpu-demo-vm --zone=us-central1-a --quiet
```
