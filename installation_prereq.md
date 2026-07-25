### Windows

*Note: On Windows, GPU support in containers is handled natively through Windows Subsystem for Linux (WSL 2). You do not install a separate container toolkit via CLI; you just install the host NVIDIA Windows driver and Docker Desktop.*

**1. Install WSL 2 (if not already installed):**

```powershell
wsl --install

```

**2. Install Docker Desktop:**

```powershell
winget install -e --id Docker.DockerDesktop

```

*(Restart your computer, open Docker Desktop, and ensure "Use the WSL 2 based engine" is checked in Settings).*

---

### Linux: Ubuntu (Debian-based)

**1. Install Docker Engine:**

```bash
# Add Docker's official GPG key and repo
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

```

**2. Install NVIDIA Container Toolkit:**

```bash
# Add NVIDIA's GPG key and repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```

---

### Linux: RHEL / CentOS / Fedora

**1. Install Docker Engine (or use Podman):**

```bash
# Add Docker repo and install
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker

```

**2. Install NVIDIA Container Toolkit:**

```bash
# Add NVIDIA's repo
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```
