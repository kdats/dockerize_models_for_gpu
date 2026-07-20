# CUDA to Cloud Session Kit

Complete material for the 25 July 2026 sessions on moving local GPU workloads to cloud and using Docker/Apptainer for HPC.

Start with the [complete book](book/CUDA_TO_CLOUD.md). Generated versions are available in `book/output/`.

## Reproduce the local GPU demo

Prerequisites: NVIDIA driver, Docker, NVIDIA Container Toolkit, Python, and `huggingface_hub`.

```powershell
git clone <PRIVATE_REPOSITORY_URL>
cd sessionpresssidency

python -m pip install huggingface_hub
python .\containerization\models\download_model.py

cd containerization
docker build -t gpu-demo:v1 .

$modelDirectory = (Resolve-Path .\models).Path
docker run --rm --gpus all `
  --mount "type=bind,source=$modelDirectory,target=/models,readonly" `
  gpu-demo:v1
```

Model files, container archives, caches, credentials, and SSH private keys are intentionally excluded from Git.

## Rebuild the book

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\book\build.ps1
```
