# CUDA to Cloud Session Kit

Complete material for  moving local GPU workloads to cloud and using Docker/Apptainer for HPC.

Start with the [complete book](book/CUDA_TO_CLOUD.md). Generated versions are available in `book/output/`.

## Reproduce the local GPU demo

Prerequisites: NVIDIA driver, Docker, NVIDIA Container Toolkit, Python, and `huggingface_hub`.

```powershell
git clone https://github.com/kdats/dockerize_models_for_gpu.git
cd dockerize_models_for_gpu

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

## License and external components

Original repository code and documentation are available under the [MIT License](LICENSE). External runtimes, cloud services, container images, and model weights retain their own licenses and terms; see [NOTICE.md](NOTICE.md). Cloud GPU resources can incur charges.

## Rebuild the book

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\book\build.ps1
```
