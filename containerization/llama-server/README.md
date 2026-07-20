# Fallback Demo — llama.cpp Server

Use this if the **primary GPU demo** (`demo-app/`) has issues with CUDA, drivers, or the Python wheel.

## When to use this

- Primary demo crashes with CUDA errors
- NVIDIA Container Toolkit not configured correctly
- Driver version mismatch

## What this does

Runs the **official llama.cpp inference server** inside Docker, then queries it via `curl`.
Same model (Llama 3.2 1B), same GPU, different delivery method.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit installed (for GPU)
- Internet access to pull the image (first time only)

## Run

```bash
bash run.sh
```

## If GPU still fails

Remove `--gpus all` from `run.sh` line 21. The server runs on CPU.
Containerization is still fully demonstrated — just slower inference.

## Teaching point

> "Same container, same model — now running as an API server. Any app can call it.
> This is how production AI inference is deployed in the real world."
