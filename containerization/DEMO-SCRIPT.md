# Live Container Demonstration

Run from the `containerization` directory.

## Docker

```bash
nvidia-smi
docker build -t gpu-demo:v1 .
docker image ls gpu-demo

docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  gpu-demo:v1
```

The model remains outside the image. `--gpus all` exposes the host GPU and the bind mount exposes the model read-only.

## Save and load the image

```bash
docker save -o gpu-demo.tar gpu-demo:v1
docker load -i gpu-demo.tar
```

The model file is not inside `gpu-demo.tar` and must be transferred separately.

## Apptainer

Push the Docker image to an OCI registry, then:

```bash
apptainer pull gpu-demo.sif \
  docker://YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1

apptainer run --nv \
  --bind "$PWD/models:/models:ro" \
  gpu-demo.sif
```

For the complete procedure and explanations, see `book/CUDA_TO_CLOUD.md`.
