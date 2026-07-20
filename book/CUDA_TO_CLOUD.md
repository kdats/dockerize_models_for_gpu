---
title: "From Local CUDA to Cloud and HPC"
subtitle: "A Practical Guide to Moving GPU Workloads with Docker and Apptainer"
author: "Faculty Development Programme — CUDA to Cloud"
date: "25 July 2026"
---

# Preface

GPU programming does not end when a CUDA kernel or AI model runs successfully on a developer's workstation. The next challenge is portability: running the same workload on another GPU machine without repeating days of environment setup and debugging.

This book follows one practical journey. We begin with an AI inference workload running on a local NVIDIA GPU. We then reproduce the workload on a cloud GPU virtual machine. Docker gives us a repeatable software environment, while Apptainer extends the same container approach to shared high-performance computing systems.

The objective is not to survey every service offered by AWS, Azure, or Google Cloud. It is to understand the small set of ideas that remain constant across all three providers:

1. A GPU workload has code, dependencies, data or model weights, and hardware requirements.
2. A cloud GPU is attached to a virtual machine that must be provisioned, secured, monitored, and stopped when it is no longer needed.
3. A container packages the user-space software environment, but it does not contain the physical GPU or replace the host's NVIDIA driver.
4. Repeating the same workload successfully on a different machine is the practical test of portability.

# 1. The Workload We Are Moving

The phrase *move a workload* can sound more complicated than it is. A workload is simply everything required to perform a useful computation.

For the demonstration used in this programme, the useful computation is language-model inference. A prompt is submitted to a small language model, the model performs numerical operations using a GPU, and generated text is returned.

The workload contains:

- the Python application that submits the prompt;
- the `llama.cpp` Python runtime used for inference;
- the quantized SmolLM model file;
- CUDA-compatible user-space libraries;
- an execution command; and
- access to an NVIDIA GPU through the container runtime.

The workload is not identical to a Docker image. A Docker image is one way to package much of the workload. Model files can be included in an image, downloaded during setup, or mounted separately. Likewise, credentials and changing datasets should normally remain outside the image.

## 1.1 What was demonstrated

The workload first ran on a local NVIDIA GPU. The same application, dependencies, model, and execution pattern were then reproduced on a Google Cloud GPU virtual machine. Successful inference on both systems established the essential result:

```text
Local NVIDIA GPU                         Google Cloud GPU
----------------                         ----------------
same application       ------------->    same application
same model             ------------->    same model
same container recipe  ------------->    same container recipe
GPU inference          ------------->    GPU inference
```

The previously built Docker image did not have to be copied byte-for-byte to the cloud. Rebuilding from the same Dockerfile is still a valid migration method. Transferring a prebuilt image through a registry is another method, discussed later.

## 1.2 Three ways to move the environment

There are three common approaches.

### Manual reconstruction

Copy the application to the cloud VM, install Python and every dependency, download the model, and run the program. This works, but it is easy for local and cloud environments to diverge.

### Rebuild from a container recipe

Copy the source code and Dockerfile, then build the image on the cloud VM. This is the approach used in the practical exercise. The Dockerfile acts as an executable record of the environment.

### Transfer a prebuilt image

Build once, push the image to a container registry, and pull it on the cloud VM. This avoids rebuilding and ensures that the same packaged artifact is executed in both places.

None of these changes the meaning of workload migration. They differ only in how the software environment is transported or reconstructed.

# 2. The Minimum Cloud Model

A cloud GPU workload needs four infrastructure elements:

1. **Compute:** a virtual machine with a suitable GPU.
2. **Storage:** a boot disk plus storage for code, models, data, and results.
3. **Networking:** a controlled path for administration and data transfer.
4. **Identity and access:** rules defining who can create and connect to resources.

The provider names differ, but the concepts remain stable.

| Need | AWS | Microsoft Azure | Google Cloud |
|---|---|---|---|
| GPU virtual machine | EC2 GPU instance | N-series virtual machine | Compute Engine with GPU |
| Object storage | Amazon S3 | Azure Blob Storage | Cloud Storage |
| Container registry | Amazon ECR | Azure Container Registry | Artifact Registry |
| Managed batch jobs | AWS Batch | Azure Batch | Google Cloud Batch |

Learning one provider deeply is more useful than memorizing three user interfaces. Once compute, storage, networking, and identity are understood, the mapping to another provider is mostly a vocabulary exercise.

## 2.1 Choose the GPU from the workload

Do not begin with a provider's longest instance list. Begin with the workload.

For inference, ask:

1. How much GPU memory is required to load the model?
2. Which numerical precision or quantization does the runtime use?
3. Is the workload one interactive request, a stream of requests, or a batch?
4. Does it require one GPU or communication between multiple GPUs?
5. Is the objective compatibility, low latency, throughput, or minimum cost?

For a small quantized model, a single inference-oriented GPU is enough. Large-model training has entirely different memory, interconnect, storage-throughput, and multi-node requirements. Choosing an H100 for a tiny instructional model would demonstrate cloud provisioning, but not sound engineering judgment.

The names evolve over time. AWS groups GPU-equipped offerings under accelerated-computing EC2 instances. Azure lists GPU-optimized families including NC and ND for compute-oriented work. Google Cloud offers accelerator-optimized machine series and also permits selected GPUs on some general-purpose machine types. Always inspect the current provider documentation for availability and compatibility.

## 2.2 GPU memory, system memory, and disk

These three capacities are different:

- **GPU memory (VRAM)** holds model parameters, activations, CUDA workspaces, and other device data.
- **System memory (RAM)** holds the operating system, application processes, preprocessing data, and any model portions not resident on the GPU.
- **Disk** stores the operating system, container layers, model files, build cache, inputs, and results.

A machine can have sufficient disk space for a model but insufficient VRAM to execute it. It can also have sufficient VRAM but run out of disk space while building the container.

For a first estimate:

```text
VRAM requirement
≈ model weights
+ runtime workspace
+ input/output state
+ safety margin
```

Measure the actual application after the estimate. Quantization reduces the storage and memory occupied by model weights, but it does not make every other memory cost disappear.

## 2.3 Regions, zones, quotas, and capacity

A provider may advertise a GPU without making it immediately available to every account in every location.

- A **region** is a geographic cloud location.
- A **zone** is an isolated deployment area within a region.
- A **quota** limits how many resources an account may allocate.
- **Capacity** indicates whether the requested hardware is presently available in the selected location.

Before a scheduled demonstration, confirm quota and create the intended instance in the intended zone. A command can be syntactically correct and still fail because quota is zero or regional capacity is unavailable.

## 2.4 Storage choices

Use the VM boot disk for the operating system and temporary working files. Use persistent disks when files must remain attached to the VM across stops and starts. Use object storage for durable datasets, models, results, and sharing between machines.

Object storage is not a normal mounted directory by default. Applications usually upload and download objects through a command-line tool, library, or managed filesystem layer. For a small demonstration, downloading a model directly to the VM is adequate. For a repeatable project, record the exact object location, version, and integrity information.

## 2.5 Networking and the smallest secure path

Administrative access normally requires SSH. An externally reachable inference API additionally requires an ingress firewall rule and a published container port.

These are different paths:

```text
Administrator --SSH/22--> VM shell

Client --HTTP/application port--> containerized inference server
```

Opening SSH does not automatically publish the application. Publishing a Docker port does not automatically permit traffic through the cloud firewall. Both layers must allow the intended connection.

For a teaching deployment, expose only the required port, restrict the source where practical, avoid placing credentials in command history or images, and remove temporary rules after the demonstration.

# 3. Virtual Machines and Containers

A virtual machine and a container solve different isolation problems.

A virtual machine presents virtualized hardware and runs its own guest operating-system kernel. A cloud GPU VM is therefore a complete remote computer that can be started, connected to, stopped, and deleted.

A container isolates an application and its user-space environment while sharing the host kernel. It packages files, libraries, environment settings, and the startup command. Containers normally start faster and use less storage than complete virtual machines.

In the demonstrated architecture, these layers are nested:

```text
Physical cloud server
└── GPU virtual machine
    └── Linux operating system and NVIDIA driver
        └── Docker runtime
            └── AI inference container
                └── application and model
```

The virtual machine supplies the remote computer. Docker supplies the repeatable application environment.

## 3.1 Why a container is not a small virtual machine

A container image may contain an Ubuntu filesystem, but it does not boot an Ubuntu kernel. Processes inside the container ultimately use the host kernel. This is why a Linux container requires a Linux kernel, supplied on Windows by Docker Desktop's Linux VM or WSL2 integration.

The distinction matters for GPUs. Kernel-level NVIDIA driver components live on the host. The container cannot bring an arbitrary replacement kernel driver. It brings the application and compatible user-space libraries, while the GPU-aware runtime connects them to the host driver and device.

## 3.2 Remote development with SSH and VS Code

SSH is the essential remote-management mechanism. VS Code Remote SSH provides an editor and terminal interface over the same connection; it does not replace SSH or move computation back to the laptop.

The flow is:

```text
Local VS Code interface
        |
        | SSH connection
        v
VS Code server on cloud VM
        |
        v
Files, terminal, Docker and GPU execute remotely
```

When a remote folder is open, terminal commands execute on the VM. This must be visible to students: the editor window is local, but the filesystem and process are remote.

A minimal SSH configuration entry is:

```text
Host gpu-cloud
    HostName CLOUD_PUBLIC_IP
    User CLOUD_USER
    IdentityFile PATH_TO_PRIVATE_KEY
```

After connecting, verify location before running expensive or destructive commands:

```bash
hostname
pwd
nvidia-smi
```

Dev Containers can subsequently open the remote project inside its declared container environment. That is useful in development, but it is not required to explain workload migration; Remote SSH plus ordinary Docker commands is the clearer foundation.

# 4. Docker: Recipe, Image, and Container

Three terms form the foundation of Docker.

## 4.1 Dockerfile

A Dockerfile is a text recipe. It states which base environment to use, which dependencies to install, which application files to copy, and which command to execute.

## 4.2 Image

An image is the read-only packaged result produced by `docker build`. Images are identified by names and tags such as `gpu-demo:v1`.

## 4.3 Container

A container is a running instance of an image. Multiple containers can be started from the same image. The `--rm` option removes the stopped container after the program finishes; it does not delete the image.

The lifecycle is:

```text
Dockerfile
    |
    | docker build -t gpu-demo:v1 .
    v
Image: gpu-demo:v1
    |
    | docker run --rm --gpus all gpu-demo:v1
    v
Running container with GPU access
```

## 4.4 What `--gpus all` means

The image does not contain the physical GPU. The NVIDIA driver belongs to the host operating system. When Docker starts the container with `--gpus all`, the NVIDIA Container Toolkit exposes the permitted GPU devices and driver capabilities to the process inside the container.

This separation explains two observations:

- the same image can run on machines with different compatible NVIDIA GPUs; and
- a GPU-aware application may report no GPU when the container is started without GPU access.

## 4.5 Build context and layers

The final dot in the build command is meaningful:

```bash
docker build -t gpu-demo:v1 .
```

It identifies the current directory as the **build context**. `COPY` instructions can use files from this context. Docker sends the required context to the builder, so placing unrelated models, datasets, archives, or secrets in the directory can slow the build or expose files unintentionally.

A `.dockerignore` file excludes items that do not belong in the context:

```text
.git
models/
results/
*.tar
*.sif
.env
```

Each Dockerfile instruction contributes to an image's layer history. If an early instruction changes, later layers normally need to be rebuilt. Stable operations should therefore appear before frequently changing application code.

```dockerfile
FROM RUNTIME_BASE
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
```

Dependencies change less often than `app.py`, so Docker can reuse the dependency layer while rebuilding only the final application layer.

## 4.6 Tags and immutable identity

The tag in `gpu-demo:v1` is a human-readable pointer. It can be reassigned to a different image. The image digest identifies exact content.

For classroom iteration, `v1` and `v2` are understandable tags. For reproducible research or deployment, preserve the source revision, dependency versions, base-image digest, model version, and resulting image digest.

Avoid using `latest` as if it meant “tested.” It is merely a conventional tag name.

## 4.7 Containers are temporary; data is not

The writable layer of a container is tied to that container. When a temporary container is removed, changes inside that layer disappear. Inputs and results should therefore live outside it.

A bind mount maps an existing host directory into the container:

```bash
docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  --mount type=bind,source="$PWD/results",target=/results \
  gpu-demo:v1
```

The model mount is read-only because inference need not modify it. The results mount is writable. This makes ownership explicit and allows another container or host program to use the same files.

A Docker-managed volume also persists independently of a container:

```bash
docker volume create model-cache
docker run --rm --mount source=model-cache,target=/models gpu-demo:v1
```

Bind mounts are easiest when a student should see ordinary host files. Named volumes are useful when Docker should manage the storage location.

## 4.8 Environment variables and secrets

Runtime settings can be passed without rebuilding:

```bash
docker run --rm -e MAX_TOKENS=100 gpu-demo:v1
```

Environment variables are appropriate for non-secret configuration, but they can be visible through process or container inspection. Cloud credentials and private keys must not be copied into the Dockerfile or committed to Git. Use provider identity roles or an appropriate secrets mechanism.

## 4.9 Registries

A registry stores and distributes container images. The sequence is:

```text
Local image
    | tag
    | push
    v
Registry repository
    | pull
    v
Cloud VM or build system
```

Docker Hub is convenient for demonstrations. AWS ECR, Azure Container Registry, and Google Artifact Registry integrate with their respective identity systems. Private images require authentication, while public images should never contain private source, credentials, or licensed data that cannot be redistributed.

## 4.10 Reading a container command

Consider:

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v "$PWD/models:/models:ro" \
  IMAGE_NAME SERVER_ARGUMENTS
```

Read it from left to right:

- `docker run` creates and starts a container;
- `--rm` removes the stopped container;
- `--gpus all` grants GPU access;
- `-p 8080:8080` maps host port 8080 to container port 8080;
- `-v ...:/models:ro` mounts model files read-only;
- `IMAGE_NAME` selects the packaged environment; and
- the remaining values become arguments for the containerized program.

Understanding one complete command is more useful than memorizing many unrelated Docker options.

# 5. The Local-to-Cloud Demonstration

The demonstration should be understood as five essential stages.

## 5.1 Establish the local result

Run the inference workload on the local GPU and retain the prompt, generated response, GPU identification, and execution command. This becomes the reference result.

## 5.2 Provision a cloud GPU VM

Create a GPU-enabled VM in one region and zone. GPU availability depends on provider quota and regional capacity. The selected GPU need not match the local GPU exactly; the goal is application portability, not identical performance.

## 5.3 Prepare the host

The VM needs a working NVIDIA driver, Docker, and NVIDIA Container Toolkit. A successful `nvidia-smi` confirms that the operating system can communicate with the GPU. A successful GPU-aware Docker test confirms that containers can receive GPU access.

## 5.4 Reproduce and run the workload

Transfer or retrieve the Dockerfile and application, obtain the model, build the image, and run the same inference command. The expected proof is not identical execution time or word-for-word generated text. The proof is that the same workload executes correctly using the cloud GPU without application redesign.

## 5.5 Stop the resource

GPU VMs are billed resources. Stop or delete the VM when the exercise ends, and inspect disks, reserved addresses, snapshots, and other retained resources separately. Stopping compute does not necessarily remove every related charge.

## 5.6 What remains local and what moves

It helps to classify the parts of the workload before migration.

| Part | Typical treatment |
|---|---|
| Application source | Copy, clone from Git, or include in the image |
| Python and native libraries | Install manually or package in the image |
| Model weights | Download from a model repository or mount from storage |
| Input dataset | Transfer to object storage or mount from shared storage |
| Secrets and credentials | Supply at runtime; never bake into the image |
| NVIDIA driver | Install on the host VM |
| CUDA user-space runtime | Install on the host or package in the container |
| Results | Write to persistent disk or object storage |

This explains why workload migration does not require copying one enormous file. Different components can move through the mechanism best suited to them.

## 5.7 Rebuild versus registry transfer

The practical exercise rebuilt the image on the cloud VM. This made the Dockerfile the portable artifact:

```bash
docker build -t gpu-demo:v1 .
docker run --rm --gpus all gpu-demo:v1
```

A registry-based workflow instead builds locally, attaches a registry-qualified name, pushes the image, and pulls it on the cloud VM:

```bash
docker tag gpu-demo:v1 REGISTRY/ACCOUNT/gpu-demo:v1
docker push REGISTRY/ACCOUNT/gpu-demo:v1

# On the cloud VM
docker pull REGISTRY/ACCOUNT/gpu-demo:v1
docker run --rm --gpus all REGISTRY/ACCOUNT/gpu-demo:v1
```

Rebuilding is appropriate for learning and for transparent recipes. Registry transfer is valuable when an exact tested artifact must be promoted between environments. Both are valid ways to move the workload.

## 5.8 Lift-and-shift versus refactoring

**Lift-and-shift** preserves the application architecture while changing where it runs. A local command-line inference program is copied or containerized and executed on a cloud GPU VM with minimal code change.

This approach is suitable for rapid access to a different GPU, temporary computation, validation on a second environment, or migration with low development effort. Its limitation is that the application still behaves like a local program: a user connects to the VM, starts the command, watches output, and manages files.

**Refactoring** changes the application to use cloud-oriented services or operating patterns. Examples include converting the command-line program into an inference API, reading models from object storage, submitting batch jobs, using a managed endpoint, or adding automatic scaling.

Refactoring adds development work and provider-specific decisions. A sound sequence is:

```text
1. Prove the unchanged workload on one cloud GPU.
2. Identify the actual operational limitation.
3. Refactor only the part that solves that limitation.
```

The demonstrated local-to-GCP inference is primarily lift-and-shift. Running a network inference server is a small refactoring that changes how clients reach the workload without changing its fundamental GPU computation.

## 5.9 Reproducibility is more than a Dockerfile

A Dockerfile improves reproducibility, but floating versions and remote downloads can produce different images on different days.

A reproducibility record should include:

- source-code and Dockerfile revisions;
- base-image tag or digest;
- exact dependency versions;
- model repository, filename, quantization, and revision;
- build command and relevant build arguments;
- final image digest;
- host GPU and driver version; and
- the execution command.

```text
Repeatable recipe: “I can perform these steps again.”
Reproducible artifact: “I can identify and run the same packaged result.”
Reproducible experiment: “I can recreate the inputs, environment and method.”
```

Containerization contributes to all three, but it does not automatically version datasets, models, prompts, random seeds, or cloud infrastructure.

## 5.10 Migration acceptance criteria

Before declaring a migration successful, define the expected result. For this inference workload:

- the container starts on the cloud VM;
- the intended model is loaded;
- the application demonstrates GPU offload;
- the prompt receives a valid response;
- required results or logs are retained; and
- the VM can be stopped or deleted without losing required output.

Performance equality is not required because the GPUs differ. Identical generated text is not required unless deterministic generation is configured. The acceptance criterion is functional equivalence of the intended workload.

# 6. Case Study: Local GPU to Google Cloud L4

The project repository records an actual migration attempt using a local NVIDIA GPU and a Google Cloud VM equipped with an NVIDIA L4. The application performed inference with a quantized SmolLM model through a CUDA-enabled `llama.cpp` runtime.

The value of this case study is not that every cloud migration should use the same VM, driver, or model. Its value is that it exposes the boundaries between application, container, operating system, driver, storage, and network.

## 6.1 Local reference execution

The local execution established four facts:

1. Docker could start Linux containers through WSL2.
2. The NVIDIA Container Toolkit could expose the local GPU to a container.
3. The model could be loaded by the inference runtime.
4. A prompt produced generated text.

These facts form the baseline. Without a working local reference, a cloud failure could be caused either by the application or by the new infrastructure.

## 6.2 Cloud execution

The cloud side repeated the same essential layers:

```text
Google Cloud GPU VM
├── Ubuntu host
├── NVIDIA driver
├── Docker Engine
├── NVIDIA Container Toolkit
└── CUDA-enabled inference container
    ├── llama.cpp runtime
    ├── SmolLM model
    └── prompt and generated response
```

The hardware changed from the local GPU to an L4, but the application-level operation remained inference through the same runtime and model family. This is the portability claim being demonstrated.

## 6.3 Provisioning decisions

The VM needs enough CPU, memory, disk space, and GPU capacity for the chosen workload. The project discovered that a small default boot disk was insufficient for container layers, build cache, packages, and model weights. The corrected deployment used a larger boot disk.

This produces a general sizing method:

```text
required disk space
= operating system
+ installed packages and drivers
+ Docker image layers and build cache
+ model weights and datasets
+ working space and results
```

GPU memory must be considered separately. A model may fit on disk but fail to load into GPU memory. Quantization reduces the model's memory requirement and makes a small instructional workload practical.

## 6.4 The driver boundary

The NVIDIA driver is installed on the VM host. It connects the operating-system kernel to the physical GPU. The container carries the application and compatible user-space CUDA libraries.

Verification therefore proceeds from the outside inward:

```bash
# Host can see the GPU
nvidia-smi

# Docker can start containers
docker run --rm hello-world

# A container can see the GPU
docker run --rm --gpus all CUDA_TEST_IMAGE nvidia-smi

# The application can use the GPU
docker run --rm --gpus all gpu-demo:v1
```

Each command tests one boundary. If the host cannot see the GPU, changing the application will not help. If the host sees the GPU but the container does not, the container runtime configuration is the likely boundary to inspect.

## 6.5 Lessons from failed attempts

### Disk exhaustion

Container builds temporarily retain layers and download archives. The project encountered a `No space left on device` error on a small boot disk. The lesson is to check available space before a build and include build cache in sizing calculations.

### Operating-system and driver compatibility

One attempted operating-system image did not provide the expected kernel-header path for the chosen driver installation process. Moving to an Ubuntu image with a supported driver path resolved that attempt. The transferable lesson is to select a documented, supported combination of cloud image, kernel, and NVIDIA driver rather than assuming every Linux image behaves identically.

### Non-interactive package setup

An existing keyring file caused a package-signing command to wait for confirmation. Automated setup must either be deliberately interactive or explicitly non-interactive. A command that works once by hand is not automatically a reliable provisioning script.

### Container build networking

A large model download stalled while building through Docker's bridge network. Building with the host network resolved the observed case. The broader lesson is to separate application errors from infrastructure transfer problems and to avoid downloading large, changing model files inside an image build when they can be managed as external data.

## 6.6 A better model-storage boundary

Including a model in an image creates one self-contained artifact, but rebuilding the image repeats the model download and changing the model invalidates a large layer. Mounting the model keeps the runtime image and model lifecycle separate:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models" \
  GPU_RUNTIME_IMAGE \
  -m /models/model.gguf
```

This approach expresses two independent concerns:

- the image defines *how inference runs*;
- the mounted file defines *which model runs*.

The same principle applies to datasets and generated results.

## 6.7 From batch inference to a hosted service

Running one prompt inside a temporary container proves execution. Running an inference server demonstrates how the workload can serve remote clients:

```text
Laptop client
    |
    | HTTP request
    v
Cloud firewall rule
    |
    v
VM port mapped to container port
    |
    v
Inference server using cloud GPU
```

The additional concepts are port publishing, process lifetime, and network access control. They do not change the GPU execution model. For an instructional deployment, firewall access should be restricted as narrowly as practical and the rule removed after use.

## 6.8 Evidence of successful migration

A migration demonstration should preserve simple evidence:

- local GPU identification;
- cloud GPU identification;
- the image tag or Dockerfile revision;
- model name and quantization;
- the prompt;
- successful generated output in both environments; and
- the cleanup or stopped-instance status.

Inference text need not be identical because generation may be nondeterministic. The evidence required is successful execution of the same intended workload, not identical hardware or timing.

## 6.9 Monitoring the workload

Monitoring should answer three questions: Is the process running? Is it using the GPU? Is the expected output being produced?

At the host level, `nvidia-smi` reports GPU identity, memory use, utilization, temperature, and processes known to the driver:

```bash
watch -n 1 nvidia-smi
```

At the container level:

```bash
docker ps
docker logs CONTAINER_NAME
docker stats CONTAINER_NAME
```

For a background server, name the container so its state and logs are easy to retrieve. GPU utilization may fluctuate for a small model because inference alternates between request handling, CPU work, memory transfer, and GPU kernels. One low utilization sample does not prove that the GPU is unused.

## 6.10 Comparing performance responsibly

Local and cloud timing is affected by model loading, quantization, offloaded layers, prompt length, token count, CPU preprocessing, disk cache, runtime build, concurrent activity, and network latency.

Measure model-loading time separately from generation. Record prompt tokens, output tokens, and tokens per second when available. Repeat the run before drawing conclusions.

This book uses timing to observe behaviour, not to publish a benchmark. A valid benchmark requires controlled inputs, warm-up, repeated trials, statistical reporting, and documented software and hardware configurations.

## 6.11 Cost control

Cloud cost depends on resource type, location, duration, storage, data transfer, and purchasing model. Prices change, so always use the provider's current calculator.

```text
estimated compute cost
= hourly VM/GPU price × expected running hours

estimated retained cost
= persistent disks + snapshots + reserved addresses + other services
```

Configure a budget alert, select the smallest compatible GPU, stop or delete resources after verification, label resources with owner and purpose, and re-check the console after scripted cleanup. A budget alert informs; it does not necessarily stop resources automatically.

## 6.12 Security boundaries

Use a cloud identity with only the necessary permissions and enable multi-factor authentication. Prefer VM identity roles over copying long-lived provider credentials to the VM.

Protect SSH private keys and keep them out of Docker images, Git, slides, and recordings. Use known container-image sources and inspect Dockerfiles before building.

An inference server exposed to the internet needs authentication, encrypted transport, input limits, logging, and narrow firewall rules before production use. For a temporary classroom demonstration, minimize exposure and remove the rule afterward.

# 7. From Docker to Apptainer for HPC

Docker is convenient on a developer workstation or a cloud VM where the user controls the machine. A shared HPC cluster has different requirements. Many users submit jobs to the same system, and administrators generally avoid giving users access to a privileged container daemon.

Apptainer descends from the Singularity project and was adopted by the Linux Foundation under its current name. SingularityCE also continues as a related implementation. A cluster may therefore provide an `apptainer` or `singularity` command; follow the software and policy supported by that site. This book uses Apptainer terminology and commands. It runs containers using the invoking user's identity and integrates naturally with shared filesystems, GPUs, and batch schedulers.

The conceptual translation is direct:

```text
Docker development image
        |
        | convert or pull
        v
Apptainer SIF image
        |
        | apptainer run --nv
        v
GPU job on an HPC compute node
```

Docker and Apptainer are not competing answers to the same operational setting. A common workflow uses Docker during development and Apptainer for execution on an HPC cluster.

## 7.1 Why HPC centres prefer Apptainer

An HPC cluster is shared infrastructure. Users normally log in to a front-end node and submit jobs to compute nodes through a scheduler. They do not administer the machines and should not require a permanently running privileged Docker daemon.

Apptainer is designed around that environment:

- the process in the container runs as the invoking user;
- the user's permissions remain meaningful on shared filesystems;
- container images can be stored as single SIF files;
- host directories can be bind-mounted into the container; and
- NVIDIA GPU libraries can be exposed with `--nv`.

## 7.2 Image sources

Apptainer can create or retrieve an image from a container-registry reference. In a connected environment, the conceptual command is:

```bash
apptainer pull gpu-demo.sif docker://REGISTRY/ACCOUNT/gpu-demo:v1
```

The resulting `gpu-demo.sif` is a portable image file. On a cluster, it can be copied to permitted storage and executed without running a Docker daemon.

## 7.3 Running with a GPU

The essential command is:

```bash
apptainer run --nv gpu-demo.sif
```

`--nv` exposes the NVIDIA devices and appropriate host driver libraries. It plays a role similar to Docker's `--gpus all`, although the runtimes implement GPU integration differently.

## 7.4 Persistent data

Containers should not be treated as the permanent home of inputs and results. A host directory can be bound into an Apptainer container:

```bash
apptainer run --nv \
  --bind "$PWD/models:/models" \
  --bind "$PWD/results:/results" \
  gpu-demo.sif
```

The model and results remain on the filesystem after the container process exits.

## 7.5 Scheduler integration

On a Slurm-based cluster, a job script requests resources and launches the workload:

```bash
#!/bin/bash
#SBATCH --job-name=gpu-inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

apptainer run --nv gpu-demo.sif
```

The scheduler decides where and when the job runs. Apptainer supplies the application environment. CUDA uses the GPU allocated to the job.

# 8. The Complete Portability Story

The two sessions form one continuous argument.

First, a GPU application that works locally is reproduced on a cloud GPU VM. This reveals the practical importance of drivers, dependencies, data, identity, networking, and cost control.

Second, containerization provides a repeatable description of the application environment. Docker supports local and cloud development. Apptainer carries the same container approach into a shared HPC system.

The final mental model is:

```text
Write and test GPU workload locally
                |
                v
Package the user-space environment
                |
        +-------+-------+
        |               |
        v               v
Cloud GPU VM       HPC compute node
Docker             Apptainer + scheduler
```

The GPU hardware may change. The infrastructure interface may change. The scheduler may change. The application and its declared environment remain the stable centre of the workflow.

# 9. Guided Lab: Read the Existing Workload

This lab is intentionally observational. Do not build or download anything until the recipe is understood.

## 9.1 Identify the components

Open the project Dockerfile and answer:

1. Which base image is selected?
2. Which dependencies are installed?
3. Is the model copied, downloaded, or mounted?
4. Which application file is placed in the image?
5. Which command runs when the container starts?

Then open the Python application and locate:

- the model path;
- the number of GPU layers requested;
- the prompt;
- the maximum generated token count; and
- the code that prints streamed output.

The purpose is to connect each source line to one element of the workload. A container should not be treated as a black box merely because it hides installation steps.

## 9.2 Draw the dependency path

Complete this path using the actual project names:

```text
Prompt
  -> Python application
  -> inference runtime
  -> model file
  -> CUDA user-space libraries
  -> host NVIDIA driver
  -> physical GPU
```

If one layer fails, identify which preceding verification command would isolate it.

# 10. Guided Lab: Build and Run Locally

## 10.1 Confirm the host

```bash
nvidia-smi
docker version
```

The first command verifies the host driver and GPU. The second verifies the Docker client and engine. Neither proves that a container can use the GPU.

## 10.2 Build

From the directory containing the Dockerfile:

```bash
docker build -t gpu-demo:v1 .
```

Observe the build output. Identify the base-image stage, dependency installation, model handling, application copy, and final image name.

Record:

```bash
docker image ls gpu-demo
docker image inspect gpu-demo:v1
```

## 10.3 Run with GPU access

```bash
docker run --rm --gpus all gpu-demo:v1
```

Record the GPU name, model name, prompt, generated response, and whether the process exits successfully.

## 10.4 Explain the result

The result proves that:

- Docker started the declared application environment;
- the runtime could locate the model;
- the NVIDIA container integration exposed a GPU; and
- inference completed.

It does not prove cloud portability yet. That requires repeating the workload on another machine.

# 11. Guided Lab: Move to a Cloud GPU

The exact provisioning command depends on provider, account, region, quota, and current machine offerings. The invariant sequence is more important than one copied command.

## 11.1 Before provisioning

Record:

- provider, project/subscription/account, region, and zone;
- selected machine and GPU;
- expected GPU memory;
- boot-disk size;
- quota confirmation;
- firewall exposure; and
- the cleanup command.

## 11.2 Provision and connect

Create one Linux GPU VM and connect through SSH. Immediately establish where commands are executing:

```bash
hostname
cat /etc/os-release
df -h
```

## 11.3 Verify from host to application

Proceed in this order:

```text
GPU attached to VM
    -> NVIDIA host driver works
    -> Docker works
    -> NVIDIA Container Toolkit works
    -> workload image exists
    -> model exists
    -> inference succeeds
```

Do not skip directly to the final application and then debug every layer at once.

## 11.4 Reproduce the workload

Choose one of two clear methods:

**Recipe method:** retrieve the Dockerfile and application, then rebuild on the VM.

**Registry method:** pull the exact previously built image.

Run the workload with GPU access and retain the same evidence captured locally.

## 11.5 Compare

| Evidence | Local | Cloud |
|---|---|---|
| Host name | | |
| GPU model | | |
| Container image/tag | | |
| Model | | |
| Prompt | | |
| Inference succeeded | | |
| Observed execution time | | |

Different GPU names and timings are expected. The portability result is successful execution without redesigning the application.

## 11.6 Stop and inspect billing resources

Stop or delete the VM according to whether its disk state must be preserved. Then inspect the provider console for remaining disks, snapshots, reserved addresses, images, and firewall rules.

# 12. Guided Lab: Docker to Apptainer

This lab belongs on a Linux system with Apptainer installed, ideally an HPC login environment approved by its administrators.

## 12.1 Obtain a SIF image

From an OCI/Docker registry:

```bash
apptainer pull gpu-demo.sif docker://REGISTRY/ACCOUNT/gpu-demo:v1
```

Inspect the result:

```bash
apptainer inspect gpu-demo.sif
ls -lh gpu-demo.sif
```

The SIF file is the packaged application environment used by Apptainer.

## 12.2 Run first without a scheduler

```bash
apptainer run --nv gpu-demo.sif
```

This verifies Apptainer and GPU integration on the current node. On many clusters, substantial GPU computation must run only on allocated compute nodes; follow the site's policy.

## 12.3 Submit through Slurm

Create a minimal job script:

```bash
#!/bin/bash
#SBATCH --job-name=gpu-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=gpu-demo-%j.out

apptainer run --nv gpu-demo.sif
```

Submit and observe:

```bash
sbatch gpu-demo.slurm
squeue --me
```

After completion, open the output file named with the Slurm job ID.

## 12.4 Identify responsibility boundaries

| Component | Responsibility |
|---|---|
| Slurm | Resource request, queueing, placement and job lifecycle |
| Apptainer | Application filesystem and runtime environment |
| Host driver | Kernel-level access to the allocated GPU |
| CUDA application | Performs the GPU computation |
| Shared filesystem | Stores image, model, inputs and outputs |

# 13. Troubleshooting by Boundaries

Troubleshooting becomes simpler when symptoms are mapped to layers.

## 13.1 The VM cannot be created

Check account quota, zone capacity, requested machine/GPU compatibility, billing status, and command parameters. Docker and application code are not yet involved.

## 13.2 `nvidia-smi` fails on the host

Check whether the GPU is attached, whether the selected operating-system image supports the installation path, whether the NVIDIA driver is installed, and whether a reboot is required. Do not debug Python first.

## 13.3 Docker works but a GPU test container fails

Check NVIDIA Container Toolkit installation and Docker runtime configuration. Restart Docker after configuration changes and confirm that the host still sees the GPU.

## 13.4 The container sees the GPU but inference fails

Check CUDA/runtime compatibility, model path, model format, GPU memory, application arguments, and container logs.

## 13.5 Build reports no disk space

Inspect both filesystem space and Docker usage:

```bash
df -h
docker system df
```

Increase the intended disk when necessary. Cleanup commands can delete useful images and cache, so inspect before removing anything.

## 13.6 A remote API cannot be reached

Check in order:

1. Is the server process running?
2. Is it listening on the expected interface and port?
3. Is the container port published to the host?
4. Does the cloud firewall allow the client's source?
5. Is the client using the correct current public address?

GPU execution and network reachability are separate concerns.

# 14. Review Questions

1. What parts of a GPU workload are normally packaged in a container?
2. Why does the NVIDIA driver remain on the host?
3. What does `--gpus all` add to `docker run`?
4. Why is rebuilding from the same Dockerfile still workload migration?
5. When is transferring a prebuilt image preferable?
6. Why should model weights often be mounted rather than baked into an image?
7. How do VRAM, RAM, and disk capacity differ?
8. Why might a valid VM creation request fail in one zone?
9. What does VS Code Remote SSH execute locally, and what executes remotely?
10. Why is Apptainer commonly used instead of Docker on shared HPC systems?
11. What does Slurm contribute that Apptainer does not?
12. Why should cloud cleanup include more than stopping the VM?

# 15. Further Reading

The following primary documentation should be consulted for current commands, supported versions, and available hardware:

- [AWS EC2 accelerated-computing instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [Azure GPU-optimized virtual-machine families](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview)
- [Google Cloud GPU machine types](https://docs.cloud.google.com/compute/docs/gpus)
- [NVIDIA Container Toolkit overview](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [NVIDIA Container Toolkit architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)
- [Apptainer GPU support](https://apptainer.org/docs/user/main/gpu.html)
- [Building Apptainer containers](https://apptainer.org/docs/user/main/build_a_container.html)
- [VS Code Remote Development over SSH](https://code.visualstudio.com/docs/remote/ssh)
- [VS Code development on a remote Docker host](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host)
- [Google Cloud CLI SSH reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/ssh)
- [Google Cloud CLI SCP reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/scp)
- [Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html)

# 16. Operator's Manual: Files Used in the Demonstration

This part is the complete implementation manual. Commands are grouped by the computer on which they run. Do not paste a command until the heading identifies the correct machine.

The demonstration uses these names consistently:

```text
Project folder:       cuda-cloud-demo
Docker image:         gpu-demo:v1
Cloud VM:             cuda-cloud-demo
Model repository:     bartowski/SmolLM2-360M-Instruct-GGUF
Model file:           SmolLM2-360M-Instruct-Q4_K_M.gguf
Container model path: /models/SmolLM2-360M-Instruct-Q4_K_M.gguf
Application file:     app.py
```

## 16.1 Placeholder register

Replace capitalized placeholders before running a command:

| Placeholder | Meaning | Example form |
|---|---|---|
| `YOUR_PROJECT_ID` | Google Cloud project ID | `faculty-gpu-demo` |
| `YOUR_WORKING_ZONE` | Zone confirmed to have quota/capacity | `region-zone` |
| `YOUR_REGISTRY` | Docker/OCI registry hostname | Provider or private registry host |
| `YOUR_ACCOUNT` | Registry namespace or account | Your registry user/project |
| `VM_NAME` | Cloud VM name in generic diagnostics | `cuda-cloud-demo` |
| `ZONE` | Cloud zone in generic diagnostics | The selected working zone |
| `CONTAINER_NAME` | Name passed to `docker run --name` | `llama-server` |
| `JOB_ID` | Slurm job ID returned by `sbatch` | Numeric scheduler ID |
| `CLUSTER_USER` | HPC login username | Site-assigned username |
| `CLUSTER_HOST` | HPC login hostname | Site-provided address |

Do not type placeholder text literally. PowerShell variables such as `$vmName` and `$zone` refer to values assigned earlier in the same terminal.

## 16.2 Directory layout

Create one working directory:

```text
cuda-cloud-demo/
├── Dockerfile
├── app.py
└── models/
    └── SmolLM2-360M-Instruct-Q4_K_M.gguf
```

The repository version stores the Dockerfile under `containerization/` and the application under `containerization/demo-app/`. The compact layout above is used in the manual because it is easier to copy to another machine.

## 16.3 Application code

Save the following as `app.py`:

```python
from llama_cpp import Llama

MODEL = "/models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
PROMPT = "Explain why GPUs are better than CPUs for AI in one short sentence."

print("Loading model...")
llm = Llama(
    model_path=MODEL,
    n_gpu_layers=-1,
    verbose=True,
)

formatted_prompt = (
    f"<|im_start|>user\n{PROMPT}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print(f"USER: {PROMPT}")
print("AI: ", end="", flush=True)

stream = llm(
    formatted_prompt,
    max_tokens=100,
    stop=["<|im_end|>"],
    stream=True,
)

for chunk in stream:
    print(chunk["choices"][0]["text"], end="", flush=True)

print("\nInference completed.")
```

Important lines:

- `model_path` identifies the mounted model file.
- `n_gpu_layers=-1` requests that all supported model layers be offloaded to the GPU.
- `verbose=True` keeps runtime messages visible so GPU offload can be observed.
- `stream=True` prints generated tokens as they arrive.

The generated sentence can vary. Successful model loading, GPU-offload messages, generated text, and a normal exit are the relevant evidence.

## 16.4 Dockerfile

Save the following as `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3 \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY app.py /app/app.py

CMD ["python3", "/app/app.py"]
```

This image contains the CUDA user-space runtime, Python, the CUDA-enabled `llama-cpp-python` wheel, and the application. The model is deliberately excluded. It is mounted at runtime, which keeps the image and model independent.

## 16.5 `.dockerignore`

Save the following as `.dockerignore`:

```text
.git
models/
results/
*.gguf
*.tar
*.sif
.env
```

The model remains available to `docker run` through a mount, but is not sent into `docker build`.

## 16.6 Download the model on Windows

From PowerShell inside `cuda-cloud-demo`:

```powershell
New-Item -ItemType Directory -Force -Path .\models | Out-Null

$modelUrl = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true"
$modelFile = ".\models\SmolLM2-360M-Instruct-Q4_K_M.gguf"

Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile
Get-Item $modelFile | Select-Object Name,Length
```

The file should have a non-zero size before continuing.

## 16.7 Download the model on Linux or WSL

```bash
mkdir -p models
curl -L \
  "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true" \
  -o models/SmolLM2-360M-Instruct-Q4_K_M.gguf
ls -lh models/SmolLM2-360M-Instruct-Q4_K_M.gguf
```

# 17. Operator's Manual: Local Windows and WSL2 Setup

This chapter starts from a Windows laptop with an NVIDIA GPU, WSL2, and Docker Desktop.

## 17.1 Identify the host GPU

Run in PowerShell:

```powershell
nvidia-smi
```

Expected evidence includes the NVIDIA driver version, CUDA compatibility reported by the driver, GPU name, and GPU memory.

The project's verified local machine reported an NVIDIA GeForce RTX 5060 Ti with approximately 16 GB of GPU memory.

## 17.2 Verify WSL2

```powershell
wsl --version
wsl --status
wsl --list --verbose
```

The Linux distribution used by Docker or development should report version 2.

Open the default distribution:

```powershell
wsl
```

Inside WSL, verify GPU visibility:

```bash
nvidia-smi
```

The Windows NVIDIA driver supplies WSL GPU support; do not install a second Windows display driver inside WSL.

## 17.3 Verify Docker Desktop

Run in PowerShell:

```powershell
docker version
docker info
```

The output must contain both client and server information. If only the client is shown, start Docker Desktop and wait until its engine is ready.

## 17.4 Verify container GPU access

Use a CUDA image compatible with the host driver:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

This command proves the complete path:

```text
Docker Desktop
-> NVIDIA container integration
-> WSL GPU support
-> Windows NVIDIA driver
-> physical GPU
```

It does not yet test the model.

## 17.5 Build the application image

From the directory containing `Dockerfile` and `app.py`:

```powershell
docker build -t gpu-demo:v1 .
docker image ls gpu-demo
```

The first build downloads the base image and Python wheel. Later builds can reuse unchanged layers.

## 17.6 Run local inference

PowerShell path mounting:

```powershell
$modelDirectory = (Resolve-Path .\models).Path

docker run --rm --gpus all `
  --mount "type=bind,source=$modelDirectory,target=/models,readonly" `
  gpu-demo:v1
```

Linux or WSL path mounting:

```bash
docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  gpu-demo:v1
```

## 17.7 Capture local evidence

Save the following before moving to cloud:

```powershell
nvidia-smi > local-nvidia-smi.txt
docker image inspect gpu-demo:v1 > local-image-inspect.json

$modelDirectory = (Resolve-Path .\models).Path
docker run --rm --gpus all `
  --mount "type=bind,source=$modelDirectory,target=/models,readonly" `
  gpu-demo:v1 2>&1 | Tee-Object local-inference.txt
```

The evidence files establish which GPU, image, model, command, and result were used locally.

# 18. Operator's Manual: Google Cloud Preparation

The commands in this chapter run on the local administration machine unless explicitly labelled **cloud VM**.

## 18.1 Install and initialize the Google Cloud CLI

After installing the Google Cloud CLI, verify and authenticate:

```powershell
gcloud version
gcloud auth login
gcloud auth list
```

Select the project:

```powershell
gcloud projects list
gcloud config set project YOUR_PROJECT_ID
gcloud config get-value project
```

Replace `YOUR_PROJECT_ID` with the project that has billing enabled.

## 18.2 Enable Compute Engine

```powershell
gcloud services enable compute.googleapis.com
```

## 18.3 Inspect available GPU accelerators

```powershell
gcloud compute accelerator-types list --filter="name:(nvidia-l4 nvidia-tesla-t4)"
```

Select a zone shown for the desired GPU. Record it once:

```powershell
$projectId = "YOUR_PROJECT_ID"
$zone = "YOUR_WORKING_ZONE"
$vmName = "cuda-cloud-demo"

gcloud config set project $projectId
gcloud config set compute/zone $zone
```

GPU quota and capacity must be available in the selected zone. A quota error is an account allocation issue; a resource-availability error can require another zone or GPU type.

## 18.4 Create an L4 VM

For an L4, use a G2 machine type in a zone where it is available:

```powershell
gcloud compute instances create $vmName `
  --zone=$zone `
  --machine-type=g2-standard-4 `
  --image-family=ubuntu-2204-lts-amd64 `
  --image-project=ubuntu-os-cloud `
  --boot-disk-size=50GB `
  --maintenance-policy=TERMINATE
```

If the chosen zone does not offer `g2-standard-4`, do not keep changing unrelated options. Return to the accelerator/machine availability check and select a compatible zone.

## 18.5 Inspect the created VM

```powershell
gcloud compute instances describe $vmName --zone=$zone `
  --format="table(name,status,machineType.basename(),zone.basename(),disks[0].diskSizeGb)"
```

Confirm that the status is `RUNNING` before connecting.

# 19. Operator's Manual: SSH and Remote Development

## 19.1 Connect with the Google Cloud CLI

```powershell
gcloud compute ssh $vmName --zone=$zone
```

On the first connection, `gcloud` can create an SSH key and register its public key. Never share the generated private key.

Inside the cloud VM, establish context:

```bash
hostname
whoami
cat /etc/os-release
df -h /
```

Exit back to the laptop with:

```bash
exit
```

## 19.2 Generate standard SSH configuration

On the laptop:

```powershell
gcloud compute config-ssh
```

Inspect the generated entries without publishing private-key content:

```powershell
Get-Content $HOME\.ssh\config
```

The entry maps a friendly host name to the VM address, user, and private-key path.

## 19.3 Connect with VS Code Remote SSH

1. Install the **Remote - SSH** extension.
2. Open the Command Palette.
3. Select **Remote-SSH: Connect to Host**.
4. Choose the generated GCP host entry.
5. Open the remote project directory.
6. Open a VS Code terminal and run `hostname`.

If `hostname` prints the cloud VM name, editor operations and terminal commands target the remote filesystem and machine.

## 19.4 Copy the project through `gcloud compute scp`

From the laptop, while positioned above the local `cuda-cloud-demo` directory:

```powershell
gcloud compute scp --recurse .\cuda-cloud-demo "${vmName}:~/" --zone=$zone
```

Do not copy the `models` folder if the model will be downloaded directly on the VM. To copy only the recipe and application:

```powershell
gcloud compute ssh $vmName --zone=$zone --command="mkdir -p ~/cuda-cloud-demo"
gcloud compute scp .\cuda-cloud-demo\Dockerfile "${vmName}:~/cuda-cloud-demo/" --zone=$zone
gcloud compute scp .\cuda-cloud-demo\app.py "${vmName}:~/cuda-cloud-demo/" --zone=$zone
gcloud compute scp .\cuda-cloud-demo\.dockerignore "${vmName}:~/cuda-cloud-demo/" --zone=$zone
```

On the cloud VM:

```bash
cd ~/cuda-cloud-demo
ls -la
```

# 20. Operator's Manual: Bootstrap the Cloud GPU Host

The following commands run **inside the cloud VM**.

## 20.1 Install the NVIDIA driver

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ubuntu-drivers-common \
  nvidia-driver-535
sudo reboot
```

The SSH connection closes during reboot. Wait for the VM to return to `RUNNING`, reconnect, and verify:

```bash
nvidia-smi
```

The output should identify the cloud GPU. In the recorded case study, this was an NVIDIA L4.

## 20.2 Install Docker

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  docker.io \
  curl \
  gnupg

sudo systemctl enable --now docker
sudo docker version
```

This manual keeps `sudo docker` on the cloud VM to avoid changing group membership during the lab.

## 20.3 Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 20.4 Verify the container-to-GPU boundary

```bash
sudo docker run --rm --gpus all \
  nvidia/cuda:12.1.1-base-ubuntu22.04 \
  nvidia-smi
```

Do not proceed to the model until this succeeds.

## 20.5 Reusable bootstrap script

Save as `bootstrap-gpu-host.sh` on an Ubuntu 22.04 GPU VM:

```bash
#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ubuntu-drivers-common \
  nvidia-driver-535 \
  docker.io \
  curl \
  gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable docker

echo "Driver installed. Reboot the VM, then verify nvidia-smi and restart Docker."
```

Run:

```bash
chmod +x bootstrap-gpu-host.sh
./bootstrap-gpu-host.sh
sudo reboot
```

# 21. Operator's Manual: Run Inference on the Cloud GPU

The following commands run **inside the cloud VM** after driver and container verification.

## 21.1 Download the model

```bash
cd ~/cuda-cloud-demo
mkdir -p models

curl -L \
  "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true" \
  -o models/SmolLM2-360M-Instruct-Q4_K_M.gguf

ls -lh models/SmolLM2-360M-Instruct-Q4_K_M.gguf
```

## 21.2 Build on the cloud VM

```bash
sudo docker build -t gpu-demo:v1 .
sudo docker image ls gpu-demo
```

This is the recipe-rebuild migration method. The Dockerfile and application moved; the environment was reconstructed on the cloud VM.

## 21.3 Run the same workload

```bash
sudo docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  gpu-demo:v1
```

Compare this command with the local Linux/WSL command. The image name, container model path, application, and GPU flag are unchanged. The host GPU is different.

## 21.4 Capture cloud evidence

```bash
nvidia-smi | tee cloud-nvidia-smi.txt
sudo docker image inspect gpu-demo:v1 > cloud-image-inspect.json

sudo docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  gpu-demo:v1 2>&1 | tee cloud-inference.txt
```

Copy evidence back to the laptop:

```powershell
gcloud compute scp "${vmName}:~/cuda-cloud-demo/cloud-*.txt" .\evidence\ --zone=$zone
gcloud compute scp "${vmName}:~/cuda-cloud-demo/cloud-image-inspect.json" .\evidence\ --zone=$zone
```

## 21.5 Registry-transfer method

To transfer an already built image, first use a registry-qualified tag on the local machine:

```powershell
docker tag gpu-demo:v1 YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1
docker push YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1
```

On the cloud VM:

```bash
sudo docker pull YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1
sudo docker run --rm --gpus all \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1
```

Use a private repository when the image is not intended for public distribution.

# 22. Operator's Manual: Host an Inference API

This chapter uses the official CUDA-enabled `llama.cpp` server image rather than adding web-server code to the Python application.

## 22.1 Start the server on the cloud VM

```bash
cd ~/cuda-cloud-demo

sudo docker run -d \
  --name llama-server \
  --gpus all \
  -p 8080:8080 \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 999
```

## 22.2 Watch startup logs

```bash
sudo docker logs -f llama-server
```

Stop following logs with `Ctrl+C`; this does not stop the container.

Check container state:

```bash
sudo docker ps --filter name=llama-server
```

## 22.3 Test from inside the VM

Testing locally on the VM separates application health from firewall configuration:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|im_start|>user\nExplain cloud GPU computing in one sentence.<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 100,
    "stream": false
  }'
```

## 22.4 Use SSH port forwarding

SSH forwarding avoids opening port 8080 to the public internet. Run on the laptop:

```powershell
gcloud compute ssh $vmName --zone=$zone -- -NL 8080:localhost:8080
```

Keep that SSH window open. In a second local PowerShell window:

```powershell
$request = @{
    prompt = "<|im_start|>user`nExplain cloud GPU computing in one sentence.<|im_end|>`n<|im_start|>assistant`n"
    max_tokens = 100
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8080/v1/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $request
```

The request travels through SSH to port 8080 on the VM; no public application firewall rule is required.

## 22.5 Stop the server

```bash
sudo docker stop llama-server
sudo docker rm llama-server
```

# 23. Operator's Manual: Stop, Start, and Delete GCP Resources

These commands run on the laptop.

## 23.1 Stop while preserving the boot disk

```powershell
gcloud compute instances stop $vmName --zone=$zone
gcloud compute instances describe $vmName --zone=$zone --format="value(status)"
```

The expected status is `TERMINATED`. Compute charges stop, but retained disks and some associated resources can continue to cost money.

## 23.2 Restart later

```powershell
gcloud compute instances start $vmName --zone=$zone
gcloud compute ssh $vmName --zone=$zone
```

Inside the VM:

```bash
nvidia-smi
sudo docker run --rm --gpus all \
  --mount type=bind,source="$HOME/cuda-cloud-demo/models",target=/models,readonly \
  gpu-demo:v1
```

## 23.3 Delete the VM

```powershell
gcloud compute instances delete $vmName --zone=$zone
```

Read the deletion prompt and confirm the exact name and zone.

## 23.4 Audit remaining resources

```powershell
gcloud compute instances list
gcloud compute disks list --filter="name~'$vmName'"
gcloud compute addresses list
gcloud compute firewall-rules list
gcloud compute snapshots list
```

The absence of the VM alone does not prove that every billable resource was removed.

## 23.5 Minimal PowerShell lifecycle script

Save as `gcp-vm.ps1`:

```powershell
param(
    [Parameter(Mandatory)]
    [ValidateSet("status", "start", "stop")]
    [string]$Action,

    [string]$Name = "cuda-cloud-demo",
    [Parameter(Mandatory)]
    [string]$Zone
)

switch ($Action) {
    "status" {
        gcloud compute instances describe $Name --zone=$Zone `
          --format="table(name,status,zone.basename(),machineType.basename())"
    }
    "start" {
        gcloud compute instances start $Name --zone=$Zone
    }
    "stop" {
        gcloud compute instances stop $Name --zone=$Zone
    }
}
```

Examples:

```powershell
.\gcp-vm.ps1 -Action status -Zone YOUR_WORKING_ZONE
.\gcp-vm.ps1 -Action stop -Zone YOUR_WORKING_ZONE
```

Deletion remains an explicit command rather than part of this convenience script.

# 24. Operator's Manual: Apptainer and Singularity for HPC

These commands run on a Linux system where Apptainer is installed. On a managed cluster, use the module and commands supplied by its administrators.

## 24.1 Install on Ubuntu or WSL2

The official Apptainer PPA provides current Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update
sudo apt-get install -y apptainer
```

Verify the installation with a small container:

```bash
apptainer --version
apptainer exec docker://alpine cat /etc/alpine-release
```

On a managed HPC system, do not install your own privileged runtime. Check available modules first:

```bash
module avail 2>&1 | grep -Ei 'apptainer|singularity'
module load apptainer
apptainer --version
```

The exact module name is controlled by the cluster.

## 24.2 Verify the runtime and GPU allocation

```bash
apptainer --version
nvidia-smi
```

On a scheduler-managed cluster, obtain a GPU allocation before running a GPU workload. Do not run heavy computation on a login node.

## 24.3 Convert from a registry image

Push `gpu-demo:v1` to an accessible registry, then run:

```bash
apptainer pull gpu-demo.sif \
  docker://YOUR_REGISTRY/YOUR_ACCOUNT/gpu-demo:v1
```

Verify:

```bash
apptainer inspect gpu-demo.sif
ls -lh gpu-demo.sif
```

## 24.4 Copy image and model to the cluster

From the laptop or authorized transfer node:

```bash
scp gpu-demo.sif CLUSTER_USER@CLUSTER_HOST:~/cuda-cloud-demo/
scp models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
  CLUSTER_USER@CLUSTER_HOST:~/cuda-cloud-demo/models/
```

## 24.5 Run interactively on an allocated GPU node

```bash
cd ~/cuda-cloud-demo

apptainer run --nv \
  --bind "$PWD/models:/models:ro" \
  gpu-demo.sif
```

The Docker `--gpus all` flag becomes Apptainer's `--nv`. The bind-mounted model remains outside the SIF image.

## 24.6 Execute a specific command

```bash
apptainer exec --nv \
  --bind "$PWD/models:/models:ro" \
  gpu-demo.sif \
  python3 /app/app.py
```

`run` uses the image's declared runscript. `exec` explicitly selects a command.

## 24.7 Slurm job script

Save as `gpu-demo.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=smollm-inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=smollm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

apptainer run --nv \
  --bind "$PWD/models:/models:ro" \
  gpu-demo.sif
```

Submit:

```bash
sbatch gpu-demo.slurm
squeue --me
```

After completion:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
cat smollm-JOB_ID.out
```

Cluster partitions, account names, GPU resource syntax, memory limits, and modules vary. Use the local scheduler documentation when additional `#SBATCH` fields are required.

# 25. Expected Outputs and Checkpoints

Use checkpoints to decide whether to proceed.

| Checkpoint | Command | Required evidence |
|---|---|---|
| Local driver | `nvidia-smi` | Local GPU listed |
| Local Docker | `docker version` | Client and server shown |
| Local container GPU | CUDA test container | GPU listed inside container |
| Local inference | `docker run ... gpu-demo:v1` | GPU offload and generated text |
| Cloud VM | `gcloud compute instances describe` | VM is running with intended type |
| Cloud driver | `nvidia-smi` | Cloud GPU listed |
| Cloud container GPU | CUDA test container | Cloud GPU listed inside container |
| Cloud inference | Same workload command | Model loads and generates text |
| SSH tunnel API | Local request to port 8080 | JSON completion response |
| Apptainer | `apptainer run --nv` | Model generates on allocated GPU |
| Slurm | `sacct` and job output | Completed state and inference output |
| Cleanup | Provider resource lists | Intended resources stopped or deleted |

Do not proceed past a failed checkpoint. Fix the boundary being tested first.

# 26. Troubleshooting Command Index

## 26.1 Identity and project

```powershell
gcloud auth list
gcloud config list
gcloud config get-value project
```

## 26.2 VM and zone

```powershell
gcloud compute instances list
gcloud compute instances describe VM_NAME --zone=ZONE
gcloud compute operations list --filter="status!=DONE"
```

## 26.3 Disk

```bash
df -h
sudo du -xh /var/lib/docker --max-depth=1 2>/dev/null | sort -h
sudo docker system df
```

## 26.4 Driver

```bash
lspci | grep -i nvidia
nvidia-smi
lsmod | grep nvidia
```

## 26.5 Docker

```bash
sudo systemctl status docker --no-pager
sudo docker version
sudo docker info
sudo docker ps -a
sudo docker logs CONTAINER_NAME
```

## 26.6 GPU inside Docker

```bash
sudo docker run --rm --gpus all \
  nvidia/cuda:12.1.1-base-ubuntu22.04 \
  nvidia-smi
```

## 26.7 Image build

```bash
sudo docker build --progress=plain -t gpu-demo:v1 .
sudo docker image ls
sudo docker system df
```

If a download stalls specifically inside the build but works on the host, compare container networking and host networking. In the recorded GCP attempt, this command resolved a stalled large download:

```bash
sudo docker build --network=host -t gpu-demo:v1 .
```

Use it as a diagnosis for that observed condition, not as an automatic default.

## 26.8 Server reachability

```bash
sudo docker ps --filter name=llama-server
sudo docker logs llama-server
curl http://127.0.0.1:8080/health
ss -ltnp | grep 8080
```

## 26.9 Apptainer

```bash
apptainer --version
apptainer inspect gpu-demo.sif
apptainer exec --nv gpu-demo.sif nvidia-smi
```

# 27. Instructor Rehearsal and Delivery Runbook

This runbook reduces the full manual to one ordered path. Every item refers to a procedure already explained above.

## 27.1 One day before delivery

On the laptop:

```powershell
nvidia-smi
docker version
docker image inspect gpu-demo:v1
Get-Item .\models\SmolLM2-360M-Instruct-Q4_K_M.gguf
```

Run local inference and retain the output:

```powershell
$modelDirectory = (Resolve-Path .\models).Path
docker run --rm --gpus all `
  --mount "type=bind,source=$modelDirectory,target=/models,readonly" `
  gpu-demo:v1
```

Check the cloud VM state:

```powershell
gcloud auth list
gcloud config get-value project
gcloud compute instances describe $vmName --zone=$zone `
  --format="table(name,status,machineType.basename())"
```

If the VM is stopped, start it, connect, and run:

```bash
nvidia-smi
sudo docker image inspect gpu-demo:v1
ls -lh ~/cuda-cloud-demo/models/
```

Run cloud inference once, then stop the VM after the rehearsal.

## 27.2 Files to keep open during the session

```text
1. Slides or HTML book
2. Local PowerShell terminal
3. VS Code local project
4. VS Code Remote SSH window or cloud SSH terminal
5. Local and cloud evidence output
6. Provider console billing/resource page
```

## 27.3 Demonstration sequence

### Local workload

1. Show `app.py` and identify model path, prompt, and GPU layers.
2. Show the Dockerfile and identify base, dependency, copy, and command.
3. Run `nvidia-smi` on the host.
4. Run the container with the mounted model and `--gpus all`.
5. Point to GPU-offload messages and generated output.

### Move to cloud

1. Show the GCP VM and its GPU/machine type.
2. Connect using SSH or VS Code Remote SSH.
3. Run `hostname` to prove the terminal is remote.
4. Run `nvidia-smi` to identify the cloud GPU.
5. Show the same `app.py` and Dockerfile.
6. Run the same container workload.
7. Compare local and cloud evidence.

### Explain containerization

1. Identify Dockerfile, image, and running container.
2. Explain why the driver stays on the host.
3. Explain the model bind mount.
4. Show `docker image ls`, `docker ps`, and `docker inspect` only as needed.
5. Relate recipe rebuild to registry image transfer.

### Connect to HPC

1. Show the Docker-to-SIF conversion command.
2. Show `apptainer run --nv`.
3. Show the Slurm job file.
4. Identify which responsibility belongs to Slurm, Apptainer, and CUDA.

### Finish

1. Stop the cloud VM.
2. Show its stopped status.
3. Remind students that persistent resources may still be billed.

## 27.4 Commands to avoid typing live

Do not spend live session time downloading the model, installing drivers, installing Docker, installing NVIDIA Container Toolkit, or building a large image from an empty cache. These operations are part of the manual and learning process, but their duration depends on external networks and package repositories.

## 27.5 Evidence pack

Keep this small set of files:

```text
evidence/
├── local-nvidia-smi.txt
├── local-inference.txt
├── cloud-nvidia-smi.txt
├── cloud-inference.txt
├── cloud-image-inspect.json
└── cloud-vm-description.txt
```

The evidence pack is not a fallback architecture. It is a record proving what was executed and a reference for discussion.

# Appendix A. Essential Commands

```bash
# Build an image from the Dockerfile in the current directory
docker build -t gpu-demo:v1 .

# Run the workload with access to NVIDIA GPUs
docker run --rm --gpus all gpu-demo:v1

# Save an image as a portable archive
docker save -o gpu-demo.tar gpu-demo:v1

# Load the archive into another Docker installation
docker load -i gpu-demo.tar

# Run an Apptainer image with NVIDIA GPU support
apptainer run --nv gpu-demo.sif
```

# Appendix B. Book Development Notes

This manuscript is the primary source for the book. Session notes, scripts, and post-mortems in the repository are evidence and supporting material; they should be integrated here rather than published as disconnected chapters.

From the repository root, validate and publish the manuscript with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\book\build.ps1
```

The build validates the title, required manual sections, chapter structure, and fenced code blocks, then uses Pandoc to generate:

```text
book/output/CUDA_TO_CLOUD.html
book/output/CUDA_TO_CLOUD.docx
```

Edit only `book/CUDA_TO_CLOUD.md`; generated files are outputs rather than independent manuscript sources.

# Appendix C. Glossary

**Apptainer:** A container platform designed for HPC and shared computing environments; formerly associated with the Singularity project and file format.

**Bind mount:** A mapping that exposes an existing host file or directory at a path inside a container.

**Build context:** The files made available to a Docker image build.

**Cloud GPU VM:** A provider-managed virtual machine configured with access to one or more GPUs.

**Container:** An isolated running process environment created from an image.

**CUDA:** NVIDIA's parallel-computing platform and programming model for GPU computing.

**Dockerfile:** A text recipe used to build a Docker image.

**GPU memory or VRAM:** High-bandwidth memory directly associated with a GPU.

**Image:** A packaged, read-only application filesystem and configuration used to create containers.

**Inference:** Using a trained model to produce predictions or generated output from new input.

**NVIDIA Container Toolkit:** Host-side components that connect supported container runtimes to NVIDIA GPUs.

**Object storage:** Storage addressed using objects and keys rather than ordinary local filesystem paths.

**Registry:** A service for storing and distributing container images.

**SIF:** Singularity Image Format, commonly used as a portable Apptainer image file.

**Slurm:** A workload manager used to allocate resources and schedule jobs on clusters.

**Spot or interruptible capacity:** Discounted cloud capacity that the provider may reclaim; appropriate only for workloads designed to tolerate interruption.

**Virtual machine:** An isolated computer environment using virtualized hardware and its own guest operating-system kernel.

**Workload:** The code, runtime, dependencies, data or model, hardware requirements, execution command, and output associated with a computation.
