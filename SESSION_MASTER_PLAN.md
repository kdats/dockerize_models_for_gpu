# GPU Cloud and HPC Containers — 3-Day Master Runbook

This file is the single source of truth for preparing the 3–4 hour college session. Update the checkboxes and progress log after every working block.

## Fixed constraints

- Delivery date: 25 July 2026
- Programme: CUDA to Cloud — Accelerating AI through Supercomputing and Advanced Computing
- Audience: Hundreds of college students
- Infrastructure: Minimal
- Delivery must survive weak or unavailable internet
- Teaching style: One continuous GPU workload and the minimum commands needed to move it
- Session 1: Moving workloads from local GPUs to AWS, Azure, or Google Cloud
- Session 2: Virtualization and containerization with Docker and Apptainer/Singularity for HPC
- Cloud coverage: Demonstrate one provider; compare AWS, Azure, and Google Cloud conceptually
- Container coverage: Package the same GPU workload with Docker, then show how Apptainer/Singularity runs it in HPC

## Central learning path

```text
Existing CUDA/GPU workload
        -> package its environment in Docker
        -> run it with access to a local GPU
        -> move the workload/image to a cloud GPU VM
        -> run the same workload with Apptainer/Singularity on HPC
```

Every example must support this path. Generic container, cloud, web, orchestration, or CPU examples are only temporary foundations and must not become the focus.

## Decisions to record before setup

- [ ] Presentation laptop OS: `TBD`
- [ ] Laptop RAM and free disk space: `TBD`
- [ ] NVIDIA GPU and model, if present: `TBD`
- [ ] Chosen cloud provider: `TBD`
- [ ] Cloud account and billing/credits available: `TBD`
- [ ] Venue internet availability: `TBD`
- [ ] Access to a real HPC cluster: `TBD`

If there is no NVIDIA GPU, cloud credit, or HPC access, preparation can still continue. The session will use CPU-compatible execution, recorded cloud material, and a local Apptainer demonstration.

## Definition of done

The session is ready only when all of these are true:

- [ ] The sample Python application runs locally.
- [ ] The application runs inside Docker.
- [ ] The Docker image can be saved and loaded without internet.
- [ ] The application runs with Apptainer.
- [ ] GPU detection is demonstrated live or through recorded evidence.
- [ ] The cloud VM workflow is demonstrated live or through a recording.
- [ ] Slides are exported to PDF.
- [ ] Every live demonstration has screenshots or a video fallback.
- [ ] All session assets exist on the laptop and a second storage device.
- [ ] A complete timed rehearsal finishes within 3–4 hours.

## Folder layout

```text
sessionpresssidency/
├── SESSION_MASTER_PLAN.md
├── movingtocloud/
│   ├── README.md
│   ├── screenshots/
│   ├── recordings/
│   └── cloud-notes/
├── containerization/
│   ├── README.md
│   ├── app/
│   ├── docker/
│   ├── apptainer/
│   └── offline-images/
└── delivery/
    ├── slides/
    ├── cheat-sheets/
    ├── backup-output/
    └── rehearsal-notes/
```

Folders and files will be created only when their phase begins, keeping the workspace understandable.

## Three-day schedule

### Day 1 — Build the working demonstrations

Target: 8–10 focused hours.

#### Phase 1: Inspect and prepare the laptop — 45–75 minutes

- [ ] Record system specifications.
- [ ] Confirm virtualization support.
- [ ] Confirm free disk space of at least 25 GB.
- [ ] Install or verify WSL2 if using Windows.
- [ ] Install or verify Ubuntu under WSL2.
- [ ] Install or verify VS Code.
- [ ] Install the required VS Code extensions.
- [ ] Install or verify Git.
- [ ] Create the final workspace structure.
- [ ] Save verification output.

Checkpoint: Linux shell, VS Code, and the workspace all open successfully.

#### Phase 2: Create the common Python workload — 60–90 minutes

- [ ] Create a small Python application.
- [ ] Print Python and platform information.
- [ ] Detect CUDA/GPU availability safely.
- [ ] Run a CPU matrix computation and record its duration.
- [ ] Run a GPU computation when CUDA is available.
- [ ] Add a small requirements file.
- [ ] Run it locally and save expected output.
- [ ] Prepare a deliberately broken dependency example if time permits.

Checkpoint: One command runs the workload locally and produces predictable output.

#### Phase 3: Docker demonstration — 2–2.5 hours

- [ ] Install or verify Docker.
- [ ] Verify Docker with a small cached test image.
- [ ] Write a beginner-friendly Dockerfile.
- [ ] Add a `.dockerignore` file.
- [ ] Build the workload image.
- [ ] Run the workload in a container.
- [ ] Demonstrate image versus container.
- [ ] Demonstrate bind mounts for persistent output.
- [ ] Demonstrate logs and basic inspection.
- [ ] If NVIDIA GPU exists, configure and test GPU passthrough.
- [ ] Create a multi-stage or optimized Dockerfile as an optional comparison.
- [ ] Save the final image to a `.tar` archive.
- [ ] Verify the offline image archive.

Checkpoint: The application runs in Docker without downloading anything during delivery.

#### Phase 4: Apptainer demonstration — 2–2.5 hours

- [ ] Install Apptainer inside Linux/WSL2.
- [ ] Verify its version.
- [ ] Create an Apptainer definition file.
- [ ] Build or obtain the `.sif` image.
- [ ] Run and execute commands in the image.
- [ ] Demonstrate bind-mounted data.
- [ ] If NVIDIA GPU exists, test the `--nv` path.
- [ ] Explain the Docker-to-Apptainer image workflow through commands.
- [ ] Prepare an example Slurm job file without requiring a real cluster.
- [ ] Save expected output and screenshots.

Checkpoint: The same application runs through Apptainer, or a tested recording exists if the local platform cannot support it.

#### Phase 5: End-of-day stabilization — 45 minutes

- [ ] Re-run the local, Docker, and Apptainer flows from clean terminals.
- [ ] Write down every command in presentation order.
- [ ] Mark any unreliable step as recording-only.
- [ ] Copy important outputs into the backup folder.
- [ ] Update the progress log below.

### Day 2 — Cloud workflow and teaching material

Target: 8–10 focused hours.

#### Phase 6: Cloud safety setup — 45–60 minutes

- [ ] Confirm the chosen provider and region.
- [ ] Confirm GPU quota before attempting provisioning.
- [ ] Configure a strict budget alert.
- [ ] Decide the maximum permitted spend.
- [ ] Prepare a resource naming convention.
- [ ] Prepare a deletion/cleanup checklist.
- [ ] Avoid placing permanent credentials in slides, code, or recordings.

Checkpoint: Billing controls exist before a chargeable GPU VM is started.

#### Phase 7: Cloud VM demonstration — 2–3 hours

- [ ] Create SSH keys safely.
- [ ] Provision the smallest suitable VM; use CPU first if GPU quota is unavailable.
- [ ] Configure firewall rules for SSH only from the required source.
- [ ] Connect from a terminal.
- [ ] Connect using VS Code Remote SSH.
- [ ] Transfer or clone the sample workload.
- [ ] Create the remote Python environment.
- [ ] Run the workload remotely.
- [ ] If using a GPU VM, verify the GPU and driver.
- [ ] Capture basic CPU/GPU monitoring output.
- [ ] Stop the instance and explain stop versus delete.
- [ ] Delete temporary resources after recording unless needed for rehearsal.
- [ ] Verify that no chargeable orphaned resources remain.

Checkpoint: A short, edited recording shows provisioning, connection, execution, monitoring, and shutdown.

#### Phase 8: Cloud provider comparison — 45–60 minutes

- [ ] Map VM services across AWS, Azure, and Google Cloud.
- [ ] Map object storage services.
- [ ] Map container registries.
- [ ] Map batch/HPC services.
- [ ] Prepare one cost-management slide.
- [ ] Prepare one IAM/firewall security slide.
- [ ] Clearly label any prices as examples that may change.

#### Phase 9: Build the slides — 3–4 hours

- [ ] Opening and learning outcomes: 3–4 slides.
- [ ] Local-to-cloud concepts: 6–8 slides.
- [ ] Cloud workflow and VS Code: 5–7 slides.
- [ ] Docker concepts: 6–8 slides.
- [ ] Apptainer/HPC concepts: 5–7 slides.
- [ ] Docker versus Apptainer comparison: 2–3 slides.
- [ ] Cost, security, and best practices: 4–5 slides.
- [ ] End-to-end workflow and recap: 2–3 slides.
- [ ] Q&A prompts: 1 slide.
- [ ] Export slides to PDF.

Target: Roughly 35–45 slides. Prefer diagrams, commands, and observed output over dense paragraphs.

#### Phase 10: Build offline fallbacks — 60–90 minutes

- [ ] Save the cloud demonstration video locally.
- [ ] Save numbered screenshots for every cloud step.
- [ ] Save Docker images locally.
- [ ] Save the Apptainer `.sif` file locally.
- [ ] Cache or locally store all required Python packages where practical.
- [ ] Save terminal output for every demo.
- [ ] Prepare a PDF command cheat sheet.
- [ ] Test all videos without internet.

### Day 3 — Rehearse and deliver

Target: 3–4 preparation hours before delivery.

#### Phase 11: Technical rehearsal — 90 minutes

- [ ] Disconnect from the internet.
- [ ] Restart the presentation laptop.
- [ ] Open slides, terminals, VS Code, and videos.
- [ ] Run the local application.
- [ ] Run the cached Docker demonstration.
- [ ] Run the Apptainer demonstration.
- [ ] Play the cloud recording.
- [ ] Confirm fonts, projector resolution, audio, and video playback.
- [ ] Record failures and replace unreliable steps with fallbacks.

#### Phase 12: Timed content rehearsal — 2–3 hours

- [ ] Rehearse Part 1 within its allotted time.
- [ ] Rehearse Part 2 within its allotted time.
- [ ] Remove optional details when over time.
- [ ] Mark questions that should be deferred to Q&A.
- [ ] Practice transitioning between slides, terminal, VS Code, and video.

#### Phase 13: Final duplication — 20–30 minutes

- [ ] Copy the complete delivery folder to a USB drive or second device.
- [ ] Confirm the copied files open.
- [ ] Carry the slides as both editable format and PDF.
- [ ] Carry the command sheet as PDF and plain text.
- [ ] Fully charge the laptop and carry power/display adapters.

## Final 3–4 hour delivery flow

### Part 1: Moving local GPU workloads to cloud — 80–95 minutes

1. Motivation and outcomes — 10 minutes
2. Infrastructure, providers, lift-and-shift, and refactoring — 20 minutes
3. Recorded provisioning plus live Remote SSH/workload walkthrough — 35–45 minutes
4. Cost, security, and questions — 15–20 minutes

### Break — 10 minutes

### Part 2: Docker and Apptainer for HPC — 85–100 minutes

1. Reproducibility problem and container overview — 10 minutes
2. Images, containers, registries, and security models — 20 minutes
3. Docker and Apptainer demonstration using the same workload — 35–45 minutes
4. Volumes, image optimization, GPU access, Slurm, and questions — 20–25 minutes

### Buffer — 15–25 minutes

Use the buffer for setup delays, audience interaction, or extended Q&A. Do not fill it with mandatory content.

## Demo reliability rules

- Never depend on provisioning a GPU VM during the live session.
- Never depend on downloading a large CUDA image at the venue.
- Do not expose API keys, private SSH keys, account IDs, or billing details.
- Use one known-good command sequence rather than improvising.
- Keep terminals at a readable font size.
- Place commands in a text file for copy/paste.
- Keep CPU fallback paths working even when GPU paths are shown.
- Stop or delete cloud resources immediately after use.

## Scope guardrails

Required:

- One provider demonstrated
- AWS/Azure/Google Cloud service comparison
- SSH and VS Code Remote SSH
- One workload moved from local to remote
- Docker build and run
- Apptainer build/run or Docker-image conversion
- GPU container flags explained
- Volumes/bind mounts
- Cost, IAM, firewall, and cleanup basics
- Slurm integration shown at command-file level

Optional only if required work is complete:

- Kubernetes
- Managed batch services
- Terraform
- Multi-node training
- Complex networking
- Building a full CUDA base image live
- Provider-specific certification-level detail

## Progress log

Add a short entry after each work block.

| Date/time | Phase | Completed | Problem/blocker | Next action |
|---|---|---|---|---|
| TBD | Planning | Master runbook created | Hardware and cloud details unknown | Record the setup decisions |

## Parking lot for questions

Record questions here without interrupting the current setup phase.

- None yet.
