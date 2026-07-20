# Moving to Cloud — Session Notes & Reference

## Cloud Provider Equivalents

| Requirement         | AWS                  | Azure                | Google Cloud          |
|---------------------|----------------------|----------------------|-----------------------|
| GPU Virtual Machine | EC2 (p3/p4d/g5)      | NC/ND-series VMs     | Compute Engine + GPU  |
| Container Registry  | ECR                  | ACR                  | Artifact Registry     |
| Object Storage      | S3                   | Blob Storage         | Cloud Storage (GCS)   |
| Managed Kubernetes  | EKS                  | AKS                  | GKE                   |
| Batch / HPC Jobs    | AWS Batch            | Azure Batch          | Google Cloud Batch    |
| Serverless GPU      | SageMaker            | Azure ML             | Vertex AI             |
| SSH Key Management  | EC2 Key Pairs        | SSH Keys in Portal   | OS Login / SSH Keys   |

---

## Migration Workflow (Talk Through This)

```
Your Local Machine
     │
     ├── Python script  ─────────────────────────────────────┐
     ├── Model weights                                        │
     ├── Dataset                                              │
     └── Environment (conda/pip)                             │
                                                             ↓
                                              Upload to Cloud Storage
                                              (S3 / Blob / GCS)
                                                             │
                                                             ↓
                                         Provision GPU VM on Cloud
                                         (EC2 p3 / Azure NC / GCP A100)
                                                             │
                                                             ↓
                                              SSH into VM
                                              Install dependencies
                                              Pull data from storage
                                                             │
                                                             ↓
                                              Run workload
                                              Monitor logs
                                                             │
                                                             ↓
                                         Save results back to storage
                                         Terminate VM (stop billing!)
```

---

## Cost Control Tips (mention these)

- Always **stop/terminate** VM after use
- Use **Spot (AWS) / Preemptible (GCP) / Spot (Azure)** for 60-90% savings
- Set **budget alerts** in each cloud console
- Use **reserved instances** for predictable workloads
- Avoid storing large data on VM disk — use cloud storage

---

## Security Basics (1 slide)

- Never hardcode API keys or passwords in code
- Use **IAM roles** to grant permissions
- Restrict SSH access to your IP only (Security Group / Firewall rule)
- Use **SSH key pairs** — never password auth on a VM
- Enable **MFA** on cloud accounts

---

## Screenshots to Prepare

Save these to: `screenshots/`

1. `aws-ec2-launch.png`    — EC2 instance type filter showing GPU instances
2. `azure-vm-gpu.png`      — Azure VM creation with NC-series selected
3. `gcp-compute-gpu.png`   — GCP Compute Engine accelerator dropdown
4. `ssh-vscode.png`        — VS Code Remote SSH connected to a VM
5. `cloud-storage.png`     — S3 or GCS bucket with uploaded files

---

## Offline Backup Checklist

- [ ] All slides exported as PDF
- [ ] Screenshots saved locally
- [ ] Cloud recording saved as .mp4
- [ ] Demo files in containerization/ folder
- [ ] USB drive copy of everything
- [ ] This file printed or open on second monitor
