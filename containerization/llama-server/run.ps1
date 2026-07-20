<#
.SYNOPSIS
FALLBACK DEMO — Use this if the primary Python GPU demo has issues.
Requirements: Docker Desktop with WSL2 backend or NVIDIA Container Toolkit on the host.
#>

$MODEL_DIR = Join-Path (Get-Location) "..\models"
$MODEL_FILE = Join-Path $MODEL_DIR "SmolLM2-360M-Instruct-Q4_K_M.gguf"
$MODEL_URL = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf?download=true"

# Step 1: Download model if not already present
if (!(Test-Path $MODEL_DIR)) {
    New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null
}

if (!(Test-Path $MODEL_FILE)) {
    Write-Host "Downloading SmolLM2 360M model (~220MB)..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $MODEL_URL -OutFile $MODEL_FILE
}

# Step 2: Start the llama.cpp inference server (GPU)
Write-Host "🚀 Starting llama.cpp server with GPU..." -ForegroundColor Green
docker run -d `
    --name llama-server `
    --gpus all `
    -p 8080:8080 `
    -v "$($MODEL_DIR.Replace('\', '/')):/models" `
    ghcr.io/ggml-org/llama.cpp:server-cuda `
    -m /models/SmolLM2-360M-Instruct-Q4_K_M.gguf `
    --host 0.0.0.0 `
    --port 8080 `
    --n-gpu-layers 999 | Out-Null

# Step 3: Wait for server to be ready
Write-Host "Waiting for server to start (15 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Step 4: Send a prompt and show the response
Write-Host "`n=== 🤖 Querying the model via API ===" -ForegroundColor Cyan
$body = @{
    prompt = "<|im_start|>user`nExplain why GPU computing matters in HPC in one short sentence.<|im_end|>`n<|im_start|>assistant`n"
    max_tokens = 100
    stop = @("<|im_end|>")
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8080/v1/completions" `
    -Method Post `
    -Headers @{"Content-Type"="application/json"} `
    -Body $body

Write-Host "🤖 AI: $($response.choices[0].text.Trim())" -ForegroundColor Green

# Cleanup
Write-Host "`n=== ✅ Done. Stopping server... ===" -ForegroundColor Yellow
docker stop llama-server | Out-Null
docker rm llama-server | Out-Null
