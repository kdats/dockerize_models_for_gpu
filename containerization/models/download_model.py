from pathlib import Path

from huggingface_hub import hf_hub_download

model_directory = Path(__file__).resolve().parent

path = hf_hub_download(
    repo_id="bartowski/SmolLM2-360M-Instruct-GGUF",
    filename="SmolLM2-360M-Instruct-Q4_K_M.gguf",
    local_dir=model_directory,
)

print(f"Saved model to: {path}")
