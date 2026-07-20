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
