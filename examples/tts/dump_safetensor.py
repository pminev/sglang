from safetensors.torch import safe_open
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="nari-labs/Dia-1.6B", filename="model.safetensors")
print(model_path)

with safe_open(model_path, framework="pt") as f:
    print("Keys:")
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")