import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id = "Qwen/Qwen-Image",
    local_dir = "models/Qwen/Qwen-Image"
)

snapshot_download(
    repo_id = "DFloat11/Qwen-Image-DF11",
    local_dir = "models/DFloat11/Qwen-Image-DF11"
)

