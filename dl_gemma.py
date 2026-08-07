from huggingface_hub import snapshot_download
snapshot_download(repo_id='google/gemma-3-1b-it', local_dir='models/gemma-3-1b-it', local_dir_use_symlinks=False)
