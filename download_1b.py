from huggingface_hub import snapshot_download

def main():
    print("Downloading Gemma 3 1B-it...")
    try:
        snapshot_download(
            repo_id="google/gemma-3-1b-it",
            local_dir="/services/model-with-adpter-logic-semantic/models/gemma-3-1b-it",
            local_dir_use_symlinks=False
        )
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
