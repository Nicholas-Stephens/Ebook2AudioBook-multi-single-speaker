#!/usr/bin/env python3
"""
Download IndexTTS models from Hugging Face
"""
import os
import sys
from pathlib import Path

def download_models(checkpoint_dir="./index-tts/checkpoints"):
    """Download required IndexTTS models from Hugging Face"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Downloading IndexTTS models to {checkpoint_dir}...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface-hub")
        from huggingface_hub import hf_hub_download
    
    # Model repository - using your custom model
    repo_id = "Kriest/IndexTTS2"
    
    # Required files - ALL files needed for IndexTTS2 to work
    required_files = [
        "bpe.model",
        "config.yaml",
        "feat1.pt",
        "feat2.pt",
        "gpt.pth",
        "pinyin.vocab",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "qwen0.6bemo4-merge/model.safetensors",
        "qwen0.6bemo4-merge/tokenizer.json",
        "qwen0.6bemo4-merge/config.json",
        "qwen0.6bemo4-merge/generation_config.json",
        "qwen0.6bemo4-merge/tokenizer_config.json",
        "qwen0.6bemo4-merge/special_tokens_map.json",
        "qwen0.6bemo4-merge/added_tokens.json",
        "qwen0.6bemo4-merge/chat_template.jinja",
        "qwen0.6bemo4-merge/vocab.json",
        "qwen0.6bemo4-merge/merges.txt"
    ]
    
    failed_files = []
    
    for filename in required_files:
        file_path = checkpoint_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        try:
            print(f"⬇️  Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            failed_files.append(filename)
    
    if failed_files:
        print(f"\n⚠️  Failed to download: {', '.join(failed_files)}")
        print(f"You may need to download these manually from: https://huggingface.co/{repo_id}")
        return False
    else:
        print(f"\n✅ All models downloaded successfully!")
        return True

if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "./index-tts/checkpoints"
    success = download_models(checkpoint_dir)
    sys.exit(0 if success else 1)
