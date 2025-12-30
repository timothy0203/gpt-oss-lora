#!/usr/bin/env python3
"""
下載 GPT-OSS-20B 模型權重
"""
from huggingface_hub import snapshot_download
import os

# 模型列表
models = [
    "unsloth/gpt-oss-20b",
    # 如果需要，可以添加其他模型
    # "unsloth/gpt-oss-120b",
]

def download_models():
    """下載所有模型"""
    for model_id in models:
        print(f"\n下載模型: {model_id}")
        
        # 提取模型名稱作為目錄名
        model_name = model_id.split("/")[-1]
        local_dir = f"model_weights/{model_name}"
        
        try:
            # 使用 snapshot_download 下載
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.txt",
                    "*.py",
                    "tokenizer*",
                    "*.md"
                ]
            )
            print(f"✓ 模型下載完成: {local_dir}")
        except Exception as e:
            print(f"✗ 下載失敗 {model_id}: {e}")

if __name__ == "__main__":
    os.makedirs("model_weights", exist_ok=True)
    download_models()