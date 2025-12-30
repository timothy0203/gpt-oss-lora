#!/usr/bin/env python3
"""
下載所有需要的 wheel 文件供離線安裝
"""
import subprocess
import os
import sys

# 創建 wheels 目錄
wheels_dir = "offline_wheels"
os.makedirs(wheels_dir, exist_ok=True)

# 包列表（根據您的需求調整）
packages = [
    "torch==2.8.0",
    "torchvision",
    "torchaudio",
    "transformers==4.56.2",
    "tokenizers==0.21.0",
    "datasets==3.2.0",
    "accelerate==1.3.1",
    "bitsandbytes==0.43.5",
    "peft==0.13.2",
    "trl==0.22.2",
    "scipy==1.13.1",
    "sentencepiece==0.2.0",
    "protobuf==5.29.0",
    "jinja2==3.1.4",
    "markupsafe==2.1.5",
    "numpy==1.26.4",
    "pillow==10.4.0",
    "triton==3.4.0",
    # NVIDIA CUDA 相關包
    "nvidia-cudnn-cu12==8.9.7.29",
    "nvidia-cublas-cu12==12.4.5.117",
    "nvidia-cufft-cu12==11.2.1.19",
    "nvidia-curand-cu12==10.3.5.119",
    "nvidia-cusolver-cu12==11.6.1.18",
    "nvidia-cusparse-cu12==12.3.1.19",
    "nvidia-nccl-cu12==2.20.5",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cudnn-frontend-cu12==0.2.2",
]

def download_wheels():
    """下載所有 wheel 文件"""
    print(f"下載 wheel 文件到 {wheels_dir}/")
    
    for package in packages:
        print(f"\n下載: {package}")
        try:
            # 使用 pip download
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--platform", "manylinux2014_x86_64",
                "--python-version", "39",
                "--implementation", "cp",
                "--abi", "cp39",
                "-d", wheels_dir,
                package
            ]
            
            # 對於 PyTorch，需要指定 CUDA 版本
            if "torch" in package:
                cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu124"])
            
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"警告: 下載 {package} 失敗: {e}")
            # 嘗試不使用平台限制
            try:
                cmd = [
                    sys.executable, "-m", "pip", "download",
                    "-d", wheels_dir,
                    package
                ]
                if "torch" in package:
                    cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu124"])
                subprocess.run(cmd, check=True)
            except:
                print(f"嚴重: 無法下載 {package}")

if __name__ == "__main__":
    download_wheels()
    print("\n" + "="*50)
    print(f"所有 wheel 文件已下載到: {os.path.abspath(wheels_dir)}")
    print("="*50)