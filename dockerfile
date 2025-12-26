# 使用 NVIDIA CUDA 12.4 devel 版本作為基底，支援 Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 設定環境變數，避免互動提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 更新系統並安裝必要工具，包括 git、curl 等
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 3.12 使用 deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-venv python3.12-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# 安裝 uv (從 Notebook)
RUN pip install --upgrade uv

# 安裝所有 Python 依賴，從 Notebook 複製邏輯
# 假設新環境，無 numpy/pillow 版本，所以使用預設
RUN uv pip install -U \
    "torch>=2.8.0" "triton>=3.4.0" numpy pillow torchvision bitsandbytes "transformers==4.56.2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    "git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels" && \
    uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo

# 安裝額外依賴（從 Notebook 的 libraries 和 tools）
RUN uv pip install -U datasets huggingface_hub

# 預下載模型和資料集到映像中（離線可用）
RUN pip install huggingface-hub[hf_transfer] && \
    huggingface-cli download unsloth/gpt-oss-20b --local-dir /models/gpt-oss-20b && \
    huggingface-cli download HuggingFaceH4/Multilingual-Thinking --local-dir /datasets/Multilingual-Thinking

# (選用) 安裝 Jupyter，如果想在容器內跑 Notebook
# RUN uv pip install -U jupyterlab ipywidgets

# 設定工作目錄
WORKDIR /app

# 將 Notebook 轉成的 Python 腳本 COPY 進去（見下方說明）
COPY finetune.py /app/finetune.py

# 預設命令：進入 bash shell，讓使用者互動運行 python finetune.py
CMD ["/bin/bash"]