FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

WORKDIR /app

# 系統工具（最小）
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 複製 wheels 與 requirements
COPY wheels/ /wheels/
COPY docker/requirements.txt /app/requirements.txt

# 離線安裝 Python 套件
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# 複製 training script
COPY scripts/finetune.py /app/finetune.py

CMD ["bash"]
