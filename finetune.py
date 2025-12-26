import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

# ---- 基本設定 ----
MODEL_PATH = "/models/gpt-oss-20b"
DATASET_PATH = "/datasets/Multilingual-Thinking"
OUTPUT_DIR = "/outputs"

max_seq_length = 1024
dtype = None

# ---- 載入模型（完全離線）----
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
    local_files_only=True,
)

# ---- LoRA 設定 ----
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---- Dataset（parquet，離線）----
dataset = load_dataset(
    "parquet",
    data_files=f"{DATASET_PATH}/data/*.parquet",
    split="train",
    cache_dir="/tmp/hf_cache",
)

dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ---- Trainer ----
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start|>user<|message|>",
    response_part="<|start|>assistant<|channel|>final<|message|>",
)

trainer.train()
