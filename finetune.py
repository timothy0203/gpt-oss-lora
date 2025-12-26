import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

# 載入模型 (從本地路徑，離線)
max_seq_length = 1024
dtype = None
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/models/gpt-oss-20b",  # 修改為本地路徑
    dtype=dtype,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)

# 添加 LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 載入資料集 (從本地路徑，離線)
dataset = load_dataset(
    "/datasets/Multilingual-Thinking",  # 修改為本地路徑
    data_files={"train": "data/train-00000-of-00001.parquet"},
    split="train"
)

# 格式化資料集
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Trainer 設定
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,  # 調整為您的需求
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# 只訓練回應部分
gpt_oss_kwargs = dict(instruction_part="<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>final<|message|>")
trainer = train_on_responses_only(trainer, **gpt_oss_kwargs)

# 開始訓練
trainer.train()