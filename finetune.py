import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
OUTPUT_DIR = "./output"
MAX_LEN = 512
BATCH_SIZE = 2
EPOCHS = 1
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ----------------------------
# Dummy dataset (replace later)
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.encodings["input_ids"][idx]),
        }

# ----------------------------
# Load model / tokenizer
# ----------------------------
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[INFO] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.to(DEVICE)

# ----------------------------
# Data
# ----------------------------
texts = [
    "module adder(input a, input b, output sum);",
    "always @(posedge clk) begin state <= next_state; end",
]

dataset = TextDataset(texts, tokenizer)

# ----------------------------
# Training
# ----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=(DEVICE == "cuda"),
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

print("[INFO] Start training...")
trainer.train()

print("[INFO] Training done. Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
