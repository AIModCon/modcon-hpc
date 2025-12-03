import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import BitsAndBytesConfig


# -------------------------
# 1. Load dataset
# -------------------------

DATA_PATH = "./InstructionResponsePairs/amrex_dataset.jsonl"

print(f"Loading dataset manually: {DATA_PATH}")

train_data = []
with open(DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()        # remove leading/trailing whitespace
        if not line:               # skip empty lines
            continue
        obj = json.loads(line)
        train_data.append(obj)

print("Loaded examples:", len(train_data))
print("First example:", train_data[0])
print("Second example:", train_data[1])

train_data = [json.load(open("amrex_dataset.jsonl"))]  # or your loaded list

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)

print("Converted to hugginface data")

# -------------------------
# 2. Load tokenizer & model
# -------------------------
BASE_MODEL = "/pscratch/sd/n/nataraj2/mistral-7b"

# 8-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,   # optional, can tune if needed
)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Set pad token to eos token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (8-bit to fit on GPU)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# -------------------------
# 3. Configure LoRA
# -------------------------
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -------------------------
# 4. Tokenization function
# -------------------------
def build_prompt(instruction, response):
    return f"Instruction:\n{instruction}\n\nResponse:\n{response}"

def tokenize(example):
    prompt = build_prompt(example["instruction"], example["response"])
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = train_dataset.map(tokenize, batched=False)

# -------------------------
# 5. Training configuration
# -------------------------
training_args = TrainingArguments(
    output_dir="./amrex-lora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,           # small starting run, adjust later
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False
)

# -------------------------
# 6. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -------------------------
# 7. Train
# -------------------------
print("Starting training...")
trainer.train()

# -------------------------
# 8. Save LoRA outputs
# -------------------------
SAVE_DIR = "./amrex-lora"

print(f"Saving LoRA adapter to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Done.")

