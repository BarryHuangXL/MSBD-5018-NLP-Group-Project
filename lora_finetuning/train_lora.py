import os
import json
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import transformers
print("✅ Transformers version:", transformers.__version__, transformers.__file__)
# ===== 配置 =====
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_FILE = "financial_sentiment_instructions.jsonl"
OUTPUT_DIR = "./qwen3_1p7b_lora_output"
MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 3
# ================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- 数据加载 --------
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data

raw_data = load_jsonl(DATA_FILE)
print(f"✅ Loaded {len(raw_data)} samples")

pairs = [
    {"text": f"Instruction: {item['instruction']}\nResponse: {item['output']}"}
    for item in raw_data
]

dataset = HFDataset.from_list(pairs)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# -------- 模型和tokenizer --------
print("🚀 Loading model & tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# -------- 配置 LoRA --------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------- Tokenize --------
def tokenize_function(examples):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
eval_ds = eval_ds.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -------- 训练参数 --------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    optim="adamw_torch",
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# -------- 开始训练 --------
trainer.train()
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

print("🎯 LoRA adapter saved at", os.path.join(OUTPUT_DIR, "lora_adapter"))

# -------- 推理测试 --------
def infer(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    print("\n🧾 Prompt:\n", prompt)
    print("\n💬 Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

test_prompt = "Instruction: Classify the sentiment of the sentence 'The company reported record profits this quarter.'\nResponse:"
infer(test_prompt)
