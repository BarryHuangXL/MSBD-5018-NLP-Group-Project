import os
import json
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import transformers

print("✅ Transformers version:", transformers.__version__)

# ===== 基础配置 =====
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_FILE = "train_final.jsonl"
BASE_OUTPUT_DIR = "./qwen3_1p7b_lora_multi_output"
MAX_LENGTH = 256
BATCH_SIZE = 5
GRAD_ACCUM = 8
LR = 5e-5
EPOCHS = 5
PATIENCE = 2  # 早5停轮次
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

# ===== 数据加载 =====
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

raw_data = load_jsonl(DATA_FILE)
print(f"✅ Loaded {len(raw_data)} samples")

# ===== 构造训练文本格式 =====
# 适配你的三字段格式 {"system": ..., "user": ..., "assistant": ...}
pairs = []
for item in raw_data:
    system = item.get("system", "")
    user = item.get("user", "")
    assistant = item.get("assistant", "")
    # 拼接为单条输入输出文本
    text = (
        f"<system>{system}</system>\n"
        f"<user>{user}</user>\n"
        f"<assistant>{assistant}</assistant>"
    )
    pairs.append({"text": text})

dataset = HFDataset.from_list(pairs)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# ===== Tokenizer & 模型加载 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# ===== LoRA 方案（可切换）=====
LORA_SCHEMES = {
    "A_minimal": {
        "target_modules": ["q_proj", "v_proj"],
        "r": 4,
        "alpha": 8,
        "dropout": 0.05
    },
    "B_standard": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "r": 8,
        "alpha": 16,
        "dropout": 0.05
    },
    "C_extended": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        "r": 16,
        "alpha": 32,
        "dropout": 0.1
    },
    "D_full_transform": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "r": 32,
        "alpha": 64,
        "dropout": 0.1
    },
    "E_regularized_small": {
        "target_modules": ["q_proj", "v_proj", "up_proj"],
        "r": 4,
        "alpha": 16,
        "dropout": 0.2
    }
}

# ===== 选择方案 =====
scheme_name = "A_minimal"
cfg = LORA_SCHEMES[scheme_name]
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, scheme_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n🧩 Using LoRA scheme: {scheme_name}")
print(f"📂 Output directory: {OUTPUT_DIR}\n")

# ===== LoRA配置 =====
lora_config = LoraConfig(
    r=cfg["r"],
    lora_alpha=cfg["alpha"],
    lora_dropout=cfg["dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=cfg["target_modules"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== Tokenize =====
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

# ===== 训练参数 =====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    fp16=True,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",
    weight_decay=WEIGHT_DECAY,  # ✅ 正则化
    max_grad_norm=1.0,          # ✅ 梯度裁剪
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    resume_from_checkpoint=True,  # 关键参数：自动从最新checkpoint恢复
)

# ===== Trainer + 早停机制 =====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

# ===== 开始训练 =====
trainer.train()

# ===== 保存模型 =====
adapter_dir = os.path.join(OUTPUT_DIR, f"lora_adapter_{scheme_name}")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"🎯 LoRA adapter saved at: {adapter_dir}")

# ===== 推理测试 =====
def infer(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print("\n🧾 Prompt:\n", prompt)
    print("\n💬 Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 示例测试
test_prompt = (
    "<system>You are a financial sentiment analysis expert. "
    "Your task is to analyze the sentiment expressed in the given text. "
    "Only reply with positive, neutral, or negative.</system>\n"
    "<user>The company reported record profits this quarter.</user>\n"
    "<assistant>"
)
infer(test_prompt)

