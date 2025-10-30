
import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import transformers
from datetime import datetime

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
PATIENCE = 2  # 早停轮次
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

# ===== 创建训练记录目录 =====
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAINING_LOG_DIR = os.path.join(BASE_OUTPUT_DIR, f"training_logs_{TIMESTAMP}")
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)

# ===== 设置日志记录 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(TRAINING_LOG_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== 训练记录类 =====
class TrainingRecorder:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'step': []
        }
        self.training_config = {}
        self.start_time = None
        
    def record_config(self, config_dict):
        """记录训练配置"""
        self.training_config = config_dict
        with open(os.path.join(self.log_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def record_metrics(self, epoch, step, train_loss, eval_loss, lr):
        """记录训练指标"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['step'].append(step)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['eval_loss'].append(eval_loss)
        self.metrics_history['learning_rate'].append(lr)
        
        # 实时保存到CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(os.path.join(self.log_dir, "training_metrics.csv"), index=False)
    
    def create_plots(self):
        """创建训练过程可视化图表"""
        if len(self.metrics_history['epoch']) == 0:
            return
            
        df = pd.DataFrame(self.metrics_history)
        
        # 创建损失曲线图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.plot(df['epoch'], df['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(df['step'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss by Step')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss by Epoch')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(df['step'], df['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的指标报告
        metrics_summary = {
            'final_train_loss': df['train_loss'].iloc[-1] if len(df) > 0 else None,
            'final_eval_loss': df['eval_loss'].iloc[-1] if len(df) > 0 else None,
            'min_train_loss': df['train_loss'].min(),
            'min_eval_loss': df['eval_loss'].min(),
            'total_epochs': df['epoch'].max() + 1,
            'total_steps': df['step'].max() + 1
        }
        
        with open(os.path.join(self.log_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

# ===== 初始化训练记录器 =====
recorder = TrainingRecorder(TRAINING_LOG_DIR)

# ===== 数据加载 =====
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

raw_data = load_jsonl(DATA_FILE)
logger.info(f"✅ Loaded {len(raw_data)} samples")

# ===== 构造训练文本格式 =====
pairs = []
for item in raw_data:
    system = item.get("system", "")
    user = item.get("user", "")
    assistant = item.get("assistant", "")
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

# ===== LoRA 方案 =====
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

logger.info(f"🧩 Using LoRA scheme: {scheme_name}")
logger.info(f"📂 Output directory: {OUTPUT_DIR}")

# ===== 记录训练配置 =====
training_config = {
    "model_name": MODEL_NAME,
    "lora_scheme": scheme_name,
    "lora_config": cfg,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRAD_ACCUM,
    "learning_rate": LR,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "dataset_size": len(raw_data),
    "train_size": len(train_ds),
    "eval_size": len(eval_ds),
    "timestamp": TIMESTAMP
}
recorder.record_config(training_config)

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

# ===== 自定义Callback用于记录指标 =====
class MetricsCallback(transformers.TrainerCallback):
    def __init__(self, recorder):
        self.recorder = recorder
        self.current_epoch = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每个log步骤记录指标"""
        if logs is not None and 'loss' in logs and 'epoch' in logs:
            train_loss = logs.get('loss', None)
            eval_loss = logs.get('eval_loss', None)
            learning_rate = logs.get('learning_rate', 0)
            
            if train_loss is not None:
                self.recorder.record_metrics(
                    epoch=logs['epoch'],
                    step=state.global_step,
                    train_loss=train_loss,
                    eval_loss=eval_loss,
                    lr=learning_rate
                )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """在每个epoch结束时记录"""
        self.current_epoch += 1
        logger.info(f"📊 Epoch {self.current_epoch} completed")
        
        # 每2个epoch创建一次图表
        if self.current_epoch % 2 == 0:
            self.recorder.create_plots()

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
    save_total_limit=1,
    optim="adamw_torch",
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    resume_from_checkpoint=True,
)

# ===== Trainer + 回调函数 =====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=PATIENCE),
        MetricsCallback(recorder)
    ]
)

# ===== 开始训练 =====
logger.info("🚀 Starting training...")
recorder.start_time = datetime.now()

try:
    train_result = trainer.train()
    logger.info("✅ Training completed successfully!")
    
    # 记录最终训练结果
    final_metrics = train_result.metrics
    with open(os.path.join(TRAINING_LOG_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
        
except Exception as e:
    logger.error(f"❌ Training failed: {str(e)}")
    # 即使训练失败也保存当前进度
    recorder.create_plots()

# ===== 保存最终图表和总结 =====
recorder.create_plots()

# 生成训练总结
training_duration = datetime.now() - recorder.start_time
training_summary = {
    "training_duration_seconds": training_duration.total_seconds(),
    "training_duration_human": str(training_duration),
    "final_status": "completed" if 'train_result' in locals() else "failed",
    "log_directory": TRAINING_LOG_DIR,
    "model_output_directory": OUTPUT_DIR
}

with open(os.path.join(TRAINING_LOG_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
    json.dump(training_summary, f, indent=2)

logger.info(f"📋 Training logs saved to: {TRAINING_LOG_DIR}")

# ===== 保存模型 =====
adapter_dir = os.path.join(OUTPUT_DIR, f"lora_adapter_{scheme_name}")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
logger.info(f"🎯 LoRA adapter saved at: {adapter_dir}")

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
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 记录推理结果
    inference_log = {
        "prompt": prompt,
        "output": result,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(TRAINING_LOG_DIR, "inference_test.json"), "w", encoding="utf-8") as f:
        json.dump(inference_log, f, indent=2, ensure_ascii=False)
    
    return result

# 示例测试
test_prompt = (
    "<system>You are a financial sentiment analysis expert. "
    "Your task is to analyze the sentiment expressed in the given text. "
    "Only reply with positive, neutral, or negative.</system>\n"
    "<user>The company reported record profits this quarter.</user>\n"
    "<assistant>"
)

logger.info("🧪 Running inference test...")
output = infer(test_prompt)

print("\n" + "="*60)
print("🧾 Prompt:")
print(test_prompt)
print("\n💬 Output:")
print(output)
print("="*60)

logger.info("🎉 All processes completed!")
logger.info(f"📁 Training logs: {TRAINING_LOG_DIR}")
logger.info(f"📁 Model output: {adapter_dir}")

