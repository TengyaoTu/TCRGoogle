from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
import warnings
warnings.filterwarnings("ignore")  # 屏蔽所有Python警告

# 屏蔽transformers特定日志
from transformers.utils import logging
logging.set_verbosity_error()
# 设置模型
model_name = "/home/u24s151013/ProtGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

# 使用 BitsAndBytesConfig 替代 load_in_8bit
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型并准备LoRA微调
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# 配置LoRA参数
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # 根据模型结构调整
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# 加载数据（假设已清洗并格式化）
with open("/home/u24s151013/Database/protgpt2_finetune_data.txt") as f:
    lines = f.read().splitlines()

dataset = Dataset.from_dict({"text": lines})

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding='max_length',
        truncation=True,
        max_length=68,
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# 训练所需
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from torch.optim import AdamW

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset.remove_columns(["text"]), batch_size=32, shuffle=True, collate_fn=data_collator)
optimizer = AdamW(model.parameters(), lr=2e-4)

model.train()
for epoch in range(3):##表示训练集会被使用3次，看所有的数据看三次
    for step, batch in enumerate(dataloader):##表示每次从 dataloader 中取出一批样本（batch）训练一次。
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

# 保存LoRA权重
model.save_pretrained("/home/u24s151013/ProtGPT2-LoRA")
tokenizer.save_pretrained("/home/u24s151013/ProtGPT2-LoRA")