from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 ProtGPT2 模型和分词器
model_name = "nferruz/ProtGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
##ProtGPT2 是一个专门针对蛋白质序列训练的 GPT-2 模型。
##---------------------------------------------------
##它经过大量训练，可以很好的输出合理的氨基酸序列。
# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 输入起始 token（可以理解为“蛋白质起始序列”）
input_text = "CASSIRSSYEQYF"  #
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 生成蛋白质序列
output_ids = model.generate(
    input_ids,
    #max_length=len(input_text)+50,   # 最大序列长度
    max_new_tokens=len(input_text),     # 明确：生成50个新的token
    num_return_sequences=1,     # 生成几条
    do_sample=True,             # 使用随机采样
    top_k=950,                 # 限制采样空间
    temperature=1.0             # 控制多样性
)

# 解码输出
generated_seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("🧬 生成的蛋白质序列：")
print(generated_seq)
print(len(input_text))
print(len(generated_seq))
##封装生成器，可以在本地复用
def generate_protein_sequence(seed_sequence: str, model_path_or_name: str, max_new_tokens: int = 50) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import re

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer 和模型（支持本地路径）
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    # 编码输入序列
    input_ids = tokenizer(seed_sequence, return_tensors="pt", padding=True).input_ids.to(device)

    # 生成新序列
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,  # 显式指定
        do_sample=True,
        top_k=950,
        temperature=1.0
    )

    # 解码输出并清洗非氨基酸字符
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned_seq = ''.join(c for c in generated_text if c in "ACDEFGHIKLMNPQRSTVWY")

    return cleaned_seq