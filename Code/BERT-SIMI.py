#%%
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import numpy as np

# === Step 1: 读取数据 ===
vdjdb_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered.csv"
trusted_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\protgpt2_finetune_data.csv"
vdjdb_df = pd.read_csv(vdjdb_path)
trusted_df = pd.read_csv(trusted_path)

# 提取需要分类的未知 TCR 序列（score = 0）
cdr3_unknown = vdjdb_df[vdjdb_df["vdjdb.score"] == 0]["cdr3"].dropna().astype(str).tolist()
score0_idx = vdjdb_df[vdjdb_df["vdjdb.score"] == 0].index

# 从可信数据中抽取 20000 条序列
trusted_seqs = trusted_df["Sequence"].dropna().astype(str).tolist()
trusted_cdr3 = [s.split("XXX")[0] for s in trusted_seqs]
random.seed(42)
trusted_sample = random.sample(trusted_cdr3, 20000)

# === Step 2: 加载 TCR-BERT 模型 ===
model_name = "wukevin/tcr-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

def encode(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu()

# === Step 3: 提取嵌入 ===
print("🚀 提取 20000 条可信 TCR 嵌入...")
trusted_vecs = [encode(s) for s in tqdm(trusted_sample)]
trusted_tensor = torch.stack(trusted_vecs)

print("🔍 提取 337 条未知 TCR 嵌入...")
unknown_vecs = [encode(s) for s in tqdm(cdr3_unknown)]
unknown_tensor = torch.stack(unknown_vecs)

# === Step 4: 计算最大匹配相似度 ===
print("📐 计算最大匹配余弦相似度...")
sim_matrix = cosine_similarity(unknown_tensor, trusted_tensor)
match_sim = sim_matrix.max(axis=1)  # ✅ 你要的最大值策略！

# === Step 5: 伪标签划分（按 match_sim 排序对半切） ===
sorted_idx = np.argsort(match_sim)
half = len(match_sim) // 2
pseudo = np.ones(len(match_sim), dtype=int)
pseudo[sorted_idx[:half]] = 0

# 插入结果
vdjdb_df["pseudo_class"] = 2  # 默认可信
vdjdb_df.loc[score0_idx, "trust_score"] = match_sim
vdjdb_df.loc[score0_idx, "pseudo_class"] = pseudo

# 输出阈值
threshold_value = np.sort(match_sim)[half]
print(f"📏 最大匹配相似度划分阈值为：{threshold_value:.4f}")

# === Step 6: 保存输出 ===
output_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered_scored_20k_maxmatch.csv"
vdjdb_df.to_csv(output_path, index=False)
print(f"✅ 已保存分类结果到：{output_path}")
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import numpy as np

# === Step 1: 读取数据 ===
vdjdb_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered.csv"
trusted_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\protgpt2_finetune_data.csv"
vdjdb_df = pd.read_csv(vdjdb_path)
trusted_df = pd.read_csv(trusted_path)

# 提取 score=0 的 TCR 序列
cdr3_unknown = vdjdb_df[vdjdb_df["vdjdb.score"] == 0]["cdr3"].dropna().astype(str).tolist()
score0_idx = vdjdb_df[vdjdb_df["vdjdb.score"] == 0].index

# 抽取 20000 条可信样本（ProtGPT2）
trusted_seqs = trusted_df["Sequence"].dropna().astype(str).tolist()
trusted_cdr3 = [s.split("XXX")[0] for s in trusted_seqs]
random.seed(42)
trusted_sample = random.sample(trusted_cdr3, 20000)

# === Step 2: 加载模型 ===
model_name = "wukevin/tcr-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

def encode(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu()

# === Step 3: 提取嵌入 ===
print("🚀 提取 20000 条可信 TCR 嵌入...")
trusted_vecs = [encode(s) for s in tqdm(trusted_sample)]
trusted_tensor = torch.stack(trusted_vecs)

print("🔍 提取 337 条未知 TCR 嵌入...")
unknown_vecs = [encode(s) for s in tqdm(cdr3_unknown)]
unknown_tensor = torch.stack(unknown_vecs)

# === Step 4: 计算最大匹配相似度 ===
print("📐 计算最大余弦相似度...")
sim_matrix = cosine_similarity(unknown_tensor, trusted_tensor)
max_sim = sim_matrix.max(axis=1)

# 插入打分（不分类）
vdjdb_df.loc[score0_idx, "trust_score"] = max_sim

# 打印最高、最低、平均值供你参考
print(f"✅ 最大匹配相似度统计：")
print(f"🔹 max: {np.max(max_sim):.4f}")
print(f"🔹 min: {np.min(max_sim):.4f}")
print(f"🔹 mean: {np.mean(max_sim):.4f}")
print(f"🔹 median: {np.median(max_sim):.4f}")

# === Step 5: 保存输出 ===
output_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered_maxscore_only.csv"
vdjdb_df.to_csv(output_path, index=False)
print(f"✅ 已保存最大匹配结果到：{output_path}")
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import numpy as np

# === Step 1: 读取数据 ===
vdjdb_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered.csv"
trusted_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\protgpt2_finetune_data.csv"
vdjdb_df = pd.read_csv(vdjdb_path)
trusted_df = pd.read_csv(trusted_path)

# 提取 score=0 的 TCR 序列
cdr3_unknown = vdjdb_df[vdjdb_df["vdjdb.score"] == 0]["cdr3"].dropna().astype(str).tolist()
score0_idx = vdjdb_df[vdjdb_df["vdjdb.score"] == 0].index

# 抽取 20000 条可信样本（ProtGPT2）
trusted_seqs = trusted_df["Sequence"].dropna().astype(str).tolist()
trusted_cdr3 = [s.split("XXX")[0] for s in trusted_seqs]
random.seed(42)
trusted_sample = random.sample(trusted_cdr3, 20000)

# === Step 2: 加载模型 ===
model_name = "wukevin/tcr-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

def encode(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu()

# === Step 3: 提取嵌入 ===
print("🚀 提取 20000 条可信 TCR 嵌入...")
trusted_vecs = [encode(s) for s in tqdm(trusted_sample)]
trusted_tensor = torch.stack(trusted_vecs)

print("🔍 提取 337 条未知 TCR 嵌入...")
unknown_vecs = [encode(s) for s in tqdm(cdr3_unknown)]
unknown_tensor = torch.stack(unknown_vecs)

# === Step 4: 计算最大匹配相似度 ===
print("📐 计算最大余弦相似度...")
sim_matrix = cosine_similarity(unknown_tensor, trusted_tensor)
max_sim = sim_matrix.max(axis=1)

# 插入打分（不分类）
vdjdb_df.loc[score0_idx, "trust_score"] = max_sim

# 打印最高、最低、平均值供你参考
print(f"✅ 最大匹配相似度统计：")
print(f"🔹 max: {np.max(max_sim):.4f}")
print(f"🔹 min: {np.min(max_sim):.4f}")
print(f"🔹 mean: {np.mean(max_sim):.4f}")
print(f"🔹 median: {np.median(max_sim):.4f}")

# === Step 5: 保存输出 ===
output_path = r"C:\Users\数学迷迷迷\Desktop\蛋白质\vdjdb_filtered_maxscore_only.csv"
vdjdb_df.to_csv(output_path, index=False)
print(f"✅ 已保存最大匹配结果到：{output_path}")
