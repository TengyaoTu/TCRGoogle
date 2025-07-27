import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import random

# ===== 加载模型和数据（只执行一次） =====
model_name = "wukevin/tcr-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

vdjdb_path = r"vdjdb_filtered.csv"
trusted_path = r"protgpt2_finetune_data.csv"

vdjdb_df = pd.read_csv(vdjdb_path)
trusted_df = pd.read_csv(trusted_path)

# 抽取 20000 条可信样本
trusted_seqs = trusted_df["Sequence"].dropna().astype(str).tolist()
trusted_cdr3 = [s.split("XXX")[0] for s in trusted_seqs]
random.seed(42)
trusted_sample = random.sample(trusted_cdr3, 20000)

def encode(seq: str) -> torch.Tensor:
    """编码一个TCR序列"""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu()

# 预计算数据库向量
trusted_vecs = [encode(s) for s in trusted_sample]
trusted_tensor = torch.stack(trusted_vecs)

def find_most_similar_tcr(query_seq: str) -> dict:
    """
    输入：待查询的TCR序列
    输出：与之最相似的数据库TCR记录和相似度
    """
    query_vec = encode(query_seq).unsqueeze(0)
    sims = cosine_similarity(query_vec, trusted_tensor)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    best_tcr = trusted_sample[best_idx]
    
    return {
        "input": query_seq,
        "matched_tcr": best_tcr,
        "cosine_similarity": round(float(best_score), 4)
    }

