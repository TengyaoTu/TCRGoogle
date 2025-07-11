import re
import pandas as pd
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from multiprocessing import Pool, cpu_count

from TCR_DataLoader import load_tcr_datasets, search, generate_expert_response


def extract_antigens_from_response(response_text):
    match = re.search(r"Antigen\(s\):\s*([A-Za-z0-9, ]+)", response_text)
    if match:
        antigen_str = match.group(1)
        antigens = [ag.strip() for ag in antigen_str.split(",") if ag.strip()]
        return antigens
    return []


def compute_f1(predicted, ground_truth):
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def process_single_example(args):
    row, tokenizer, model, data = args
    try:
        if 'cdr3' in row:
            cdr3_seq = row['cdr3']
        elif 'CDR3.beta.aa' in row:
            cdr3_seq = row['CDR3.beta.aa']
        elif 'TCR BioIdentity' in row:
            cdr3_seq = row['TCR BioIdentity']
        else:
            return None

        search_result_text, true_antigens = search(cdr3_seq, data)

        if search_result_text.strip() == "{None, No matched}":
            return None

        expert_response = generate_expert_response(search_result_text, tokenizer, model)
        pred_antigens = extract_antigens_from_response(expert_response)
        p, r, f1 = compute_f1(pred_antigens, true_antigens)
        return p, r, f1, cdr3_seq, true_antigens, pred_antigens
    except Exception as e:
        return None





def evaluate_llm_on_dataset(dataset_name, df, model_dir='QuantFactory/Bio-Medical-Llama-3-8B-GGUF', max_samples=100):
    from tqdm import tqdm
    print(f"\n=== Evaluating on dataset: {dataset_name} ===")
    print(f"Loading model from {model_dir}...")
    if model_dir=="QuantFactory/Bio-Medical-Llama-3-8B-GGUF":
        from llama_cpp import Llama

        # 加载 GGUF 格式模型
        llm = Llama(
            model_path="/home/u24s151013/.cache/modelscope/hub/models/QuantFactory/Bio-Medical-Llama-3-8B-GGUF/Bio-Medical-Llama-3-8B.Q8_0.gguf",
            n_ctx=4096,
            n_threads=16,
            n_gpu_layers=32  # 根据你的 A100 显存大小可调整
        )
        print("Loading TCR reference databases...")
        data = load_tcr_datasets()

        precisions, recalls, f1s = [], [], []

        test_subset = df.head(max_samples)

        for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
            if 'cdr3' in row:
                cdr3_seq = row['cdr3']
            elif 'CDR3.beta.aa' in row:
                cdr3_seq = row['CDR3.beta.aa']
            elif 'TCR BioIdentity' in row:
                cdr3_seq = row['TCR BioIdentity']
            else:
                continue

            # 调用search函数
            search_result_text, true_antigens = search(cdr3_seq, data)

            if search_result_text.strip() == "{None, No matched}":
                print(f"{cdr3_seq}: No matched antigens in DB, skipping")
                continue
                # 构造消
            else:
                messages = [
                    {"role": "system", "content": "You are a professional immunologist."},
                    {"role": "user", "content": f"""Below is the result of querying a T-cell receptor (TCR) sequence in several databases.

                    ### Search Results ###
                    {search_result_text}
                    ### End of Search Results ###

                    Please analyze the data and respond with your expert interpretation in the following format:

                    Antigen(s): <comma-separated list of antigen peptide sequences>  
                    Justification: <brief explanation referring to database match, confidence, MHC type, disease relevance, etc.>

                    Instructions:
                    - Include all plausible antigen sequences.
                    - Refer specifically to which databases matched and the confidence level.
                    - Mention any known disease associations (virus, tumor, autoimmune).
                    - You MUST respond ONLY in this EXACT format. DO NOT include any other text or greetings.
                    """}
                ]

                # 推理
                output = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.7,
                    top_p=0.95,
                )
            expert_response = output["choices"][0]["message"]["content"]
            pred_antigens = extract_antigens_from_response(expert_response)

            p, r, f1 = compute_f1(pred_antigens, true_antigens)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            print(f"\nCDR3: {cdr3_seq}")
            print(f"True Antigens: {true_antigens}")
            print(f"Predicted Antigens: {pred_antigens}")
            print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

        avg_p = sum(precisions) / len(precisions) if precisions else 0
        avg_r = sum(recalls) / len(recalls) if recalls else 0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0

        print(f"\n=== Final Metrics for {dataset_name} ===")
        print(f"Precision: {avg_p:.3f}")
        print(f"Recall:    {avg_r:.3f}")
        print(f"F1-score:  {avg_f1:.3f}")

        return avg_p, avg_r, avg_f1

    model_path = snapshot_download(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

    print("Loading TCR reference databases...")
    data = load_tcr_datasets()

    precisions, recalls, f1s = [], [], []

    test_subset = df.head(max_samples)

    for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        if 'cdr3' in row:
            cdr3_seq = row['cdr3']
        elif 'CDR3.beta.aa' in row:
            cdr3_seq = row['CDR3.beta.aa']
        elif 'TCR BioIdentity' in row:
            cdr3_seq = row['TCR BioIdentity']
        else:
            continue

        # 调用search函数
        search_result_text, true_antigens = search(cdr3_seq, data)

        if search_result_text.strip() == "{None, No matched}":
            print(f"{cdr3_seq}: No matched antigens in DB, skipping")
            continue

        expert_response = generate_expert_response(search_result_text, tokenizer, model)
        pred_antigens = extract_antigens_from_response(expert_response)

        p, r, f1 = compute_f1(pred_antigens, true_antigens)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        print(f"\nCDR3: {cdr3_seq}")
        print(f"True Antigens: {true_antigens}")
        print(f"Predicted Antigens: {pred_antigens}")
        print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0

    print(f"\n=== Final Metrics for {dataset_name} ===")
    print(f"Precision: {avg_p:.3f}")
    print(f"Recall:    {avg_r:.3f}")
    print(f"F1-score:  {avg_f1:.3f}")

    return avg_p, avg_r, avg_f1
def run_all_dataset_evaluations():
    datasets = load_tcr_datasets()

    # 评估 VDJdb
    vdjdb = datasets['vdjdb']
    evaluate_llm_on_dataset("VDJdb", vdjdb, max_samples=100)

    # 评估 McPAS（只用beta链）
    mcpas = datasets['mcpas']
    mcpas_beta = mcpas[mcpas['CDR3.beta.aa'].notna()]
    evaluate_llm_on_dataset("McPAS (beta)", mcpas_beta, max_samples=100)

    # 评估 ImmuneCODE (CI)
    peptide_ci = datasets['peptide_ci']
    evaluate_llm_on_dataset("ImmuneCODE CI", peptide_ci, max_samples=100)
