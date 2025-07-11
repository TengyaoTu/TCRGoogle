import pandas as pd
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

from ProtGPT_Generating import generate_protein_sequence
from mlx_lm import load, generate
def load_tcr_datasets():
    # 辅助函数：拆分 TCR BioIdentity 字段
    def split_bioidentity(bioidentity):
        if isinstance(bioidentity, str):
            parts = bioidentity.split('+', 1)
            cdr3 = parts[0].strip()
            additional = parts[1].strip() if len(parts) > 1 else ''
            return pd.Series([cdr3, additional])
        else:
            return pd.Series(['', ''])

    # 1. VDJdb 数据集
    try:
        vdjdb = pd.read_csv("Database/vdjdb.txt", sep='\t', dtype=str)

        vdjdb_filtered = vdjdb[vdjdb['vdjdb.score'] != '0']



    except Exception as e:
        vdjdb_filtered = None
        print(f"Failed to load vdjdb.txt: {e}")

    # 2. McPAS-TCR 数据集
    try:
        mcpas = pd.read_csv("Database/McPAS-TCR.csv", encoding='latin1', dtype=str)
        mcpas = mcpas[mcpas["Epitope.peptide"].notna() & (mcpas["Epitope.peptide"] != "")]


    except Exception as e:
        mcpas = None
        print(f"Failed to load McPAS-TCR.csv: {e}")

    # 3. ImmuneCODE - Class I
    try:
        peptide_ci = pd.read_csv("Database/peptide-detail-ci.csv", dtype=str)

        peptide_ci[['TCR BioIdentity', 'Additional Information']] = peptide_ci['TCR BioIdentity'].apply(split_bioidentity)
        # 仅保留 Amino Acids 不为空的行
        ppeptide_ci = peptide_ci[peptide_ci['Amino Acids'].notna() & (peptide_ci['Amino Acids'].str.strip() != '')]

        peptide_ci = peptide_ci.assign(
            Amino_Acids=peptide_ci['Amino Acids'].str.split(',')
        ).explode('Amino_Acids')

        peptide_ci['Amino Acids'] = peptide_ci['Amino_Acids'].str.strip()
        peptide_ci = peptide_ci.drop(columns=['Amino_Acids'])

    except Exception as e:
        peptide_ci = None
        print(f"Failed to load peptide-detail-ci.csv: {e}")

    # 4. ImmuneCODE - Class II
    try:
        peptide_cii = pd.read_csv("Database/peptide-detail-cii.csv", dtype=str)
        peptide_cii[['TCR BioIdentity', 'Additional Information']] = peptide_cii['TCR BioIdentity'].apply(split_bioidentity)
        # 仅保留 Amino Acids 不为空的行
        peptide_cii = peptide_cii[peptide_cii['Amino Acids'].notna() & (peptide_cii['Amino Acids'].str.strip() != '')]

        peptide_cii = peptide_cii.assign(
            Amino_Acids=peptide_cii['Amino Acids'].str.split(',')
        ).explode('Amino_Acids')

        peptide_cii['Amino Acids'] = peptide_cii['Amino_Acids'].str.strip()
        peptide_cii = peptide_cii.drop(columns=['Amino_Acids'])


    except Exception as e:
        peptide_cii = None
        print(f"Failed to load peptide-detail-cii.csv: {e}")

    return {
        "vdjdb": vdjdb,
        "mcpas": mcpas,
        "peptide_ci": peptide_ci,
        "peptide_cii": peptide_cii
    }
def search(sequence, data):
    found_sources = []
    matched_info = ""
    antigens = []
    # === VDJdb ===
    vdjdb_df = data['vdjdb']
    vdjdb_result = vdjdb_df[vdjdb_df['cdr3'] == sequence]
    if not vdjdb_result.empty:
        found_sources.append("vdjdb")
        antigen_values = vdjdb_result['antigen.epitope'].dropna().unique().tolist()
        antigens.extend(antigen_values)
        matched_info += "[vdjdb matched result]:\n" + vdjdb_result.to_string(index=False) + "\n\n"

    # === McPAS ===
    mcpas_df = data['mcpas']
    matched_rows_alpha = mcpas_df[mcpas_df['CDR3.alpha.aa'] == sequence]
    matched_rows_beta = mcpas_df[mcpas_df['CDR3.beta.aa'] == sequence]
    if not matched_rows_alpha.empty:
        found_sources.append("mcpas.alpha")
        antigen_values = matched_rows_alpha['Epitope.peptide'].dropna().unique().tolist()
        antigens.extend(antigen_values)
        matched_info += "[McPAS CDR3.alpha.aa matched result]:\n" + matched_rows_alpha.to_string(index=False) + "\n\n"
    if not matched_rows_beta.empty:
        found_sources.append("mcpas.beta")
        antigen_values = matched_rows_beta['Epitope.peptide'].dropna().unique().tolist()
        antigens.extend(antigen_values)
        matched_info += "[McPAS CDR3.beta.aa matched result]:\n" + matched_rows_beta.to_string(index=False) + "\n\n"

    # === Peptide CI ===
    peptide_ci_df = data['peptide_ci']
    peptide_ci_result = peptide_ci_df[peptide_ci_df['TCR BioIdentity'] == sequence]
    if not peptide_ci_result.empty:
        found_sources.append("peptide_ci")
        antigen_values = peptide_ci_result['Amino Acids'].dropna().unique().tolist()
        antigens.extend(antigen_values)
        matched_info += "[peptide-detail-ci matched result]:\n" + peptide_ci_result.to_string(index=False) + "\n\n"

    # === Peptide CII ===
    peptide_cii_df = data['peptide_cii']
    peptide_cii_result = peptide_cii_df[peptide_cii_df['TCR BioIdentity'] == sequence]
    if not peptide_cii_result.empty:
        found_sources.append("peptide_cii")
        antigen_values = peptide_cii_result['Amino Acids'].dropna().unique().tolist()
        antigens.extend(antigen_values)
        matched_info += "[peptide-detail-cii matched result]:\n" + peptide_cii_result.to_string(index=False) + "\n\n"

    if found_sources:
        antigens = list(set([ag for ag in antigens if isinstance(ag, str) and ag.strip() != '']))
        antigen_str = ", ".join(antigens) if antigens else "Unknown"
        header = f"the antigen(s) identified are: {antigen_str}. Matches found in the datasets: {', '.join(found_sources)}\n\n"
        print(header)
        if len(matched_info) > 100:
            matched_info = matched_info[:100] + "...(truncated)\n"
        # 返回元组：展示文本 + 真实antigen列表（用于测试对比）
        #return header , antigens
        return header + matched_info, antigens
    else:
        msg = "{None, No matched}"
        return msg, []


def generate_expert_response(search_result_text, tokenizer, model):
    """
    Generate a structured expert explanation based on TCR search results using chat-style prompt.
    """
    # 1. 构造消息
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

    # 2. 应用对话模板生成模型输入文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. 编码 + 上设备
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. 推理生成（推荐设置终止符号）
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
            temperature=0.7
        )

    # 5. 去除输入部分，仅保留生成的 token
    generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]

    # 6. 解码
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    gc.collect()  # 触发 Python 垃圾回收
    torch.cuda.empty_cache()  # 清空 PyTorch 缓存显存
    return response





def run_tcr_query(sequence, model_dir):
    print(f"Loading model from: {model_dir} ...")
    model_path = snapshot_download(model_dir)

    if model_dir=="QuantFactory/Bio-Medical-Llama-3-8B-GGUF":
        from llama_cpp import Llama

        # 加载 GGUF 格式模型
        llm = Llama(
            model_path="/home/u24s151013/.cache/modelscope/hub/models/QuantFactory/Bio-Medical-Llama-3-8B-GGUF/Bio-Medical-Llama-3-8B.Q8_0.gguf",
            n_ctx=4096,
            n_threads=16,
            n_gpu_layers=32  # 根据你的 A100 显存大小可调整
        )
        print("Loading TCR databases...")
        data = load_tcr_datasets()

        print(f"\nSearching for sequence: {sequence}")
        search_result, true_antigens = search(sequence, data)

        if search_result.strip() == "{None, No matched}":
            print("\n=== Expert Response ===")
            print("{None, No matched}")
            return "{None, No matched}"

        print("\nGenerating expert interpretation...")
        # 构造消息
        messages = [
            {"role": "system", "content": "You are a professional immunologist."},
            {"role": "user", "content": f"""Below is the result of querying a T-cell receptor (TCR) sequence in several databases.

        ### Search Results ###
        {search_result}
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
        print(output["choices"][0]["message"]["content"])
        return output["choices"][0]["message"]["content"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
    print("Model is on:", model.device)

    print("Loading TCR databases...")
    data = load_tcr_datasets()

    print(f"\nSearching for sequence: {sequence}")
    search_result, true_antigens  = search(sequence, data)

    if search_result.strip() == "{None, No matched}":
        expert_response=generate_protein_sequence(sequence)
        print("ProtGPT2 Generate:"+expert_response)

        return expert_response

    print("\nGenerating expert interpretation...")
    expert_response = generate_expert_response(search_result, tokenizer, model)

    print("\n=== Expert Response ===")
    print(expert_response)
    return expert_response
