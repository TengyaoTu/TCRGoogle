import pandas as pd
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
        print(f"vdjdb loaded. ")
        vdjdb_filtered = vdjdb[vdjdb['vdjdb.score'] != '0']
        print(vdjdb_filtered.iloc[0].to_dict())

        print(f"vdjdb filtered.")
    except Exception as e:
        vdjdb_filtered = None
        print(f"Failed to load vdjdb.txt: {e}")

    # 2. McPAS-TCR 数据集
    try:
        mcpas = pd.read_csv("Database/McPAS-TCR.csv", encoding='latin1', dtype=str)
        print(f"McPAS-TCR loaded.")
    except Exception as e:
        mcpas = None
        print(f"Failed to load McPAS-TCR.csv: {e}")

    # 3. ImmuneCODE - Class I
    try:
        peptide_ci = pd.read_csv("Database/peptide-detail-ci.csv", dtype=str)
        print(f"peptide-detail-ci loaded.")
        peptide_ci[['TCR BioIdentity', 'Additional Information']] = peptide_ci['TCR BioIdentity'].apply(split_bioidentity)

        print(f"peptide-detail-ci processed.")
    except Exception as e:
        peptide_ci = None
        print(f"Failed to load peptide-detail-ci.csv: {e}")

    # 4. ImmuneCODE - Class II
    try:
        peptide_cii = pd.read_csv("Database/peptide-detail-cii.csv", dtype=str)
        print(f"peptide-detail-cii loaded.")
        peptide_cii[['TCR BioIdentity', 'Additional Information']] = peptide_cii['TCR BioIdentity'].apply(split_bioidentity)
        print(f"peptide-detail-cii processed.")
    except Exception as e:
        peptide_cii = None
        print(f"Failed to load peptide-detail-cii.csv: {e}")

    return {
        "vdjdb": vdjdb_filtered,
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

    # === Output ===
    if found_sources:
        antigens = list(set([ag for ag in antigens if isinstance(ag, str) and ag.strip() != '']))
        antigen_str = ", ".join(antigens) if antigens else "Unknown"
        header = f"Search successful. The antigen(s) of this TCR may be: {antigen_str}. Matches found in the following datasets: {', '.join(found_sources)}\n\n"
        print(header)
        return header + matched_info
    else:
        msg = "{None, No matched}"
        return msg

def generate_expert_response(search_result_text, tokenizer, model):
    """
    Generate a structured expert explanation based on TCR search results.
    """
    prompt = f"""You are a professional immunologist. The following content contains the search results of a T-cell receptor (TCR) sequence from multiple immunological databases:

{search_result_text}

Please analyze the search results and summarize your interpretation. Your response **must follow this strict format**:

{{Predicted Antigen Sequence(s)}}, {{Reason for prediction based on the databases}}

Make sure your explanation includes:
- What antigen(s) the TCR likely recognizes (e.g., virus, tumor, autoimmune epitope);
- Which databases support this, and how reliable they are;
- Any available MHC information;
- Potential immunological or clinical relevance.

Respond clearly and professionally. Do **not** write anything outside the specified format.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1024
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = response.replace(prompt.strip(), "").strip()
    return final_response


def run_tcr_query(sequence, model_dir='Qwen/Qwen1.5-1.8B-Chat'):
    print(f"Loading model from: {model_dir} ...")
    model_path = snapshot_download(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

    print("Loading TCR databases...")
    data = load_tcr_datasets()

    print(f"\nSearching for sequence: {sequence}")
    search_result = search(sequence, data)

    if search_result.strip() == "{None, No matched}":
        print("\n=== Expert Response ===")
        print("{None, No matched}")
        return "{None, No matched}"

    print("\nGenerating expert interpretation...")
    expert_response = generate_expert_response(search_result, tokenizer, model)

    print("\n=== Expert Response ===")
    print(expert_response)
    return expert_response
