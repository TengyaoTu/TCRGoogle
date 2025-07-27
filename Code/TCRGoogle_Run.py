import argparse
from TCRGoogleTest import run_all_dataset_evaluations
from TCR_DataLoader import load_tcr_datasets, search, run_tcr_query

def run(cdr3, model_dir):
    run_tcr_query(cdr3, model_dir=model_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TCRGoogle query with specified CDR3 and model.")
    parser.add_argument("--cdr3", type=str, required=True,
                        help="The CDR3 sequence to query (e.g., CASSIVGGNEQFF)")
    parser.add_argument("--model", type=str, required=True,
                        help="The model directory or HuggingFace model name (e.g., deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)")
    args = parser.parse_args()
    run(cdr3=args.cdr3, model_dir=args.model)
