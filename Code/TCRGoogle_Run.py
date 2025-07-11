from TCRGoogleTest import run_all_dataset_evaluations
from TCR_DataLoader import load_tcr_datasets, search, run_tcr_query


def run():
    # 加载数据
    ##示例Demo数据，方便读者使用
    ##vdjdb中的数据：CASSIVGGNEQFF
    ##mcpas的数据：CASSLGNEQF
    ##peptide的数据：CASSAQGTGDRGYTF
    ##'Qwen/Qwen1.5-1.8B-Chat'
    ##'AI-ModelScope/TinyLlama-1.1B-Chat-v1.0'
    ##'Qwen/Qwen1.5-0.5B-Chat'
    ##‘Qwen/Qwen1.5-4B-Chat’
    ##"Qwen/Qwen1.5-MoE-A2.7B-Chat"
    ##'Hunyuan-A13B-Instruct'
    ##'Qwen/QwQ-32B'
    ##deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    ##deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    ##'QuantFactory/Bio-Medical-Llama-3-8B-GGUF'
    run_tcr_query("CASSIVGGNEQFF",
                  model_dir='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                  )

if __name__ == "__main__":
    #data = load_tcr_datasets()
    #vdjdb_df = data['vdjdb']
    # 只用有antigen标签且不为空的样本
    #vdjdb_df = vdjdb_df[vdjdb_df['antigen.epitope'].notna() & (vdjdb_df['antigen.epitope'] != '')]
    # 运行测试
    #wtest_llm_on_vdjdb(vdjdb_df, max_samples=20)

    #run_all_dataset_evaluations()
    run()