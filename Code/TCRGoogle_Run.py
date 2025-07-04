from TCR_DataLoader import load_tcr_datasets, search, run_tcr_query


def run():
    # 加载数据
    ##示例Demo数据，方便读者使用
    ##vdjdb中的数据：CASSIVGGNEQFF
    ##mcpas的数据：CASSAQGTGDRGYTF
    ##peptide-detail-ci的数据：CASSAQGTGDRGYTF
    run_tcr_query("CASSIVGGNEQFF", model_dir='Qwen/Qwen1.5-1.8B-Chat')

if __name__ == "__main__":
    run()