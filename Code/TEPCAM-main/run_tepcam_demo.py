import sys
import os
import importlib.util

scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
sys.path.append(scripts_dir)

# 强制加载 scripts/test.py
test_path = os.path.join(scripts_dir, "test.py")
spec = importlib.util.spec_from_file_location("tepcam_test", test_path)
tepcam_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tepcam_test)

# 现在就能用 test_model 了
test_model = tepcam_test.test_model



# ✅ 配置文件路径（你自己修改对应）
FILE_PATH = "scripts/your_test"  # 测试数据（需含 TCR, epitope, Label）
MODEL_PATH = "ckpts/tepcam_test.pt"  # 模型文件
OUTPUT_FILE = "output/demo_output.csv"
METRIC_FILE = "output/demo_metrics.txt"

BATCH_SIZE = 128
GPU_NUM = 0
ALIGN = True

# 检查路径
assert os.path.exists(FILE_PATH), f"❌ 测试数据文件不存在：{FILE_PATH}"
assert os.path.exists(MODEL_PATH), f"❌ 模型文件不存在：{MODEL_PATH}"

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(METRIC_FILE), exist_ok=True)

# ✅ 调用测试函数
test_model(
    file_path=FILE_PATH,
    model_path=MODEL_PATH,
    output_file=OUTPUT_FILE,
    metric_file=METRIC_FILE,
    batch_size=BATCH_SIZE,
    GPU_num=GPU_NUM,
    align=ALIGN
)

print(f"\n✅ 推理完成！预测结果保存于: {OUTPUT_FILE}")
print(f"📊 指标保存于: {METRIC_FILE}")
import pandas as pd
import uuid

def run_tepcam(tcr1: str, tcr2: str) -> float:
    """比较两个TCR的TEP-CAM相似度分数"""

    # === 1. 构造唯一测试文件 ===
    tmp_id = uuid.uuid4().hex[:8]
    tmp_input_path = f"scripts/tmp_tepcam_input_{tmp_id}.csv"
    tmp_output_path = f"output/tmp_tepcam_output_{tmp_id}.csv"
    tmp_metric_path = f"output/tmp_tepcam_metric_{tmp_id}.txt"

    df = pd.DataFrame([{
        "TCR": tcr1,
        "epitope": tcr2,
        "Label": 1  # dummy label，模型需要这一列
    }])
    df.to_csv(tmp_input_path, index=False)

    # === 2. 调用模型 ===
    test_model(
        file_path=tmp_input_path,
        model_path=MODEL_PATH,
        output_file=tmp_output_path,
        metric_file=tmp_metric_path,
        batch_size=BATCH_SIZE,
        GPU_num=GPU_NUM,
        align=ALIGN
    )

    # === 3. 提取预测分数 ===
    if os.path.exists(tmp_output_path):
        df_out = pd.read_csv(tmp_output_path)
        if 'predict_proba' in df_out.columns:
            score = float(df_out['predict_proba'].iloc[0])
        else:
            raise RuntimeError("⚠️ 缺少 predict_proba 列，TEP-CAM输出格式可能不正确")
    else:
        raise RuntimeError("❌ TEP-CAM输出文件未生成")

    # === 4. 清理临时文件（可选）===
    try:
        os.remove(tmp_input_path)
        os.remove(tmp_output_path)
        os.remove(tmp_metric_path)
    except:
        pass

    return score
