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
