import torch
import torch.nn.functional as F
from TEPCAM import TEPCAM

# 氨基酸到索引的映射，21 表示 PAD / 未知字符
AA_TO_INDEX = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17,
    "W": 18, "Y": 19, "X": 20, "*": 21, "-": 21
}

def aa_sequence_to_tensor(seq: str, max_len: int) -> torch.Tensor:
    """将氨基酸序列转换为Tensor（带PAD填充）"""
    seq = seq.upper()
    indices = [AA_TO_INDEX.get(aa, 21) for aa in seq]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [21] * (max_len - len(indices))  # PAD填充
    return torch.LongTensor(indices)

def load_model(weights_path: str, device: str = "cpu"):
    """加载TEPCAM模型及权重"""
    model = TEPCAM(d_model=32, modelseed=23, n_heads=6)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_tcr_epitope(tcr_seq: str, epitope_seq: str, model, device="cpu"):
    """对单个TCR-Epitope对进行推理预测"""
    tcr_tensor = aa_sequence_to_tensor(tcr_seq, max_len=20).unsqueeze(0).to(device)
    epitope_tensor = aa_sequence_to_tensor(epitope_seq, max_len=11).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, _, _, logits = model(tcr_tensor, epitope_tensor)
        probs = F.softmax(logits, dim=-1).squeeze()  # shape: [2]
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        pred_class = torch.argmax(probs).item()
        return {
            "tcr": tcr_seq,
            "epitope": epitope_seq,
            "probability": probs.tolist(),
            "prediction": int(pred_class)
        }

# 示例运行
if __name__ == "__main__":
    model_path = "ckpts/tepcam_test.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)

    tcr = "CASSLGQGAETLYF"
    epitope = "NLVPMVATV"
    result = predict_tcr_epitope(tcr, epitope, model, device=device)

    print("Prediction Result:")
    print(result)
