import os
import pandas as pd
import numpy as np
import argparse
import warnings
import torch
from torch.utils.data import Dataset

# 删除了 from utils import get_numbering
warnings.filterwarnings("ignore")

AMINO_ACIDS = "RHKDESTNQCUGPAVILMFYW"  # 21 amino acid
PAD = "-"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"
AA_List = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS


class TEPDataset(Dataset):
    def __init__(self, sequence, labels, max_tcr_len=20, max_antigen_len=11, align=False):
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.data = self._initilize_data(sequence, labels, align)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _initilize_data(self, sequences, labels, align):
        data_list = []

        # ✅ 移除了 get_numbering 调用
        # 如果 align=True，也不再尝试对齐，直接用原始序列
        # 保证不会因 ANARCI 缺失报错
        if align:
            print("Warning: alignment skipped due to platform limitation. Using raw TCR sequences.")

        sequences = sequences.values
        labels = labels.values.astype('int')

        for index, sequence in enumerate(sequences):
            tcr, pep = sequence
            tcr_tsfed = self.seq_transform(tcr, self.max_tcr_len)
            pep_tsfed = self.seq_transform(pep, self.max_antigen_len)
            data_list.append((tcr_tsfed, pep_tsfed, labels[index], tcr, pep))

        return data_list

    def seq_transform(self, sequence, max_len):
        # Sequence padding
        sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + '-' * (max_len - len(sequence))
        try:
            data_list = [AA_List.index(aa) for aa in sequence]
        except:
            print(f"Invalid AA in seq: {sequence}")
        return torch.tensor(data_list)


def dropInvalid(df_x, df_y):
    df_x.drop(df_x[df_x['TCR'].str.contains('-')].index, inplace=True)
    df_y.drop(df_x[df_x['TCR'].str.contains('-')].index, inplace=True)
    return df_x, df_y
