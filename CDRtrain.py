import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import time
from datetime import timedelta


# ============ 数据加载器 ============
class DNADataset(Dataset):
    def __init__(self, file_path):
        self.data = load_dna_sequences(file_path)
        self.data_tensor = torch.from_numpy(self.data).unsqueeze(1)  # 增加通道维度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_tensor[idx]  # 返回Tensor


def load_dna_sequences(file_path):
    bit_sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip()
            if sequence:
                bit_sequence = [int(bit) for bit in sequence]
                bit_sequences.append(bit_sequence)
    return np.array(bit_sequences, dtype=np.float32)


# ============ 模型定义 ============
class CUDAOptimizedDebiasCNN(nn.Module):
    def __init__(self, seq_len: int = 1000, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 编码器:特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            EfficientResidualBlock(64, groups=1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            EfficientResidualBlock(128, groups=1),
        )

        # 中间层:膨胀卷积扩大感受野
        self.middle = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            EfficientResidualBlock(128, groups=2),
        )

        # 解码器:输出生成
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            EfficientResidualBlock(64, groups=1),
            nn.Conv1d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            feat = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
            feat = torch.utils.checkpoint.checkpoint(self.middle, feat, use_reentrant=False)
            out = torch.utils.checkpoint.checkpoint(self.decoder, feat, use_reentrant=False)
        else:
            feat = self.encoder(x)
            feat = self.middle(feat)
            out = self.decoder(feat)
        return out.squeeze(1)  # [batch, seq_len]


class EfficientResidualBlock(nn.Module):
    def __init__(self, channels, groups=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out





# ============ 推理函数 ============
@torch.no_grad()
def save_generated_sequences(model, data_loader, device, output_path="1-1check.txt", sample_with_bernoulli=True):
    """推理函数(带时间统计)"""
    inference_start_time = time.time()
    
    model.eval()
    all_sequences = []

    print("\n开始推理...")
    for batch_idx, bitstream in enumerate(data_loader):
        bitstream = bitstream.to(device, non_blocking=True)
        probs = model(bitstream)
        if sample_with_bernoulli:
            samples = torch.bernoulli(probs).cpu().numpy().astype(int)
        else:
            samples = (probs > 0.5).float().cpu().numpy().astype(int)
        all_sequences.extend(samples)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  处理进度: {batch_idx + 1}/{len(data_loader)} batches")

    with open(output_path, "w") as f:
        for seq in all_sequences:
            f.write(''.join(map(str, seq)) + '\n')

    inference_time = time.time() - inference_start_time
    
    print(f"\n推理完成!")
    print(f"已保存 {len(all_sequences)} 条去偏序列到 {output_path}")
    print(f"推理耗时: {timedelta(seconds=int(inference_time))}")
    print(f"平均每条序列: {inference_time / len(all_sequences):.4f}s")


# ============ 主程序 ============
if __name__ == "__main__":
    script_start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    data_load_start = time.time()
    file_path = "dataset1-1.txt"
    dataset = DNADataset(file_path)
    print(f"数据加载耗时: {time.time() - data_load_start:.2f}s")
    print(f"数据集大小: {len(dataset)} 条序列")

    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True
    )

    # ===== 推理模式 =====
    model = CUDAOptimizedDebiasCNN(seq_len=1000, use_checkpoint=False).to(device)
    
    print("\n加载预训练模型...")
    model_load_start = time.time()
    model.load_state_dict(torch.load("debiased_cnn_cuda_optimized_v4_4070_accel.pth", map_location=device))
    print(f"模型加载耗时: {time.time() - model_load_start:.2f}s")
    
    model.eval()
    
    # 推理并保存
    save_generated_sequences(
        model, data_loader, device, 
        output_path="1-1check.txt",
        sample_with_bernoulli=True
    )
    
    # 脚本总耗时
    total_script_time = time.time() - script_start_time
    print("\n" + "=" * 60)
    print(f"脚本总运行时间: {timedelta(seconds=int(total_script_time))}")
    print("=" * 60)