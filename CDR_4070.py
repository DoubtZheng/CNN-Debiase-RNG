# 5-CNN-pro3_4070_accel_v4.py
# 4070 加速版 v4：提升数据加载/训练吞吐，保留数值稳定性，并加入内置 NIST(部分)验证
# 主要改动：
# 1) 支持 dataset2.txt -> dataset2.npy 预处理（mmap 读取）
# 2) TF32 + AMP + 可选 torch.compile
# 3) DataLoader：更多 workers + persistent_workers + 更高 prefetch
# 4) “重损失项”(FFT/多lag自相关/LongestRun/CumSum) 降频计算（默认每4步一次）
# 5) fused AdamW（可用则启用）
#
# 原始基础来自你上传的 5-CNN-pro3.py（v5_cumsum 版本）

import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

# SciPy 用于 NIST p-value（若你不想装 SciPy，可把 NIST 部分改回外部脚本）
from scipy.special import erfc, gammaincc

# ===================== 设置与工具 =====================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 性能/确定性折中
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def enable_4070_speedups():
    """
    4070 推荐加速开关：
    - TF32：通常不影响本任务效果，但显著加速
    - matmul precision：PyTorch 2.x
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

def log_cuda_memory(step_name: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[{step_name}] GPU内存 - 已分配: {allocated:.1f}MB, 预留: {reserved:.1f}MB")

# ===================== 数据集：txt -> npy（mmap） =====================
def preprocess_txt01_to_npy(txt_path: str, npy_path: str, seq_len: int = 1000, max_rows: int | None = None):
    """
    一次性把 '0/1' 文本转换成 uint8 的 npy，加速训练。
    - 每行一个序列（长度=seq_len）
    """
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if len(s) != seq_len:
                raise ValueError(f"发现序列长度不等于 {seq_len}: len={len(s)}")
            # ASCII '0'(48)/'1'(49) -> 0/1
            arr = (np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48)
            rows.append(arr)
            if max_rows is not None and len(rows) >= max_rows:
                break

    data = np.stack(rows, axis=0).astype(np.uint8)  # [N,L]
    np.save(npy_path, data)
    print(f"[Preprocess] saved npy: {npy_path} shape={data.shape} dtype={data.dtype}")

class DNADatasetFast(Dataset):
    """
    优先使用 .npy mmap 读取；否则退回读取 txt（慢很多）
    返回：float32 tensor [1, L]
    """
    def __init__(self, file_path: str, seq_len: int = 1000, prefer_npy: bool = True):
        self.seq_len = seq_len
        self.file_path = file_path

        self.npy_path = None
        if prefer_npy and file_path.endswith(".txt"):
            candidate = file_path[:-4] + ".npy"
            if os.path.exists(candidate):
                self.npy_path = candidate

        if self.npy_path is not None:
            self.data = np.load(self.npy_path, mmap_mode="r")  # uint8 [N,L]
            if self.data.ndim != 2 or self.data.shape[1] != seq_len:
                raise ValueError(f"npy 维度不符合预期：got {self.data.shape}, expected (*,{seq_len})")
            self.mode = "npy"
        else:
            # 慢路径：读 txt 到内存
            bit_sequences = []
            with open(file_path, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if len(s) != seq_len:
                        raise ValueError(f"发现序列长度不等于 {seq_len}: len={len(s)}")
                    bit_sequences.append([int(ch) for ch in s])
            self.data = np.array(bit_sequences, dtype=np.float32)
            self.mode = "txt_in_memory"

        print(f"[Dataset] mode={self.mode}, N={len(self)}, L={seq_len}")

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        if self.mode == "npy":
            x = self.data[idx].astype(np.float32)  # [L]
        else:
            x = self.data[idx]  # float32 [L]
        return torch.from_numpy(x).unsqueeze(0)  # [1,L]

# ===================== 模型组件 =====================
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

class CUDAOptimizedDebiasCNN(nn.Module):
    def __init__(self, seq_len: int = 1000, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

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

        self.middle = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            EfficientResidualBlock(128, groups=2),
        )

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
        return out.squeeze(1)  # [batch, seq]

# ===================== 数值稳定的损失函数 =====================
_GLOBAL_EPS = 1e-6

def compute_entropy_loss(output, target_entropy=math.log(2)):
    probs = output.clamp(_GLOBAL_EPS, 1 - _GLOBAL_EPS)
    probs_f32 = probs.float()
    global_entropy = -(probs_f32 * torch.log(probs_f32 + _GLOBAL_EPS) + (1 - probs_f32) * torch.log(1 - probs_f32 + _GLOBAL_EPS))
    global_entropy = global_entropy.mean(dim=1)
    global_loss = (global_entropy - target_entropy).abs().mean()

    window_size = 50
    stride = window_size // 2
    if probs_f32.shape[1] >= window_size:
        windows = probs_f32.unfold(dimension=1, size=window_size, step=stride)
        local_entropy = -(windows * torch.log(windows + _GLOBAL_EPS) + (1 - windows) * torch.log(1 - windows + _GLOBAL_EPS)).mean(dim=2)
        local_loss = (local_entropy - target_entropy).abs().mean()
    else:
        local_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

    return 0.5 * global_loss + 0.5 * local_loss

def compute_bit_balance_loss(output):
    global_mean = output.mean(dim=1)
    global_loss = ((global_mean - 0.5) ** 2).mean()

    local_loss = 0.0
    window_sizes = (16, 32, 64, 128, 256, 512)
    for window_size in window_sizes:
        if output.shape[1] >= window_size:
            chunks = output.unfold(dimension=1, size=window_size, step=window_size)
            chunk_means = chunks.mean(dim=2)
            local_loss = local_loss + ((chunk_means - 0.5) ** 2).mean()
    if len(window_sizes) > 0:
        local_loss = local_loss / len(window_sizes)
    return 2.0 * global_loss + local_loss

def compute_expected_transition_loss(output):
    p = output
    trans_prob = p[:, :-1] * (1 - p[:, 1:]) + (1 - p[:, :-1]) * p[:, 1:]
    return ((trans_prob.mean(dim=1) - 0.5) ** 2).mean()

def compute_autocorr_loss(output, max_lag=20):
    centered = output - output.mean(dim=1, keepdim=True)
    total = 0.0
    weight_sum = 0.0
    for lag in range(1, min(max_lag, max(2, output.shape[1] // 2))):
        x1 = centered[:, :-lag]
        x2 = centered[:, lag:]
        numerator = (x1 * x2).sum(dim=1)
        denom = (x1.pow(2).sum(dim=1) * x2.pow(2).sum(dim=1))
        denominator = torch.sqrt(denom + _GLOBAL_EPS)
        corr = (numerator / (denominator + _GLOBAL_EPS)).abs()
        weight = 1.0 / math.sqrt(lag)
        total = total + weight * corr.mean()
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else torch.tensor(0.0, device=output.device, dtype=output.dtype)

def compute_longest_run_loss_from_binary(binary, block_size=128, max_blocks=8):
    batch, seq_len = binary.shape
    total_len = (seq_len // block_size) * block_size
    if total_len == 0:
        return torch.tensor(0.0, device=binary.device, dtype=binary.dtype)

    blocks = binary[:, :total_len].view(batch, -1, block_size)
    num_blocks = blocks.shape[1]
    if num_blocks == 0:
        return torch.tensor(0.0, device=binary.device, dtype=binary.dtype)

    if max_blocks is not None and num_blocks > max_blocks:
        idx = torch.randperm(num_blocks, device=binary.device)[:max_blocks]
        blocks = blocks[:, idx, :]
        num_blocks = blocks.shape[1]

    current = blocks[:, :, 0].clone()
    max_run = current.clone()
    for t in range(1, block_size):
        current = (current + 1) * blocks[:, :, t]
        max_run = torch.maximum(max_run, current)

    counts = torch.stack([
        (max_run <= 10).sum(dim=1),
        (max_run == 11).sum(dim=1),
        (max_run == 12).sum(dim=1),
        (max_run == 13).sum(dim=1),
        (max_run == 14).sum(dim=1),
        (max_run >= 15).sum(dim=1),
    ], dim=1).float()

    expected_pi = torch.tensor([0.117, 0.243, 0.249, 0.175, 0.102, 0.114],
                               device=binary.device, dtype=binary.dtype)
    exp_counts = expected_pi * num_blocks
    chi = ((counts - exp_counts) ** 2 / (exp_counts + 1e-6)).sum(dim=1)
    return (chi / (num_blocks + 1e-8)).mean()

def compute_cumulative_sum_loss_from_binary(binary):
    batch, seq_len = binary.shape
    increments = binary * 2.0 - 1.0
    partial = torch.cumsum(increments, dim=1)
    norm = math.sqrt(seq_len)
    max_abs = partial.abs().max(dim=1)[0]
    loss = (max_abs / (norm + _GLOBAL_EPS)).mean()
    prefix_means = partial / (torch.arange(1, seq_len + 1, device=binary.device, dtype=binary.dtype) + 0.0)
    prefix_abs_mean = prefix_means.abs().mean(dim=1).mean()
    return loss + prefix_abs_mean

def compute_spectral_flatness_loss(output):
    centered = output - output.mean(dim=1, keepdim=True)
    centered_f32 = centered.float()
    n = centered_f32.shape[1]
    window = torch.hann_window(n, device=centered_f32.device, dtype=torch.float32)
    windowed = centered_f32 * window
    spectrum = torch.fft.rfft(windowed, dim=1)
    power = (spectrum.real.pow(2) + spectrum.imag.pow(2)).clamp(min=_GLOBAL_EPS)
    geometric_mean = torch.exp(torch.mean(torch.log(power + _GLOBAL_EPS), dim=1))
    arithmetic_mean = power.mean(dim=1) + _GLOBAL_EPS
    flatness = geometric_mean / arithmetic_mean
    return (1 - flatness).mean().to(output.dtype)

def straight_through_bernoulli(probs):
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0 - _GLOBAL_EPS, neginf=_GLOBAL_EPS)
    probs = probs.clamp(_GLOBAL_EPS, 1.0 - _GLOBAL_EPS)
    rand = torch.rand_like(probs)
    sampled = (probs > rand).float()
    return sampled + (probs - probs.detach())

def compute_loss(output, step: int | None = None, heavy_every: int = 4, autocorr_max_lag: int = 20, lrun_max_blocks: int = 16):
    """
    加速关键：把重损失项降频计算（默认每 heavy_every 步算一次）
    """
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("[Warning] compute_loss: output has NaN/Inf -> nan_to_num applied")

    output = torch.nan_to_num(output, nan=0.5, posinf=1.0 - _GLOBAL_EPS, neginf=_GLOBAL_EPS)
    output = output.clamp(_GLOBAL_EPS, 1.0 - _GLOBAL_EPS)
    probs = output

    heavy = (step is None) or (heavy_every <= 1) or (step % heavy_every == 0)

    loss_entropy = compute_entropy_loss(probs)
    loss_balance = compute_bit_balance_loss(probs)
    loss_trans = compute_expected_transition_loss(probs)

    # 重项降频
    if heavy:
        loss_autocorr = compute_autocorr_loss(probs, max_lag=autocorr_max_lag)
        loss_flat = compute_spectral_flatness_loss(probs)
        sampled_binary = straight_through_bernoulli(probs).clamp(0.0, 1.0)
        loss_lrun = compute_longest_run_loss_from_binary(sampled_binary, max_blocks=lrun_max_blocks)
        loss_cumsum = compute_cumulative_sum_loss_from_binary(sampled_binary)
    else:
        # 0*mean 保持图连通，不影响 backward
        z = 0.0 * probs.mean()
        loss_autocorr = z
        loss_flat = z
        loss_lrun = z
        loss_cumsum = z

    total_loss = (
        2.4 * loss_balance +
        1.6 * loss_cumsum +
        1.2 * loss_trans +
        1.0 * loss_lrun +
        0.8 * loss_entropy +
        0.6 * loss_flat +
        0.4 * loss_autocorr
    )

    items = {
        'total_loss': total_loss, 'entropy': loss_entropy, 'balance': loss_balance,
        'trans': loss_trans, 'autocorr': loss_autocorr, 'lrun': loss_lrun, 'flat': loss_flat, 'cumsum': loss_cumsum
    }
    for k, v in items.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"[NaN-Detected] loss component {k} is NaN/Inf. value={v}")
            items[k] = torch.tensor(1e6, device=output.device, dtype=output.dtype)

    total_loss = items['total_loss'] if not (torch.isnan(items['total_loss']).any() or torch.isinf(items['total_loss']).any()) else (
        2.4 * items['balance'] + 1.6 * items['cumsum'] + 1.2 * items['trans']
    )

    return total_loss, items['entropy'], items['balance'], items['trans'], items['autocorr'], items['lrun'], items['flat'], items['cumsum'], heavy

# ===================== 评估 / 保存 =====================
@torch.no_grad()
def evaluate_sequences(model, data_loader, device, num_batches=4):
    model.eval()
    entropies, means, trans_rates, autocorrs = [], [], [], []
    lrun_vals, cumsum_vals = [], []

    for i, bitstream in enumerate(data_loader):
        if i >= num_batches:
            break
        bitstream = bitstream.to(device, non_blocking=True)
        output = model(bitstream)

        probs = output.clamp(_GLOBAL_EPS, 1 - _GLOBAL_EPS)
        probs_f32 = probs.float()
        entropy = -(probs_f32 * torch.log(probs_f32 + _GLOBAL_EPS) + (1 - probs_f32) * torch.log(1 - probs_f32 + _GLOBAL_EPS)).mean(dim=1)
        entropies.extend(entropy.cpu().numpy())
        means.extend(output.mean(dim=1).cpu().numpy())

        trans = probs[:, :-1] * (1 - probs[:, 1:]) + (1 - probs[:, :-1]) * probs[:, 1:]
        trans_rates.extend(trans.mean(dim=1).cpu().numpy())

        centered = output - output.mean(dim=1, keepdim=True)
        x1 = centered[:, :-1]
        x2 = centered[:, 1:]
        corr = (x1 * x2).sum(dim=1) / (torch.sqrt((x1 ** 2).sum(dim=1) * (x2 ** 2).sum(dim=1)) + _GLOBAL_EPS)
        autocorrs.extend(corr.abs().cpu().numpy())

        sampled = (probs > torch.rand_like(probs)).float()
        lrun_vals.append(compute_longest_run_loss_from_binary(sampled, max_blocks=32).item())
        cumsum_vals.append(compute_cumulative_sum_loss_from_binary(sampled).item())

    print(f"[Eval] 熵={np.mean(entropies):.6f}(目标≈{math.log(2):.6f}), "
          f"均值={np.mean(means):.6f}(目标0.5), "
          f"期望变化率={np.mean(trans_rates):.6f}(目标0.5), "
          f"|自相关(lag=1)|={np.mean(autocorrs):.6f}(目标≈0), "
          f"LRUN_loss≈{np.mean(lrun_vals):.6f}, CumSum_loss≈{np.mean(cumsum_vals):.6f}")

@torch.no_grad()
def save_generated_sequences(model, data_loader, device, output_path="train_CNN5.txt", sample_with_bernoulli=True):
    model.eval()
    all_sequences = []
    for bitstream in data_loader:
        bitstream = bitstream.to(device, non_blocking=True)
        probs = model(bitstream)
        if sample_with_bernoulli:
            samples = (probs > torch.rand_like(probs)).cpu().numpy().astype(int)
        else:
            samples = (probs > 0.5).float().cpu().numpy().astype(int)
        all_sequences.extend(samples)
    with open(output_path, "w") as f:
        for seq in all_sequences:
            f.write(''.join(map(str, seq)) + '\n')
    print(f"已保存 {len(all_sequences)} 条去偏序列到 {output_path}")

# ===================== NIST（部分）验证：Monobit/BlockFreq/Runs/Cusum =====================
def _bits_from_file(path: str, max_bits: int | None = 1_000_000):
    bits = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            bits.extend([1 if c == "1" else 0 for c in s])
            if max_bits is not None and len(bits) >= max_bits:
                break
    return bits[:max_bits] if max_bits is not None else bits

def nist_monobit_test(bits):
    n = len(bits)
    s = sum(1 if b == 1 else -1 for b in bits)
    sobs = abs(s) / math.sqrt(n)
    p = erfc(sobs / math.sqrt(2))
    return float(p)

def nist_block_frequency_test(bits, block_size=128):
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks <= 0:
        return None
    chi_sq = 0.0
    for i in range(num_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi_sq += (pi - 0.5) ** 2
    chi_sq *= 4 * block_size
    p = gammaincc(num_blocks / 2, chi_sq / 2)
    return float(p)

def nist_runs_test(bits):
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1
    expected = 2 * n * pi * (1 - pi)
    variance = 2 * n * pi * (1 - pi) * (2 * pi * (1 - pi) - 1)
    z = abs(runs - expected) / math.sqrt(max(variance, 1e-12))
    p = erfc(z / math.sqrt(2))
    return float(p)

def nist_cumulative_sums_test(bits):
    n = len(bits)
    s = [2 * b - 1 for b in bits]
    partial = np.cumsum(s)
    z = int(np.max(np.abs(partial)))
    if z == 0:
        return 1.0
    # 简化版 Cusum（前向）
    p_value = 1.0
    sqrt_n = math.sqrt(n)
    for k in range(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1):
        p_value -= erfc(((4 * k + 1) * z) / (sqrt_n * math.sqrt(2)))
        p_value += erfc(((4 * k - 1) * z) / (sqrt_n * math.sqrt(2)))
    return float(max(min(p_value, 1.0), 0.0))

def run_nist_suite(file_path: str, alpha: float = 0.01, max_bits: int = 1_000_000, block_size: int = 128):
    bits = _bits_from_file(file_path, max_bits=max_bits)
    results = {
        "Monobit": nist_monobit_test(bits),
        "BlockFrequency": nist_block_frequency_test(bits, block_size=block_size),
        "Runs": nist_runs_test(bits),
        "CumulativeSums": nist_cumulative_sums_test(bits),
    }
    print("\n=== NIST SP800-22（子集）测试结果 ===")
    for name, p in results.items():
        status = "PASS" if p is not None and p >= alpha else "FAIL"
        if p is None:
            print(f"{name:20s} p-value=None [{status}]")
        else:
            print(f"{name:20s} p-value={p:.6f} [{status}]")
    return results

# ===================== 训练主循环 =====================
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=120,
    patience=12,
    use_amp=True,
    heavy_every=4,
    eval_every=20,
    val_max_batches: int | None = 200,
):
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    epoch_start_time = time.time()

    global_step = 0

    for epoch in range(epochs):
        model.train()
        totals = {name: 0.0 for name in ['loss', 'entropy', 'balance', 'trans', 'autocorr', 'lrun', 'flat', 'cumsum']}
        batch_times = []
        heavy_count = 0

        for batch_idx, bitstream in enumerate(train_loader):
            batch_start = time.time()
            bitstream = bitstream.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                output = model(bitstream)

                if epoch == 0 and batch_idx == 0:
                    print(f"[Debug] epoch0 batch0 output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, anynan={torch.isnan(output).any().item()}")

                (loss,
                 loss_entropy,
                 loss_balance,
                 loss_trans,
                 loss_autocorr,
                 loss_lrun,
                 loss_flat,
                 loss_cumsum,
                 heavy_used) = compute_loss(output, step=global_step, heavy_every=heavy_every, autocorr_max_lag=20, lrun_max_blocks=16)

            heavy_count += int(bool(heavy_used))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Error] total loss is NaN/Inf. loss_entropy={loss_entropy}, loss_balance={loss_balance}, loss_cumsum={loss_cumsum}")
                raise RuntimeError("Total loss became NaN/Inf.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            totals['loss'] += loss.item()
            totals['entropy'] += float(loss_entropy.item()) if hasattr(loss_entropy, "item") else float(loss_entropy)
            totals['balance'] += float(loss_balance.item()) if hasattr(loss_balance, "item") else float(loss_balance)
            totals['trans'] += float(loss_trans.item()) if hasattr(loss_trans, "item") else float(loss_trans)
            totals['autocorr'] += float(loss_autocorr.item()) if hasattr(loss_autocorr, "item") else float(loss_autocorr)
            totals['lrun'] += float(loss_lrun.item()) if hasattr(loss_lrun, "item") else float(loss_lrun)
            totals['flat'] += float(loss_flat.item()) if hasattr(loss_flat, "item") else float(loss_flat)
            totals['cumsum'] += float(loss_cumsum.item()) if hasattr(loss_cumsum, "item") else float(loss_cumsum)

            batch_times.append(time.time() - batch_start)
            global_step += 1

        avg_train_loss = totals['loss'] / max(1, len(train_loader))
        avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0

        # 验证（可截断）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, bitstream in enumerate(val_loader):
                if val_max_batches is not None and j >= val_max_batches:
                    break
                bitstream = bitstream.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                    loss, *_ = compute_loss(model(bitstream), step=None, heavy_every=1)
                val_loss += (loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0)

        denom = max(1, (min(len(val_loader), val_max_batches) if val_max_batches is not None else len(val_loader)))
        avg_val_loss = val_loss / denom
        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"训练: {avg_train_loss:.6f} | 验证: {avg_val_loss:.6f} | "
            f"熵: {totals['entropy']/max(1,len(train_loader)):.6f} | "
            f"平衡: {totals['balance']/max(1,len(train_loader)):.6f} | "
            f"变化率: {totals['trans']/max(1,len(train_loader)):.6f} | "
            f"自相关: {totals['autocorr']/max(1,len(train_loader)):.6f} | "
            f"LongestRun: {totals['lrun']/max(1,len(train_loader)):.6f} | "
            f"频谱: {totals['flat']/max(1,len(train_loader)):.6f} | "
            f"CumSum: {totals['cumsum']/max(1,len(train_loader)):.6f} | "
            f"批次时间: {avg_batch_time:.3f}s | heavy步占比: {heavy_count/max(1,len(train_loader)):.2f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'checkpoints/best_model_v4_4070_accel.pth')
            print(f"  ✓ 保存最佳模型 (验证损失: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！{patience}个epoch无改善")
                break

        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            evaluate_sequences(model, val_loader, device, num_batches=6)

    total_time = time.time() - epoch_start_time
    print(f"\n训练耗时: {total_time/3600:.2f}小时")

    ckpt_path = 'checkpoints/best_model_v4_4070_accel.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (Epoch {checkpoint['epoch']+1}, 验证损失: {checkpoint['val_loss']:.6f})")
    else:
        print("未找到保存的最佳模型检查点。")
    return model

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    set_seed(2025)
    enable_4070_speedups()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        try:
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        print(f"CUDA版本: {torch.version.cuda}")
        try:
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        except Exception:
            pass
        log_cuda_memory("初始化")

    # ======== 数据路径与预处理 ========
    seq_len = int(os.environ.get("SEQ_LEN", "1000"))
    file_path = os.environ.get("DATASET", "dataset1-1.txt")
    prefer_npy = os.environ.get("PREFER_NPY", "1") == "1"
    auto_preprocess = os.environ.get("AUTO_PREPROCESS", "1") == "1"

    if file_path.endswith(".txt") and prefer_npy:
        npy_path = file_path[:-4] + ".npy"
        if auto_preprocess and (not os.path.exists(npy_path)):
            print(f"[Info] 未发现 {npy_path}，开始自动预处理（只做一次，后续会很快）...")
            preprocess_txt01_to_npy(file_path, npy_path, seq_len=seq_len, max_rows=None)

    dataset = DNADatasetFast(file_path, seq_len=seq_len, prefer_npy=prefer_npy)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # ======== DataLoader（4070 推荐） ========
    batch_size = int(os.environ.get("BATCH_SIZE", "256"))
    num_workers = int(os.environ.get("NUM_WORKERS", "6")) if torch.cuda.is_available() else 0
    prefetch_factor = int(os.environ.get("PREFETCH", "4")) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if prefetch_factor is not None else 2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if prefetch_factor is not None else 2,
        drop_last=False,
    )

    print(f"数据集大小: 训练={train_size}, 验证={val_size}, batch={batch_size}, workers={num_workers}")

    model = CUDAOptimizedDebiasCNN(seq_len=seq_len, use_checkpoint=False).to(device)

    # 可选：torch.compile（若报错/变慢，设 COMPILE=0）
    do_compile = os.environ.get("COMPILE", "0") == "1"
    if do_compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode=os.environ.get("COMPILE_MODE", "reduce-overhead"))
            print("[Info] torch.compile enabled.")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练)")

    # fused AdamW（可用则启用）
    lr = float(os.environ.get("LR", "1e-4"))
    wd = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
        print("[Info] Using fused AdamW.")
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        print("[Info] Using standard AdamW (fused not available).")

    epochs = int(os.environ.get("EPOCHS", "120"))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (device.type == 'cuda') and (os.environ.get("AMP", "1") == "1")
    print(f"混合精度训练(AMP): {use_amp}")

    heavy_every = int(os.environ.get("HEAVY_EVERY", "4"))
    eval_every = int(os.environ.get("EVAL_EVERY", "20"))
    val_max_batches = os.environ.get("VAL_MAX_BATCHES", "200")
    val_max_batches = None if val_max_batches.lower() == "none" else int(val_max_batches)

    model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        patience=int(os.environ.get("PATIENCE", "12")),
        use_amp=use_amp,
        heavy_every=heavy_every,
        eval_every=eval_every,
        val_max_batches=val_max_batches,
    )

    print("\n=== 最终评估 ===")
    evaluate_sequences(model, val_loader, device, num_batches=10)

    out_txt = os.environ.get("OUT_TXT", "train1_4070.txt")
    save_generated_sequences(model, train_loader, device, output_path=out_txt, sample_with_bernoulli=True)

    out_pth = os.environ.get("OUT_PTH", "debiased_cnn_cuda_optimized_v4_4070_accel.pth")
    torch.save(model.state_dict(), out_pth)
    print(f"训练完成！模型已保存到 {out_pth}")

    # 训练结束跑 NIST（子集）
    try:
        run_nist_suite(out_txt, alpha=float(os.environ.get("NIST_ALPHA", "0.01")),
                       max_bits=int(os.environ.get("NIST_MAX_BITS", "1000000")),
                       block_size=int(os.environ.get("NIST_BLOCK", "128")))
    except Exception as e:
        print(f"[Warn] NIST suite skipped due to error: {e}")
