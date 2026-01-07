import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

# ===================== 设置与工具 =====================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 性能/确定性折中
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def log_cuda_memory(step_name: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[{step_name}] GPU内存 - 已分配: {allocated:.1f}MB, 预留: {reserved:.1f}MB")

# ===================== 数据集加载 =====================
def load_dna_sequences(file_path):
    bit_sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip()
            if sequence:
                bit_sequence = [int(bit) for bit in sequence]
                bit_sequences.append(bit_sequence)
    return np.array(bit_sequences, dtype=np.float32)

class DNADataset(Dataset):
    def __init__(self, file_path):
        self.data = load_dna_sequences(file_path)
        self.data_tensor = torch.from_numpy(self.data).unsqueeze(1)  # [N,1,L]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_tensor[idx]

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
    # 强制 float32 进行对数计算以避免 float16 精度问题
    probs = output.clamp(_GLOBAL_EPS, 1 - _GLOBAL_EPS)
    probs_f32 = probs.float()  # compute logs in float32
    # 全局熵（float32）
    global_entropy = -(probs_f32 * torch.log(probs_f32 + _GLOBAL_EPS) + (1 - probs_f32) * torch.log(1 - probs_f32 + _GLOBAL_EPS))
    global_entropy = global_entropy.mean(dim=1)
    global_loss = (global_entropy - target_entropy).abs().mean()

    # 局部熵（窗口）
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
    # 关键：在 float32 上做 FFT / log 等计算
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

def compute_loss(output):
    # 全局数值保护 & clamp
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("[Warning] compute_loss: output has NaN/Inf -> nan_to_num applied")
    output = torch.nan_to_num(output, nan=0.5, posinf=1.0 - _GLOBAL_EPS, neginf=_GLOBAL_EPS)
    output = output.clamp(_GLOBAL_EPS, 1.0 - _GLOBAL_EPS)
    probs = output

    # 计算每一项损失（注意某些项在 float32 上做内部计算）
    loss_entropy = compute_entropy_loss(probs)
    loss_balance = compute_bit_balance_loss(probs)
    loss_trans = compute_expected_transition_loss(probs)
    loss_autocorr = compute_autocorr_loss(probs)
    loss_flat = compute_spectral_flatness_loss(probs)

    sampled_binary = straight_through_bernoulli(probs).clamp(0.0, 1.0)

    loss_lrun = compute_longest_run_loss_from_binary(sampled_binary)
    loss_cumsum = compute_cumulative_sum_loss_from_binary(sampled_binary)

    # 合并并设置权重
    total_loss = (
        2.4 * loss_balance +
        1.6 * loss_cumsum +
        1.2 * loss_trans +
        1.0 * loss_lrun +
        0.8 * loss_entropy +
        0.6 * loss_flat +
        0.4 * loss_autocorr
    )

    # 逐项 NaN 检查：若某项为 NaN，打印并将其替换为大数以触发训练异常保护（更容易定位）
    items = {
        'total_loss': total_loss, 'entropy': loss_entropy, 'balance': loss_balance,
        'trans': loss_trans, 'autocorr': loss_autocorr, 'lrun': loss_lrun, 'flat': loss_flat, 'cumsum': loss_cumsum
    }
    for k, v in items.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"[NaN-Detected] loss component {k} is NaN/Inf. value={v}")
            # 防止返回 NaN 进一步破坏训练：把 NaN 替为大值（以便 optimizer 能抛出梯度或我们能看到问题）
            items[k] = torch.tensor(1e6, device=output.device, dtype=output.dtype)

    # 使用可能被替换后的值
    total_loss = items['total_loss'] if not (torch.isnan(items['total_loss']).any() or torch.isinf(items['total_loss']).any()) else (
        2.4 * items['balance'] + 1.6 * items['cumsum'] + 1.2 * items['trans']
    )

    return total_loss, items['entropy'], items['balance'], items['trans'], items['autocorr'], items['lrun'], items['flat'], items['cumsum']

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
        lrun_vals.append(compute_longest_run_loss_from_binary(sampled).item())
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


# ===================== 训练主循环 =====================
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=120, patience=12, use_amp=True):
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    epoch_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        totals = {name: 0.0 for name in ['loss', 'entropy', 'balance', 'trans', 'autocorr', 'lrun', 'flat', 'cumsum']}
        batch_times = []

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
                 loss_cumsum) = compute_loss(output)

            # 检查 loss 是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Error] total loss is NaN/Inf. loss_entropy={loss_entropy}, loss_balance={loss_balance}, loss_cumsum={loss_cumsum}")
                # 为了安全停止训练以避免继续破坏权重
                raise RuntimeError("Total loss became NaN/Inf. See printed loss components above.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            totals['loss'] += loss.item()
            totals['entropy'] += (loss_entropy.item() if not (torch.isnan(loss_entropy) or torch.isinf(loss_entropy)) else 0.0)
            totals['balance'] += (loss_balance.item() if not (torch.isnan(loss_balance) or torch.isinf(loss_balance)) else 0.0)
            totals['trans'] += (loss_trans.item() if not (torch.isnan(loss_trans) or torch.isinf(loss_trans)) else 0.0)
            totals['autocorr'] += (loss_autocorr.item() if not (torch.isnan(loss_autocorr) or torch.isinf(loss_autocorr)) else 0.0)
            totals['lrun'] += (loss_lrun.item() if not (torch.isnan(loss_lrun) or torch.isinf(loss_lrun)) else 0.0)
            totals['flat'] += (loss_flat.item() if not (torch.isnan(loss_flat) or torch.isinf(loss_flat)) else 0.0)
            totals['cumsum'] += (loss_cumsum.item() if not (torch.isnan(loss_cumsum) or torch.isinf(loss_cumsum)) else 0.0)

            batch_times.append(time.time() - batch_start)

        avg_train_loss = totals['loss'] / max(1, len(train_loader))
        avg_batch_time = np.mean(batch_times) if batch_times else 0.0

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bitstream in val_loader:
                bitstream = bitstream.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                    loss, *_ = compute_loss(model(bitstream))
                val_loss += (loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0)

        avg_val_loss = val_loss / max(1, len(val_loader))
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
            f"批次时间: {avg_batch_time:.3f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'checkpoints/best_model_v5_cumsum.pth')
            print(f"  ✓ 保存最佳模型 (验证损失: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！{patience}个epoch无改善")
                break

        if (epoch + 1) % 8 == 0:
            evaluate_sequences(model, val_loader, device)

    total_time = time.time() - epoch_start_time
    print(f"\n训练耗时: {total_time/3600:.2f}小时")

    if os.path.exists('checkpoints/best_model_v5_cumsum.pth'):
        checkpoint = torch.load('checkpoints/best_model_v5_cumsum.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (Epoch {checkpoint['epoch']+1}, 验证损失: {checkpoint['val_loss']:.6f})")
    else:
        print("未找到保存的最佳模型检查点。")
    return model

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    set_seed(2025)
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

    file_path = "dataset1-1.txt"
    dataset = DNADataset(file_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    print(f"数据集大小: 训练={train_size}, 验证={val_size}")

    model = CUDAOptimizedDebiasCNN(seq_len=1000, use_checkpoint=False).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)

    use_amp = device.type == 'cuda'
    print(f"混合精度训练: {use_amp}")

    model = train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=120, patience=12, use_amp=use_amp)

    print("\n=== 最终评估 ===")
    evaluate_sequences(model, val_loader, device, num_batches=8)

    save_generated_sequences(model, train_loader, device, output_path="train_CNN.txt", sample_with_bernoulli=True)

    torch.save(model.state_dict(), "debiased_cnn.pth")
    print("训练完成！模型已保存到 debiased_cnn.pth")

