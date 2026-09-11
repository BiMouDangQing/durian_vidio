# -*- coding: utf-8 -*-
"""深度学习训练: PANNs CNN14 迁移学习(三分类), GPU 加速。

针对 64ms 短敲击信号(1024 点 @16kHz)做了适配:
- 输入: mel 谱图 64 bins x 25 帧(n_fft=256, hop=32);
- 网络: PANNs CNN14(AudioSet 预训练), 前 4 个卷积块只对频率维度池化(保持时间维度),
        卷积核权重与预训练模型一致, 可加载官方 Cnn14_mAP=0.431 权重迁移;
- 训练: 冻结低层 conv_block1-4, 微调 conv_block5-6 + fc1 + 分类头;
- 评估: 按样本聚合(20 条信号概率平均 -> argmax), 复用 train.model 的分层划分与报告。

运行:
    D:\\model\\model\\vidio\\.venv\\python.exe -m train.dl_train
    (或: D:\\model\\model\\vidio\\.venv\\python.exe train\\dl_train.py)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 兼容两种运行方式: python -m train.dl_train 与 python train/dl_train.py
try:
    from . import config
    from .data import load_data
    from .model import stratified_split, report
    from .preprocess import bandpass_filter
except ImportError:
    from train import config  # noqa: F401
    from train.data import load_data
    from train.model import stratified_split, report
    from train.preprocess import bandpass_filter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 训练超参数 ==================
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_BLOCKS = 4          # 冻结前 4 个卷积块(低层特征通用), 微调后 2 块 + fc
NUM_WORKERS = 0            # Windows 下多进程 DataLoader 易出问题, 置 0
MIXUP_ALPHA = 0.2
USE_WANDB = False          # 是否同步训练记录到 wandb(需先 wandb login)
MODEL_TYPE = "cnn14"       # 模型框架: cnn14 / cnn10 / resnet18

WEIGHT_DIR = os.path.join(config.PROJECT_ROOT, "weights")
# 权重文件(优先 hf-mirror 的 safetensors, 其次 zenodo 官方 pth)
WEIGHT_FILES = [
    os.path.join(WEIGHT_DIR, "model.safetensors"),
    os.path.join(WEIGHT_DIR, "Cnn14_mAP=0.431.pth"),
]
WEIGHT_URL = "https://hf-mirror.com/nicofarr/panns_Cnn14/resolve/main/model.safetensors"


# ================== 信号 -> mel 谱图 ==================
def signal_to_mel(x, sr=config.SR, n_mels=64, n_fft=256, hop=32, fmin=50, fmax=8000):
    """1024 点信号 -> (n_mels, T) dB 谱图。"""
    x = np.asarray(x, dtype=np.float64)
    mel = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop,
        fmin=fmin, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def get_mel_params(model_type="cnn14"):
    """不同模型框架使用不同的 mel 谱图参数: (n_mels, n_fft, hop)。"""
    if model_type == "resnet18":
        return 128, 512, 8    # 更细频率+时间分辨率, 适配 224x224 resize
    return 64, 256, 32        # PANNs 标准(64 bins)


def auto_batch_size(model_type="cnn14"):
    """根据 GPU 显存与模型类型自动适配 batch size, 尽量吃满显存。"""
    if not torch.cuda.is_available():
        return 16
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_gb >= 20:          # RTX 3090/4090 等大显存
        return {"resnet18": 512, "cnn10": 512, "cnn14": 256}[model_type]
    if total_gb >= 16:
        return {"resnet18": 256, "cnn10": 256, "cnn14": 128}[model_type]
    if total_gb >= 8:
        return {"resnet18": 128, "cnn10": 128, "cnn14": 64}[model_type]
    return {"resnet18": 64, "cnn10": 64, "cnn14": 32}[model_type]


def preprocess_signal(x):
    """带通滤波 + 去均值 + 幅度归一化(不截取, 保留完整 1024 点)。"""
    x = np.asarray(x, dtype=np.float64)
    try:
        x = bandpass_filter(x, config.SR)
    except Exception:
        pass
    x = x - np.mean(x)
    s = np.std(x)
    if s > 1e-12:
        x = x / s
    m = np.max(np.abs(x))
    if m > 1e-12:
        x = x / m
    return x.astype(np.float32)


# ================== PANNs CNN14(短信号适配) ==================
def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14Short(nn.Module):
    """PANNs CNN14 骨干 + 三分类头, 针对短信号只对频率维池化。"""

    def __init__(self, num_classes=3, mel_bins=64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_head = nn.Linear(2048, num_classes, bias=True)
        _init_layer(self.fc1)
        _init_layer(self.fc_head)

    def forward(self, x):
        # x: (B, 1, T, F)
        x = x.transpose(1, 3)              # (B, F, T, 1)
        x = self.bn0(x)                    # 对 F 维归一化
        x = x.transpose(1, 3)              # (B, 1, T, F)

        # 前 4 块只池化频率(1,2), 保持时间维度 25 帧; 后 2 块不池化
        x = self.conv_block1(x, pool_size=(1, 2))
        x = self.conv_block2(x, pool_size=(1, 2))
        x = self.conv_block3(x, pool_size=(1, 2))
        x = self.conv_block4(x, pool_size=(1, 2))
        x = self.conv_block5(x, pool_size=(1, 1))
        x = self.conv_block6(x, pool_size=(1, 1))

        x = torch.mean(x, dim=3)            # (B, 2048, T) 频率维平均
        x1, _ = torch.max(x, dim=2)         # (B, 2048) 时间最大
        x2 = torch.mean(x, dim=2)           # (B, 2048) 时间平均
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc_head(x)


class Cnn10Short(nn.Module):
    """PANNs CNN10 骨干 + 三分类头(轻量版, 少一个卷积块)。"""

    def __init__(self, num_classes=3, mel_bins=64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_head = nn.Linear(1024, num_classes, bias=True)
        _init_layer(self.fc1)
        _init_layer(self.fc_head)

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(1, 2))
        x = self.conv_block2(x, pool_size=(1, 2))
        x = self.conv_block3(x, pool_size=(1, 2))
        x = self.conv_block4(x, pool_size=(1, 2))
        x = self.conv_block5(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc_head(x)


class ResNet18Audio(nn.Module):
    """ImageNet 预训练 ResNet18, 输入 mel 谱图(resize 224), 三分类头。"""

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        # x: (B, 1, T, F) -> (B, 3, 224, 224)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        # min-max 归一化到 [0,1] + ImageNet 标准化
        x = (x - x.amin(dim=(2, 3), keepdim=True)) / \
            (x.amax(dim=(2, 3), keepdim=True) - x.amin(dim=(2, 3), keepdim=True) + 1e-8)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.backbone(x)


def _load_state_dict(path):
    """按扩展名加载权重 dict(safetensors 或 torch pth)。"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def load_pretrained(model):
    """加载 AudioSet 预训练权重(卷积层 + fc1), 分类头随机初始化。"""
    for path in WEIGHT_FILES:
        if not os.path.exists(path):
            continue
        state = _load_state_dict(path)
        if "model" in state:
            state = state["model"]
        # 去常见前缀(backbone. / model. / panns. / cnn14.), 兼容不同保存格式
        cleaned = {}
        for k, v in state.items():
            kk = k
            for prefix in ("backbone.", "model.", "panns.", "cnn14."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
                    break
            cleaned[kk] = v
        # 只加载名称匹配的参数(bn0/conv_block1-6/fc1), 忽略 fc_audioset 等
        model_dict = model.state_dict()
        matched = {k: v for k, v in cleaned.items()
                   if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"已加载预训练权重: {len(matched)}/{len(model_dict)} 层, 来源 {path}")
        return len(matched) > 0
    return False


def make_model(num_classes=3, model_type="cnn14", pretrained=True):
    if model_type == "resnet18":
        model = ResNet18Audio(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "cnn10":
        model = Cnn10Short(num_classes=num_classes)
        if pretrained:
            loaded = load_pretrained(model)   # 复用 CNN14 权重(前 5 块)
            if not loaded:
                print("[警告] 未找到预训练权重, 将从零训练。")
    else:
        model = Cnn14Short(num_classes=num_classes)
        if pretrained:
            loaded = load_pretrained(model)
            if not loaded:
                print("[警告] 未找到预训练权重, 将从零训练。")
                print(f"        可手动下载: {WEIGHT_URL}")
                print(f"        保存到: {WEIGHT_FILES[0]}")
    model = model.to(DEVICE)
    model.model_type = model_type   # 记录模型类型(供预测时选择 mel 参数)
    return model


def freeze_backbone(model, n_blocks=FREEZE_BLOCKS):
    """冻结主干: CNN 系列冻结前 n_blocks 个卷积块, ResNet 冻结 backbone(仅训 fc)。"""
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.backbone.fc.parameters():
            p.requires_grad = True
        return model
    for i in range(1, n_blocks + 1):
        block = getattr(model, f"conv_block{i}", None)
        if block is None:
            continue
        for p in block.parameters():
            p.requires_grad = False
    return model


# ================== 数据增强 ==================
def spec_augment(mel, freq_mask=8, time_mask=5):
    """mel: (1, T, F) tensor, 随机时/频遮蔽。"""
    _, T, F_ = mel.shape
    if np.random.rand() < 0.5:
        f0 = np.random.randint(0, max(1, F_ - freq_mask))
        mel[:, :, f0:f0 + freq_mask] = 0.0
    if np.random.rand() < 0.5 and T > time_mask + 1:
        t0 = np.random.randint(0, T - time_mask)
        mel[:, t0:t0 + time_mask, :] = 0.0
    return mel


def add_noise(x, snr_db=20.0):
    """加高斯噪声。"""
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    return x + np.random.randn(*x.shape).astype(np.float32) * np.sqrt(noise_power)


# ================== Dataset ==================
class KnockDataset(Dataset):
    def __init__(self, X, groups, y, augment=False, model_type="cnn14"):
        self.X = X
        self.groups = groups
        self.y = (y - 1).astype(np.int64)   # 1/2/3 -> 0/1/2
        self.augment = augment
        self.n_mels, self.n_fft, self.hop = get_mel_params(model_type)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = preprocess_signal(self.X[idx])
        if self.augment:
            x = add_noise(x)
        mel = signal_to_mel(x, n_mels=self.n_mels, n_fft=self.n_fft, hop=self.hop)   # (F, T)
        mel = torch.from_numpy(mel).unsqueeze(0)          # (1, F, T)
        mel = mel.permute(0, 2, 1)                        # (1, T, F)
        if self.augment:
            mel = spec_augment(mel)
        return mel, self.groups[idx], self.y[idx]


def mixup_batch(mel, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return mel, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(mel.size(0), device=mel.device)
    mel_mixed = lam * mel + (1 - lam) * mel[idx]
    return mel_mixed, (y, y[idx], lam)


def mixup_loss(criterion, logits, y_tuple):
    y_a, y_b, lam = y_tuple
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ================== 样本级聚合 ==================
def aggregate_probs(probs, groups, y):
    """信号级 softmax 概率 -> 样本级(平均概率 -> argmax)。probs: (N, 3), y 为 0/1/2。"""
    df = pd.DataFrame({"sample_id": groups,
                       "p0": probs[:, 0], "p1": probs[:, 1], "p2": probs[:, 2],
                       "true": y})
    agg = df.groupby("sample_id").agg(
        p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean"),
        true_class=("true", "first")).reset_index()
    pred_class = np.argmax(agg[["p0", "p1", "p2"]].values, axis=1) + 1
    agg["true_class"] = agg["true_class"].astype(int) + 1
    agg["pred_class"] = pred_class
    return agg


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs_list, groups_list, y_list = [], [], []
    for mel, g, y in loader:
        mel = mel.to(DEVICE)
        logits = model(mel)
        p = torch.softmax(logits, dim=1)
        probs_list.append(p.cpu().numpy())
        groups_list.extend(g.numpy().tolist())
        y_list.extend(y.numpy().tolist())
    return np.vstack(probs_list), np.array(groups_list), np.array(y_list)


# ================== 主流程 ==================
def main(progress_callback=None, pause_event=None):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"设备: {DEVICE}")
    print(f"输出目录: {config.OUTPUT_DIR}")

    # wandb 训练跟踪(可选, 需先 `wandb login`)
    wandb_run = None
    if USE_WANDB:
        import wandb
        os.environ.setdefault("WANDB_SILENT", "true")
        wandb_run = wandb.init(
            project="durian-maturity",
            name=os.path.basename(config.OUTPUT_DIR),
            config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                    "model": MODEL_TYPE, "test_size": config.TEST_SIZE},
            silent=True,
        )
        print("wandb 跟踪已开启(后台静默)")

    # 1. 数据加载 + 分层划分(按样本, 不泄漏)
    X, groups, y, label_map = load_data()
    train_sids, val_sids = stratified_split(groups, label_map, config.TEST_SIZE, config.SEED)
    tr_mask = np.array([g in train_sids for g in groups])
    va_mask = np.array([g in val_sids for g in groups])
    print(f"训练集: {len(train_sids)} 样本 / {int(tr_mask.sum())} 条信号; "
          f"测试集: {len(val_sids)} 样本 / {int(va_mask.sum())} 条信号")

    tr_ds = KnockDataset(X[tr_mask], groups[tr_mask], y[tr_mask], augment=True, model_type=MODEL_TYPE)
    va_ds = KnockDataset(X[va_mask], groups[va_mask], y[va_mask], augment=False, model_type=MODEL_TYPE)
    # 自动适配 batch size(0 表示自动, 根据 GPU 显存与模型类型)
    effective_batch = BATCH_SIZE if BATCH_SIZE > 0 else auto_batch_size(MODEL_TYPE)
    if BATCH_SIZE <= 0:
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"自动适配 batch size: {effective_batch} (GPU {gb:.0f}GB, 模型 {MODEL_TYPE})")
        else:
            print(f"自动适配 batch size: {effective_batch} (CPU)")
    tr_loader = DataLoader(tr_ds, batch_size=effective_batch, shuffle=True,
                           num_workers=NUM_WORKERS, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=effective_batch, shuffle=False,
                           num_workers=NUM_WORKERS)

    # 2. 模型 + 冻结 + 优化器
    model = make_model(num_classes=3, model_type=MODEL_TYPE, pretrained=True)
    model = freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"模型参数: 总 {n_total/1e6:.2f}M, 可训练 {n_train/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 3. 训练循环
    best_acc = 0.0
    best_path = os.path.join(config.OUTPUT_DIR, f"model_{MODEL_TYPE}.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss, n_batch = 0.0, 0
        total_batches = len(tr_loader)
        for batch_idx, (mel, g, yb) in enumerate(tr_loader):
            if pause_event is not None:
                pause_event.wait()   # 暂停时阻塞, 恢复后继续
            mel, yb = mel.to(DEVICE), yb.to(DEVICE)
            mel, yb_mix = mixup_batch(mel, yb)
            optimizer.zero_grad()
            logits = model(mel)
            loss = mixup_loss(criterion, logits, yb_mix) if isinstance(yb_mix, tuple) \
                else criterion(logits, yb_mix)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batch += 1
            # 每批上报进度(loss 为当前批 loss, acc=None 表示训练中)
            if progress_callback is not None:
                progress_callback(epoch, EPOCHS, batch_idx + 1, total_batches, loss.item(), None)
        scheduler.step()

        # 验证(样本级)
        probs, g_va, y_va = predict_probs(model, va_loader)
        va_df = aggregate_probs(probs, g_va, y_va)
        acc = (va_df["true_class"] == va_df["pred_class"]).mean()
        avg_loss = running_loss / max(n_batch, 1)
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.4f} "
              f"| 测试样本级 acc {acc:.4f} | {time.time()-t0:.1f}s")

        # 每轮结束上报(avg_loss + 验证集 acc)
        if progress_callback is not None:
            progress_callback(epoch, EPOCHS, total_batches, total_batches, avg_loss, float(acc))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": avg_loss,
                           "val_acc": float(acc),
                           "lr": scheduler.get_last_lr()[0]})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    # 4. 最终报告(加载最佳模型)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    probs, g_va, y_va = predict_probs(model, va_loader)
    va_df = aggregate_probs(probs, g_va, y_va)
    report(va_df, "深度学习模型(PANNs CNN14) 测试集样本级评估")
    print(f"最佳模型已保存: {best_path} (样本级 acc {best_acc:.4f})")
    full_path = os.path.join(config.OUTPUT_DIR, f"model_{MODEL_TYPE}_full.pt")
    torch.save(model, full_path)
    print(f"完整模型已保存: {full_path}")

    # 保存验证结果(供前端画图) + 记录模型路径
    va_df["true_name"] = va_df["true_class"].map(config.CLASS_NAMES)
    va_df["pred_name"] = va_df["pred_class"].map(config.CLASS_NAMES)
    va_df.to_csv(config.VAL_RESULT_PATH, index=False, encoding="utf-8-sig")
    config.MODEL_PATH = best_path
    print(f"验证结果已保存: {config.VAL_RESULT_PATH}")

    if wandb_run is not None:
        wandb_run.log({"best_val_acc": float(best_acc)})
        wandb_run.finish()
        print("wandb 跟踪已结束")


if __name__ == "__main__":
    main()
