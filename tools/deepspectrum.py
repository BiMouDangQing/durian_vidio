# -*- coding: utf-8 -*-
"""Deep Spectrum: 用 ImageNet 预训练 CNN 从梅尔谱图提取深度学习特征。

参考: Cummins et al. 2017 "Image classification of audio spectrograms"。
流程: 信号 -> 梅尔谱图(dB) -> 归一化 -> resize 224x224 -> 3通道复制
      -> ResNet18 -> 512 维特征向量 / 中间层 feature maps。
首次调用会加载/下载 ResNet18 预训练权重(约 44MB), 之后缓存复用。
"""

import numpy as np
import librosa

_model = None
_torch_failed = False


def _import_torch():
    """导入 torch; 若此前已失败则直接抛错, 避免重复 DLL 加载崩溃。"""
    global _torch_failed
    if _torch_failed:
        raise RuntimeError("torch 不可用(此前加载失败)")
    try:
        import torch
        return torch
    except Exception as e:
        _torch_failed = True
        raise RuntimeError(f"torch 加载失败: {e}")


def _load_model():
    """懒加载 ImageNet 预训练 ResNet18(完整模型, 含中间层)。"""
    global _model
    if _model is None:
        torch = _import_torch()
        import torchvision.models as models
        _model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _model.eval()
    return _model


def _preprocess(x, sr):
    """信号 -> (1,3,224,224) ImageNet 标准化 tensor。"""
    torch = _import_torch()
    import torch.nn.functional as F

    mel = librosa.feature.melspectrogram(
        y=np.asarray(x, dtype=np.float64), sr=sr,
        n_mels=224, fmin=50, fmax=4000, n_fft=256, hop_length=64)
    mel_db = librosa.power_to_db(mel)
    img = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = np.asarray(img, dtype=np.float32)

    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)  # 3 通道复制(RGB)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (t - mean) / std


def compute_deep_spectrum(x, sr):
    """从敲击信号 x 提取 Deep Spectrum 特征向量 (512,)。"""
    torch = _import_torch()
    model = _load_model()
    t = _preprocess(x, sr)
    activation = {}

    def hook(module, inp, out):
        activation["f"] = out

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(t)
    handle.remove()
    return activation["f"].flatten().numpy()  # (512,)


def compute_feature_maps(x, sr, layer_name="layer3", n_channels=16):
    """返回 ResNet18 指定中间层的 feature maps, 形状 (n_channels, H, W)。

    layer_name: layer1(64ch,56x56) / layer2(128ch,28x28) / layer3(256ch,14x14) / layer4(512ch,7x7)
    """
    torch = _import_torch()
    model = _load_model()
    layer = getattr(model, layer_name)
    t = _preprocess(x, sr)
    activation = {}

    def hook(module, inp, out):
        activation["f"] = out

    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(t)
    handle.remove()
    return activation["f"][0][:n_channels].cpu().numpy()  # (n_channels, H, W)


def feature_maps_grid(fmaps):
    """把 (n,H,W) feature maps 拼成网格图 (rows*H, cols*W), 逐通道 min-max 归一化。"""
    n, h, w = fmaps.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        ch = fmaps[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = ch
    return grid

