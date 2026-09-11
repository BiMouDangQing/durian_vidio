# -*- coding: utf-8 -*-
"""深度学习模型(PANNs CNN14)加载与预测。"""

import numpy as np
import torch


def load_dl_model(path, device=None):
    """加载深度学习模型(.pt): 兼容 state_dict 或完整模型, 从文件名推断模型类型。"""
    import os
    import sys
    from train import dl_train as DL
    if device is None:
        device = DL.DEVICE
    basename = os.path.basename(path)
    model_type = "cnn14"
    for mt in ("cnn14", "cnn10", "resnet18"):
        if mt in basename:
            model_type = mt
            break
    # 兼容旧版完整模型(类曾保存在 __main__ 命名空间)
    for cls_name in ("Cnn14Short", "Cnn10Short", "ResNet18Audio", "ConvBlock"):
        if hasattr(DL, cls_name):
            setattr(sys.modules["__main__"], cls_name, getattr(DL, cls_name))
    model = DL.make_model(num_classes=3, model_type=model_type, pretrained=False)
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        model.load_state_dict(obj)
    else:
        model = obj
    if not hasattr(model, "model_type"):
        model.model_type = model_type
    model.to(device)
    model.eval()
    return model, device


def predict_dl(model, knocks, device):
    """多条敲击信号 -> 样本级类别(平均 softmax 概率 -> argmax)。"""
    from train.dl_train import preprocess_signal, signal_to_mel, get_mel_params
    model_type = getattr(model, "model_type", "cnn14")
    n_mels, n_fft, hop = get_mel_params(model_type)
    model.eval()
    probs = []
    with torch.no_grad():
        for x in knocks:
            x = preprocess_signal(x)
            mel = signal_to_mel(x, n_mels=n_mels, n_fft=n_fft, hop=hop)   # (F, T)
            t = torch.from_numpy(mel).unsqueeze(0).permute(0, 2, 1).unsqueeze(0)  # (1,1,T,F)
            t = t.to(device)
            logits = model(t)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    probs = np.vstack(probs)
    return int(np.argmax(probs.mean(axis=0))) + 1       # 0/1/2 -> 1/2/3
