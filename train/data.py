# -*- coding: utf-8 -*-
"""数据加载: 结构化加载(代表性选择用) 与 扁平信号加载(直接训练用)。"""

import os
import numpy as np
import pandas as pd

from . import config
from .preprocess import _read_csv_any


# ================== 数据加载(扁平, 每条信号一行) ==================
def load_data():
    files = sorted(
        [
            f for f in os.listdir(config.DATA_PATH)
            if f.endswith(".csv")
            and "label" not in f
            and f.split("_")[0].isdigit()
        ],
        key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1].split(".")[0]))
    )

    # 显式 sample_id -> label 映射(自动跳空行/非法行)
    df_label = pd.read_csv(config.LABEL_PATH, header=None)
    labels = pd.to_numeric(df_label.iloc[:, 0], errors="coerce").dropna().astype(int).values
    sample_ids = sorted({int(f.split("_")[0]) for f in files})
    assert len(labels) >= len(sample_ids), f"label 数量({len(labels)}) < 样本数({len(sample_ids)})"
    label_map = {sid: int(lb) for sid, lb in zip(sample_ids, labels[:len(sample_ids)])}

    X, groups, y = [], [], []
    for f in files:
        sid = int(f.split("_")[0])
        df_file = pd.read_csv(os.path.join(config.DATA_PATH, f), header=None)
        cnt = 0
        for i in range(df_file.shape[0]):
            sig = df_file.iloc[i].values.astype(float)
            if np.all(sig == 0):
                continue
            sig = sig[:1024] if len(sig) >= 1024 else np.pad(sig, (0, 1024 - len(sig)))
            X.append(sig)
            groups.append(sid)
            y.append(label_map[sid])
            cnt += 1
            if cnt >= config.SIGNALS_PER_FILE:
                break

    print(f"加载 {len(files)} 个文件, {len(sample_ids)} 个样本, {len(X)} 条信号")
    print("标签分布:", dict(pd.Series(y).value_counts().sort_index()))
    return np.array(X), np.array(groups), np.array(y), label_map


# ================== 结构化加载(每部位 K 次敲击) ==================
def load_dataset_structured():
    """结构化加载: {sample_id: {'label': int, 'positions': {pos: array(K,1024)}}}"""
    df_label = _read_csv_any(config.LABEL_PATH)
    labels = pd.to_numeric(df_label.iloc[:, 0], errors="coerce").dropna().astype(int).values
    files = [f for f in os.listdir(config.DATA_PATH)
             if f.endswith(".csv") and "label" not in f and f.split("_")[0].isdigit()]
    sids = sorted({int(f.split("_")[0]) for f in files})
    assert len(labels) >= len(sids), f"标签数({len(labels)}) < 样本数({len(sids)})"
    label_map = {sid: int(v) for sid, v in zip(sids, labels[:len(sids)])}

    samples = {}
    for f in files:
        sid = int(f.split("_")[0])
        pos = f.split("_")[1].split(".")[0]
        if pos not in config.POSITIONS or sid not in label_map:
            continue
        df = _read_csv_any(os.path.join(config.DATA_PATH, f))
        arr = df.values
        arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr[np.any(arr != 0, axis=1)]
        if arr.shape[0] == 0:
            continue
        if arr.shape[1] > 1024:
            arr = arr[:, :1024]
        elif arr.shape[1] < 1024:
            arr = np.pad(arr, ((0, 0), (0, 1024 - arr.shape[1])))
        samples.setdefault(sid, {"label": label_map[sid], "positions": {}})
        samples[sid]["positions"][pos] = arr
    return samples


__all__ = ["load_data", "load_dataset_structured"]
