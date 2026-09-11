# -*- coding: utf-8 -*-
"""模型加载: 自动查找最新的训练模型。"""

import os
import glob

import joblib


def load_model(train_module):
    """加载 results/train/*/model.pkl 中最新的模型。

    train_module 为 train 包(提供 PROJECT_ROOT 定位 fqb 根目录)。
    """
    if train_module is None:
        return None
    try:
        root = getattr(train_module, "PROJECT_ROOT", None)
        if not root:
            root = os.path.dirname(train_module.__file__)
        cands = sorted(glob.glob(os.path.join(
            root, "results", "train", "*", "model_lgbm.pkl")))
        if not cands:  # 兼容旧命名 model.pkl
            cands = sorted(glob.glob(os.path.join(
                root, "results", "train", "*", "model.pkl")))
        if not cands:
            return None
        return joblib.load(cands[-1])
    except Exception:
        return None
