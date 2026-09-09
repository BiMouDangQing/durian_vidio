# -*- coding: utf-8 -*-
"""模型构建、样本级聚合、评估报告与分层划分。"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lightgbm import LGBMRegressor

from . import config


# ================== 连续分数 -> 类别 ==================
def score_to_class(score):
    if score < config.THRESH_1_2:
        return 1
    elif score < config.THRESH_2_3:
        return 2
    else:
        return 3


# ================== 模型(低复杂度 + 强正则) ==================
def build_model():
    return Pipeline([
        ("scaler", RobustScaler()),
        ("lgbm", LGBMRegressor(**config.LGB_REGR_PARAMS))
    ])


# ================== 样本级聚合 ==================
def aggregate_by_sample(scores, groups, y):
    df = pd.DataFrame({"sample_id": groups, "score": scores, "true": y})
    agg = df.groupby("sample_id").agg(
        score=("score", "mean"),
        true_class=("true", "first")
    ).reset_index()
    agg["pred_class"] = agg["score"].apply(score_to_class)
    return agg


# ================== 混淆矩阵 + 报告 ==================
def report(sample_df, title):
    y_true = sample_df["true_class"].values
    y_pred = sample_df["pred_class"].values
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print("混淆矩阵 (行=真实, 列=预测)  [1=生果 2=中间果 3=熟果]:")
    print(pd.DataFrame(
        cm,
        index=["True-1", "True-2", "True-3"],
        columns=["Pred-1", "Pred-2", "Pred-3"]
    ))
    print(f"样本级准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], zero_division=0))


# ================== 7:3 按样本分层划分 ==================
def stratified_split(groups, label_map, test_size=0.2, seed=42):
    """按 sample_id 分层划分, 保证训练/测试集类别比例一致"""
    unique_sids = sorted(set(groups.tolist()))
    unique_labels = np.array([label_map[s] for s in unique_sids])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.array(unique_sids), unique_labels))
    train_sids = {unique_sids[i] for i in tr_idx}
    val_sids = {unique_sids[i] for i in va_idx}
    return train_sids, val_sids


__all__ = ["score_to_class", "build_model", "aggregate_by_sample", "report", "stratified_split"]
