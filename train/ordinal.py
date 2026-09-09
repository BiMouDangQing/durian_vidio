# -*- coding: utf-8 -*-
"""有序分类(Ordinal Classification): 两个累积二分类器 P(y>=2) / P(y>=3)。

与回归版(pipeline.py)的区别: 不再"回归 + 阈值切分", 而是用两个二分类器
对有序标签建模, 从根上消除对阈值的依赖。

原理:
- 标签 1 < 2 < 3 是有序类别(非等距连续值);
- clf2 学习 P(y>=2): 生果(1) vs 中间果+熟果(2,3);
- clf3 学习 P(y>=3): 生果+中间果(1,2) vs 熟果(3);
- 决策: p2<0.5 -> 1级; 否则 p3<0.5 -> 2级; 否则 -> 3级;
- 单调性修复: p3 = min(p3, p2)(两个独立二分类器概率可能违反 p3<=p2)。

复用 train 包的 load_data / extract_features / stratified_split, 保证特征一致。
"""

import os
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lightgbm import LGBMClassifier

from . import config
from .data import load_data
from .features import extract_features
from .model import stratified_split


# ================== 输出 ==================
def _new_run_time():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _output_dir():
    return os.path.join(config.PROJECT_ROOT, "results", "train_ordinal", _new_run_time())


# ================== 模型(两个二分类器) ==================
def build_models():
    base = dict(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=8,
        max_depth=3,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=config.SEED,
        verbosity=-1
    )
    clf2 = Pipeline([("scaler", RobustScaler()), ("lgbm", LGBMClassifier(objective="binary", **base))])
    clf3 = Pipeline([("scaler", RobustScaler()), ("lgbm", LGBMClassifier(objective="binary", **base))])
    return clf2, clf3


# ================== 有序决策 ==================
def ordinal_decision(p2, p3):
    """p2=P(y>=2), p3=P(y>=3); 修复单调性后按阈值决策"""
    p3 = min(p3, p2)
    if p2 < 0.5:
        return 1
    elif p3 < 0.5:
        return 2
    else:
        return 3


def predict_ordinal(clf2, clf3, X):
    p2 = clf2.predict_proba(X)[:, 1]
    p3 = clf3.predict_proba(X)[:, 1]
    return p2, p3


# ================== 样本级聚合 ==================
def aggregate_by_sample(p2, p3, groups, y):
    df = pd.DataFrame({"sample_id": groups, "p2": p2, "p3": p3, "true": y})
    agg = df.groupby("sample_id").agg(
        p2=("p2", "mean"),
        p3=("p3", "mean"),
        true_class=("true", "first")
    ).reset_index()
    agg["pred_class"] = [ordinal_decision(a, b) for a, b in zip(agg["p2"], agg["p3"])]
    return agg


# ================== 报告 ==================
def report(sample_df, title):
    y_true = sample_df["true_class"].values
    y_pred = sample_df["pred_class"].values
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print("混淆矩阵 (行=真实, 列=预测)  [1=生果 2=中间果 3=熟果]:")
    print(pd.DataFrame(cm, index=["True-1", "True-2", "True-3"], columns=["Pred-1", "Pred-2", "Pred-3"]))
    print(f"样本级准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], zero_division=0))


# ================== 主流程 ==================
def main():
    output_dir = _output_dir()
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")
    val_result_path = os.path.join(output_dir, "val_result.csv")
    print(f"输出目录: {output_dir}")

    X, groups, y, label_map = load_data()
    print("\n提取特征中...")
    X_feat = extract_features(X)
    print(f"特征维度: {X_feat.shape}")

    train_sids, val_sids = stratified_split(groups, label_map, config.TEST_SIZE, config.SEED)
    tr_mask = np.array([g in train_sids for g in groups])
    va_mask = np.array([g in val_sids for g in groups])
    print(f"\n训练集: {len(train_sids)} 样本, 测试集: {len(val_sids)} 样本")

    clf2, clf3 = build_models()
    y_bin2 = (y[tr_mask] >= 2).astype(int)
    y_bin3 = (y[tr_mask] >= 3).astype(int)
    clf2.fit(X_feat[tr_mask], y_bin2)
    clf3.fit(X_feat[tr_mask], y_bin3)

    for name, mask in [("训练集", tr_mask), ("测试集", va_mask)]:
        p2, p3 = predict_ordinal(clf2, clf3, X_feat[mask])
        report(aggregate_by_sample(p2, p3, groups[mask], y[mask]), name)

    joblib.dump({"clf2": clf2, "clf3": clf3}, model_path)
    print(f"\n模型已保存: {model_path}")

    va_p2, va_p3 = predict_ordinal(clf2, clf3, X_feat[va_mask])
    va_df = aggregate_by_sample(va_p2, va_p3, groups[va_mask], y[va_mask])
    va_df["true_name"] = va_df["true_class"].map(config.CLASS_NAMES)
    va_df["pred_name"] = va_df["pred_class"].map(config.CLASS_NAMES)
    va_df.to_csv(val_result_path, index=False, encoding="utf-8-sig")
    print(f"测试集结果已保存: {val_result_path}")


__all__ = ["build_models", "ordinal_decision", "predict_ordinal", "main"]
