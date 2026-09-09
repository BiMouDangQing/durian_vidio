# -*- coding: utf-8 -*-
"""训练主流程: 数据加载 -> 代表性选择/筛选 -> 特征 -> 训练 -> 评估 -> 保存。"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score

from . import config
from .data import load_data, load_dataset_structured
from .selection import select_representative
from .screening import screen_outlier_signals
from .features import extract_features
from .model import (build_model, aggregate_by_sample, report,
                    stratified_split, score_to_class)


# ================== 课程学习: 逐步增加训练集 ==================
def run_curriculum():
    """课程学习: 每部位敲击按代表性排名从 1 条逐步增至 KNOCKS_PER_POSITION 条, 每轮 warm-start 续训。

    测试集固定使用每部位 top-1 代表信号(4 条/样本), 训练集随轮次逐步扩大。
    输出 curriculum_metrics.csv 记录每轮训练/测试准确率。
    """
    samples = load_dataset_structured()
    sids = sorted(samples.keys())
    label_map = {sid: samples[sid]["label"] for sid in sids}

    sig_list, sid_list, rank_list = [], [], []
    skipped = 0
    for sid in sids:
        s = samples[sid]
        buf = []
        ok = True
        for pos in config.POSITIONS:
            knocks = s["positions"].get(pos)
            if knocks is None or len(knocks) < config.MIN_KNOCKS:
                ok = False
                break
            res = select_representative(sid, pos, knocks[:config.KNOCKS_PER_POSITION], config.SR)
            for i, d in enumerate(res["diagnostics"]):
                buf.append((knocks[i], sid, int(d["rank"])))
        if not ok:
            skipped += 1
            continue
        for sig, sid_, rank_ in buf:
            sig_list.append(sig)
            sid_list.append(sid_)
            rank_list.append(rank_)

    X_all = np.array(sig_list)
    sid_all = np.array(sid_list)
    rank_all = np.array(rank_list)
    y_all = np.array([label_map[s] for s in sid_all])
    print(f"课程学习: {len(sids)} 颗榴莲, 共 {len(X_all)} 条敲击信号(含排名)"
          + (f", 跳过 {skipped} 颗" if skipped else ""))

    train_sids, val_sids = stratified_split(sid_all, label_map, config.TEST_SIZE, config.SEED)
    tr_base = np.array([s in train_sids for s in sid_all])
    va_base = np.array([s in val_sids for s in sid_all])

    print("提取特征中...")
    X_feat = extract_features(X_all)
    print(f"特征维度: {X_feat.shape}")

    scaler = RobustScaler().fit(X_feat[tr_base])

    va_sel = va_base & (rank_all == 1)
    X_va = scaler.transform(X_feat[va_sel])
    y_va = y_all[va_sel]
    g_va = sid_all[va_sel]

    rows = []
    model = None
    for r in range(1, config.KNOCKS_PER_POSITION + 1):
        tr_sel = tr_base & (rank_all <= r)
        X_tr = scaler.transform(X_feat[tr_sel])
        y_tr = y_all[tr_sel]
        g_tr = sid_all[tr_sel]

        m = LGBMRegressor(**config.LGB_REGR_PARAMS)
        m.fit(X_tr, y_tr, init_model=model)
        model = m

        tr_agg = aggregate_by_sample(m.predict(X_tr), g_tr, y_tr)
        va_agg = aggregate_by_sample(m.predict(X_va), g_va, y_va)
        tr_acc = accuracy_score(tr_agg["true_class"], tr_agg["pred_class"])
        va_acc = accuracy_score(va_agg["true_class"], va_agg["pred_class"])
        rows.append({
            "round": r,
            "knocks_per_position": r,
            "train_signals": int(tr_sel.sum()),
            "train_acc": round(tr_acc, 4),
            "val_acc": round(va_acc, 4),
        })
        print(f"Round {r}: 每部位 {r} 条敲击, 训练信号 {int(tr_sel.sum())}, "
              f"训练acc {tr_acc:.4f}, 测试acc {va_acc:.4f}")

    curriculum_df = pd.DataFrame(rows)
    curriculum_df.to_csv(os.path.join(config.OUTPUT_DIR, "curriculum_metrics.csv"),
                         index=False, encoding="utf-8-sig")
    print("\n课程学习各轮指标:")
    print(curriculum_df.to_string(index=False))

    joblib.dump(model, config.MODEL_PATH)
    va_df = aggregate_by_sample(model.predict(X_va), g_va, y_va)
    va_df["true_name"] = va_df["true_class"].map(config.CLASS_NAMES)
    va_df["pred_name"] = va_df["pred_class"].map(config.CLASS_NAMES)
    va_df.to_csv(config.VAL_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n课程学习结果已保存: curriculum_metrics.csv")
    print(f"模型已保存: {config.MODEL_PATH}")
    print(f"测试集结果已保存: {config.VAL_RESULT_PATH}")


# ================== 主流程 ==================
def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {config.OUTPUT_DIR}")

    if config.CURRICULUM:
        run_curriculum()
        return

    # ===== 1. 数据加载 + 前处理(代表性敲击选择 4×5 -> 4×1) =====
    if config.REP_SELECT:
        samples = load_dataset_structured()
        sids = sorted(samples.keys())
        X_signals, groups, y = [], [], []
        diag_rows, sel_rows = [], []
        skipped = 0
        for sid in sids:
            s = samples[sid]
            pos_signals = []
            ok = True
            for pos in config.POSITIONS:
                knocks = s["positions"].get(pos)
                if knocks is None or len(knocks) < config.MIN_KNOCKS:
                    ok = False
                    break
                res = select_representative(sid, pos, knocks[:config.KNOCKS_PER_POSITION], config.SR)
                diag_rows.extend(res["diagnostics"])
                sel_rows.append({"sample_id": sid, "position": pos,
                                 "selected_knock": res["selected_knock"],
                                 "selected_score": res["selected_score"]})
                pos_signals.append(res["selected_signal"])
            if not ok:
                skipped += 1
                continue
            X_signals.extend(pos_signals)
            groups.extend([sid] * len(pos_signals))
            y.extend([s["label"]] * len(pos_signals))
        X = np.array(X_signals)
        groups = np.array(groups)
        y = np.array(y)
        label_map = {sid: samples[sid]["label"] for sid in sids}
        print(f"代表性敲击选择: {len(sids)} 颗榴莲 -> {len(X)} 条代表信号 (4×1/样本)"
              + (f", 跳过 {skipped} 颗" if skipped else ""))
        pd.DataFrame(sel_rows).to_csv(os.path.join(config.OUTPUT_DIR, "selected_samples.csv"),
                                      index=False, encoding="utf-8-sig")
        pd.DataFrame(diag_rows).to_csv(os.path.join(config.OUTPUT_DIR, "selection_diagnostics.csv"),
                                       index=False, encoding="utf-8-sig")
    else:
        X, groups, y, label_map = load_data()

    # ===== 2. 7:3 按样本分层划分 =====
    train_sids, val_sids = stratified_split(groups, label_map, config.TEST_SIZE, config.SEED)
    tr_idx = np.array([i for i, g in enumerate(groups) if g in train_sids])
    va_idx = np.array([i for i, g in enumerate(groups) if g in val_sids])
    print(f"\n训练集: {len(train_sids)} 样本({len(tr_idx)} 条信号), "
          f"测试集: {len(val_sids)} 样本({len(va_idx)} 条信号)")

    # ===== 3. 离群信号筛选(仅训练集) =====
    if config.SCREEN_SIGNALS:
        keep_tr, vote, mel_dist = screen_outlier_signals(X[tr_idx])
        n_removed = int((~keep_tr).sum())
        removed_idx = tr_idx[~keep_tr]
        pd.DataFrame({"sample_id": groups[removed_idx], "signal_idx": removed_idx,
                      "vote": vote[~keep_tr], "mel_distance": mel_dist[~keep_tr]}).to_csv(
            os.path.join(config.OUTPUT_DIR, "screened_signals.csv"), index=False, encoding="utf-8-sig")
        tr_idx = tr_idx[keep_tr]
        print(f"离群信号筛选(梅尔频谱): 训练集剔除 {n_removed} 条离群信号, 剩余 {len(tr_idx)} 条")

    print("训练集标签分布:", {int(k): int(v) for k, v in pd.Series(
        [label_map[s] for s in sorted(set(groups[tr_idx].tolist()))]).value_counts().sort_index().items()})
    print("测试集标签分布:", {int(k): int(v) for k, v in pd.Series(
        [label_map[s] for s in sorted(set(groups[va_idx].tolist()))]).value_counts().sort_index().items()})

    # ===== 4. 特征提取(只提取一次) =====
    print("\n提取特征中...")
    X_feat = extract_features(X)
    print(f"特征维度: {X_feat.shape}")

    # ===== 5. 训练模型 =====
    model = build_model()
    model.fit(X_feat[tr_idx], y[tr_idx])

    for name, idx in [("训练集", tr_idx), ("测试集", va_idx)]:
        scores = model.predict(X_feat[idx])
        report(aggregate_by_sample(scores, groups[idx], y[idx]), name)

    # ===== 6. 保存模型与结果 =====
    joblib.dump(model, config.MODEL_PATH)
    print(f"\n模型已保存: {config.MODEL_PATH}")

    va_scores = model.predict(X_feat[va_idx])
    va_df = aggregate_by_sample(va_scores, groups[va_idx], y[va_idx])
    va_df["true_name"] = va_df["true_class"].map(config.CLASS_NAMES)
    va_df["pred_name"] = va_df["pred_class"].map(config.CLASS_NAMES)
    va_df.to_csv(config.VAL_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"测试集结果已保存: {config.VAL_RESULT_PATH}")


__all__ = ["run_curriculum", "main"]
