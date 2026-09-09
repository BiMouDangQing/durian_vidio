# -*- coding: utf-8 -*-
"""离群筛选: 样本级三路投票 + 信号级梅尔频谱筛选。"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN

from . import config
from .features import compute_mel_db


# ================== 无监督异常数据筛选(样本级, 已弃用) ==================
def screen_outliers(X_feat, groups, y):
    """三路投票(IF+LOF+DBSCAN噪声)筛选异常样本, 返回 {sample_id: 投票数}"""
    df = pd.DataFrame(X_feat)
    df["sid"] = groups
    df["y"] = y
    agg = df.groupby("sid").mean()
    sids = agg.index.values
    X_s = StandardScaler().fit_transform(agg.drop(columns=["y"]).values)

    iso = IsolationForest(random_state=config.SEED, contamination=config.CLEAN_CONTAMINATION).fit(X_s)
    iso_flag = iso.predict(X_s) == -1
    lof_pred = LocalOutlierFactor(n_neighbors=20, contamination=config.CLEAN_CONTAMINATION).fit_predict(X_s)
    lof_flag = lof_pred == -1
    nn = NearestNeighbors(n_neighbors=min(5, len(X_s) - 1)).fit(X_s)
    dist, _ = nn.kneighbors(X_s)
    eps = float(np.quantile(dist[:, -1], 0.90))
    db_flag = DBSCAN(eps=eps, min_samples=5).fit_predict(X_s) == -1

    vote = iso_flag.astype(int) + lof_flag.astype(int) + db_flag.astype(int)
    return {int(s): int(v) for s, v in zip(sids, vote)}


# ================== 信号级离群筛选(基于梅尔频谱) ==================
def screen_outlier_signals(X_raw):
    """基于梅尔频谱的离群信号筛选(音频层面, 不涉及样本剔除):
    1. 计算每条信号的梅尔频谱(dB)并展平;
    2. 以训练集梅尔频谱中位数为参考, 计算每条信号的频谱距离(可解释);
    3. IF + LOF + DBSCAN噪声 三路投票(作用于标准化后的展平梅尔频谱)。
    返回 (保留掩码, 投票数, 梅尔频谱距离)。
    """
    mel_mat = np.array([compute_mel_db(x).ravel() for x in X_raw])
    med = np.median(mel_mat, axis=0)
    mel_dist = np.mean(np.abs(mel_mat - med), axis=1)

    X_s = StandardScaler().fit_transform(mel_mat)
    iso = IsolationForest(random_state=config.SEED, contamination=config.SIGNAL_CONTAMINATION).fit(X_s)
    iso_flag = iso.predict(X_s) == -1
    lof_pred = LocalOutlierFactor(n_neighbors=20, contamination=config.SIGNAL_CONTAMINATION).fit_predict(X_s)
    lof_flag = lof_pred == -1
    nn = NearestNeighbors(n_neighbors=min(5, len(X_s) - 1)).fit(X_s)
    dist, _ = nn.kneighbors(X_s)
    eps = float(np.quantile(dist[:, -1], 0.90))
    db_flag = DBSCAN(eps=eps, min_samples=5).fit_predict(X_s) == -1

    vote = iso_flag.astype(int) + lof_flag.astype(int) + db_flag.astype(int)
    keep = vote < config.SIGNAL_VOTE_REMOVE
    return keep, vote, mel_dist


__all__ = ["screen_outliers", "screen_outlier_signals"]
