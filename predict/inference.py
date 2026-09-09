# -*- coding: utf-8 -*-
"""成熟度预测: 信号级特征 -> 样本级众数聚合。"""

import numpy as np
import pandas as pd


def predict_class(model, knock_list, train_module):
    """对多次敲击信号提取特征并预测样本级类别(众数聚合)。

    兼容:
    - 普通多分类(有 predict_proba): 直接输出类别;
    - 回归(无 predict_proba): 连续分数 -> score_to_class 阈值映射;
    - 有序分类(tuple 两个二分类器): 累积概率决策。
    """
    feats = train_module.extract_features(np.vstack(knock_list))
    if isinstance(model, tuple):
        clf1, clf2 = model
        p1 = clf1.predict_proba(feats)[:, 1]
        p2 = clf2.predict_proba(feats)[:, 1]
        preds = np.array([train_module.ordinal_decision(float(a), float(b))
                          for a, b in zip(p1, p2)])
    elif hasattr(model, "predict_proba"):
        preds = np.asarray(model.predict(feats)).ravel()
    else:
        scores = np.asarray(model.predict(feats)).ravel()
        preds = np.array([train_module.score_to_class(float(s)) for s in scores])
    return int(pd.Series(preds.astype(int)).mode().iloc[0])
