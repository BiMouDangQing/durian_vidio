# -*- coding: utf-8 -*-
"""训练包: 汇总导出各子模块, 保持 `import train as T` 的兼容接口。

模块划分:
- config.py      参数常量(路径/阈值/权重/模型参数)
- preprocess.py  滤波、包络、PSD、瞬态指标
- features.py    声学特征提取
- data.py        数据加载
- selection.py   代表性敲击选择
- screening.py   离群筛选
- model.py       模型/聚合/报告/分层划分
- ordinal.py     有序分类(两个二分类器)
- pipeline.py    训练主流程

用法:
    import train as T
    T.main()                           # 训练
    T.extract_features(X)              # 特征
    T.config.DATA_PATH = "..."         # 运行时改数据路径
"""

from . import config
from .config import *
from .preprocess import *
from .features import *
from .data import *
from .selection import *
from .screening import *
from .model import *
from .ordinal import *
from .pipeline import *

PROJECT_ROOT = config.PROJECT_ROOT
