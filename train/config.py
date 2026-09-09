# -*- coding: utf-8 -*-
"""训练配置: 路径、阈值、模型参数、代表性选择与筛选权重。

所有训练相关的全局常量集中在此, 各子模块通过 `from . import config`
访问 `config.XXX`, 保证运行时改写的变量(如 DATA_PATH)能被所有模块读到。
"""

import os
import datetime

# ================== 路径与输出 ==================
SR = 16000

DATA_PATH = r"C:\Users\reemoon\Desktop\data_test\data_0830"
LABEL_PATH = r"C:\Users\reemoon\Desktop\data_test\data_0830\label.csv"

# 项目根目录(fqb), 输出目录相对它归档
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输出目录: 按日期时间归档到 fqb/results/train/<时间戳>/
# 注意: RUN_TIME / OUTPUT_DIR / MODEL_PATH / VAL_RESULT_PATH 运行时可由外部(Qt)改写,
#       请用 `train.config.RUN_TIME = ...` 的方式改写, 保证各子模块读到一致值。
RUN_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "train", RUN_TIME)
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
VAL_RESULT_PATH = os.path.join(OUTPUT_DIR, "val_result.csv")

# ================== 数据 ==================
FILES_PER_SAMPLE = 4
SIGNALS_PER_FILE = 5
TEST_SIZE = 0.3          # 测试集比例: 0.3 = 训练70% : 测试30%(7:3)
SEED = 42

# 样本级异常筛选(已弃用: 本项目只剔除离群"信号"而非"样本", 保持 False)
CLEAN_DATA = False
CLEAN_VOTE_REMOVE = 2
CLEAN_CONTAMINATION = 0.1

# 三分类阈值(回归连续分数 -> 类别)
THRESH_1_2 = 1.7
THRESH_2_3 = 2.2
CLASS_NAMES = {1: "生果", 2: "中间果", 3: "熟果"}

# ================== 前处理开关 ==================
REP_SELECT = True        # 是否先做 4×5 -> 4×1 代表性敲击选择
SCREEN_SIGNALS = True    # 是否对训练集做离群信号筛选
SIGNAL_CONTAMINATION = 0.1   # IF/LOF 期望离群比例
SIGNAL_VOTE_REMOVE = 2       # IF+LOF+DBSCAN噪声 投票 >= 此值 -> 剔除该信号
CURRICULUM = False           # 课程学习(每部位敲击按代表性排名逐步增加)

POSITIONS = ["01", "02", "03", "04"]     # 4 个敲击部位(data_all 命名)
KNOCKS_PER_POSITION = 5
MIN_KNOCKS = 3

# ================== 代表性选择权重 ==================
W_EXC_PEAK = 0.15
W_EXC_RMS = 0.15
W_EXC_ENERGY = 0.15
W_EXC_ACTIVE = 0.25
W_EXC_SNR = 0.30
W_EXC = 0.30
W_TIME = 0.25
W_SPEC = 0.25
W_FEAT = 0.20
W_REC = 0.05
W_ANOM = 0.50
RECENCY_PRIOR = [0.0, 0.0, 0.0, 0.5, 1.0]
ANOM_PEAK_Z = 3.0
ANOM_CREST_Z = 3.0
ANOM_SPEC_TH = 0.5
WELCH_NPERSEG = 256

# ================== 模型参数 ==================
LGB_REGR_PARAMS = dict(
    objective="regression_l2",
    n_estimators=150,
    learning_rate=0.03,
    num_leaves=8,
    max_depth=4,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=10.0,
    random_state=SEED,
    verbosity=-1,
)

__all__ = [
    "SR", "DATA_PATH", "LABEL_PATH", "PROJECT_ROOT", "RUN_TIME", "OUTPUT_DIR",
    "MODEL_PATH", "VAL_RESULT_PATH", "FILES_PER_SAMPLE", "SIGNALS_PER_FILE",
    "TEST_SIZE", "SEED", "CLEAN_DATA", "CLEAN_VOTE_REMOVE", "CLEAN_CONTAMINATION",
    "THRESH_1_2", "THRESH_2_3", "CLASS_NAMES", "REP_SELECT", "SCREEN_SIGNALS",
    "SIGNAL_CONTAMINATION", "SIGNAL_VOTE_REMOVE", "CURRICULUM", "POSITIONS",
    "KNOCKS_PER_POSITION", "MIN_KNOCKS", "W_EXC_PEAK", "W_EXC_RMS", "W_EXC_ENERGY",
    "W_EXC_ACTIVE", "W_EXC_SNR", "W_EXC", "W_TIME", "W_SPEC", "W_FEAT", "W_REC",
    "W_ANOM", "RECENCY_PRIOR", "ANOM_PEAK_Z", "ANOM_CREST_Z", "ANOM_SPEC_TH",
    "WELCH_NPERSEG", "LGB_REGR_PARAMS",
]
