# -*- coding: utf-8 -*-
"""Qt 前端包: 声音采集工具。

导入本包时自动把 fqb 目录加入 sys.path,
使得 tools / train / predict 模块可被本包内各子模块引用。
"""
import os
import sys

FQB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if FQB_DIR not in sys.path:
    sys.path.insert(0, FQB_DIR)
