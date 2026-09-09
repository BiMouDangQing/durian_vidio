# -*- coding: utf-8 -*-
"""预测包: 模型加载与成熟度预测。"""

from .loader import load_model
from .inference import predict_class

__all__ = ["load_model", "predict_class"]
