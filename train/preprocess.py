# -*- coding: utf-8 -*-
"""信号预处理: 滤波、包络、PSD、瞬态指标与通用小工具。"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, welch

from . import config

SR = config.SR


# ================== 滤波 ==================
def bandpass_filter(x, sr, low=50, high=4000, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


# ================== 预处理 ==================
def preprocess(x):
    x = bandpass_filter(x, SR)
    x = x[200:800]
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-8)
    x = x / (np.max(np.abs(x)) + 1e-8)
    return x


def preprocess_signal(x, sr, normalize=True):
    """代表帧选择用预处理: 去均值 + 带通滤波 + (可选) 幅值归一化。"""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim > 1:
        x = x.ravel()
    x = x - np.mean(x)
    if len(x) > 24:
        try:
            x = bandpass_filter(x, sr)
        except Exception:
            pass
        x = x - np.mean(x)
    if normalize:
        s = np.std(x)
        if s > 1e-12:
            x = x / s
        m = np.max(np.abs(x))
        if m > 1e-12:
            x = x / m
    return x


def envelope(x):
    e = np.abs(hilbert(x))
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    m = e.max()
    if m > 1e-12:
        e = e / m
    return e


def normalized_psd(x, sr):
    nperseg = min(config.WELCH_NPERSEG, len(x))
    freqs, psd = welch(x, sr, nperseg=nperseg)
    psd = np.asarray(psd, dtype=np.float64) + 1e-12
    psd = psd / psd.sum()
    return freqs, psd


def zero_crossing_rate(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    s = np.sign(x)
    s = s[s != 0]
    if len(s) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(s)) > 0)) / (len(x) - 1)


def active_duration(env, sr, rel=0.1):
    pk = env.max()
    idx = np.where(env >= rel * pk)[0]
    if len(idx) == 0:
        return 0.0
    return (idx[-1] - idx[0]) / sr


def signal_snr(x, env, rel=0.1):
    pk = env.max()
    idx = np.where(env >= rel * pk)[0]
    if len(idx) == 0:
        return 0.0
    lo, hi = idx[0], idx[-1]
    sig_power = np.mean(x[lo:hi + 1] ** 2)
    tail = x[hi + 1:]
    noise_power = np.mean(tail ** 2) if len(tail) > 5 else 0.0
    return 10.0 * np.log10((sig_power + 1e-12) / (noise_power + 1e-12))


def _band_energy(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.sum(psd[m]))


# ================== 通用小工具 ==================
def _read_csv_any(path):
    """兼容 utf-8-sig / utf-8 / gbk 的 CSV 读取。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, header=None, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, header=None, encoding="latin1")


def _minmax01(vals):
    v = np.asarray(vals, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    span = v.max() - v.min()
    if span < 1e-12:
        return np.full_like(v, 0.5)
    return (v - v.min()) / span


def _mad(vals):
    v = np.asarray(vals, dtype=np.float64)
    return np.median(np.abs(v - np.median(v))) + 1e-12


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


__all__ = [
    "bandpass_filter", "preprocess", "preprocess_signal", "envelope",
    "normalized_psd", "zero_crossing_rate", "active_duration", "signal_snr",
]
