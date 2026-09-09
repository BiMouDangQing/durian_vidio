# -*- coding: utf-8 -*-
"""数字信号处理模块: 滤波、降噪、归一化、包络。"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, iirnotch, medfilt, wiener, hilbert

# ================== 常量 ==================
SR = 16000
LOW_CUT = 50
HIGH_CUT = 700
NOISE_PERCENTILE = 20
NOISE_K = 3.0
NOTCH_FREQS = [50, 100, 150, 200, 250, 300]
NOTCH_Q = 30.0
SPEC_SUB_ALPHA = 1.5
SPEC_SUB_BETA = 0.05
N_FFT, HOP = 256, 64


# ================== 滤波 ==================
def bandpass(x, sr, low=LOW_CUT, high=HIGH_CUT, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def notch_filter(x, sr, freqs, q=NOTCH_Q):
    x = np.asarray(x, dtype=np.float64)
    for f0 in freqs:
        if f0 >= sr / 2:
            continue
        try:
            b, a = iirnotch(f0, q, sr)
            x = filtfilt(b, a, x)
        except Exception:
            pass
    return x


def spectral_subtraction(x, sr, alpha=SPEC_SUB_ALPHA, beta=SPEC_SUB_BETA):
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(D), np.angle(D)
    noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)
    mag_sub = np.maximum(mag - alpha * noise_mag, beta * mag)
    D_sub = mag_sub * np.exp(1j * phase)
    x_sub = librosa.istft(D_sub, hop_length=HOP, length=len(x))
    return np.asarray(x_sub, dtype=np.float64)


def noise_gate(x, percentile=NOISE_PERCENTILE, k=NOISE_K):
    noise_floor = float(np.percentile(np.abs(x), percentile))
    thresh = k * noise_floor
    return np.where(np.abs(x) < thresh, 0.0, x)


def median_filter(x, kernel=3):
    """中值滤波: 去除脉冲噪声/尖峰毛刺。"""
    return medfilt(np.asarray(x, dtype=np.float64), kernel_size=kernel)


def wiener_filter(x, mysize=5):
    """维纳滤波: 自适应统计去噪(最小均方误差)。"""
    return wiener(np.asarray(x, dtype=np.float64), mysize=mysize)


def moving_average(x, window=5):
    """移动平均平滑: 去除高频毛刺。"""
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def denoise(x, sr, use_notch=True, use_bandpass=True, use_specsub=True, use_gate=True,
            gate_k=NOISE_K, use_median=False, use_wiener=False, use_smooth=False):
    """按开关组合的降噪链路。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) > 24:
        try:
            if use_notch:
                x = notch_filter(x, sr, NOTCH_FREQS)
            if use_bandpass:
                x = bandpass(x, sr)
            if use_median:
                x = median_filter(x)
            if use_wiener:
                x = wiener_filter(x)
            if use_smooth:
                x = moving_average(x)
            if use_specsub:
                x = spectral_subtraction(x, sr)
        except Exception:
            pass
    if use_gate:
        x = noise_gate(x, k=gate_k)
    return x


def normalize(x):
    """归一化: 去均值 + 幅值归一化到 [-1, 1]。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 1e-8:
        x = x / peak
    return x


def envelope(x):
    """希尔伯特包络(幅度)。"""
    return np.abs(hilbert(np.asarray(x, dtype=np.float64)))
