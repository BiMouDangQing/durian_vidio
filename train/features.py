# -*- coding: utf-8 -*-
"""声学特征提取: 训练用特征 + 代表性选择用单次敲击特征。"""

import numpy as np
import librosa
from scipy import stats
from scipy.signal import hilbert

from . import config
from .preprocess import (preprocess, envelope, normalized_psd,
                         zero_crossing_rate, active_duration, signal_snr,
                         _band_energy)

SR = config.SR


# ================== 特征提取(与训练/预测一致) ==================
def extract_features(X):
    feats = []
    for x in X:
        x = preprocess(x)

        mel = librosa.feature.melspectrogram(
            y=x, sr=SR, n_mels=64, fmin=50, fmax=4000,
            n_fft=256, hop_length=64
        )
        mel_db = librosa.power_to_db(mel)

        mfcc = librosa.feature.mfcc(
            y=x, sr=SR, n_mfcc=20, n_fft=256, hop_length=64
        )

        feat = np.concatenate([
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1),
            np.max(mel_db, axis=1),

            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),

            [
                np.mean(x), np.std(x), np.max(x), np.min(x),
                np.ptp(x), np.sqrt(np.mean(x ** 2))
            ]
        ])
        feats.append(feat)
    return np.array(feats)


# ================== 单次敲击声学特征(仅用于代表性评分) ==================
def extract_single_features(x, sr):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim > 1:
        x = x.ravel()
    f = {}

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    energy = float(np.sum(x ** 2))
    f["peak"] = peak
    f["rms"] = rms
    f["energy"] = energy
    f["crest_factor"] = peak / (rms + 1e-8)

    xn = x / (np.max(np.abs(x)) + 1e-8)
    f["skewness"] = float(stats.skew(xn))
    f["kurtosis"] = float(stats.kurtosis(xn))
    f["zcr"] = zero_crossing_rate(xn)

    env = np.abs(hilbert(xn))
    env = env / (env.max() + 1e-8)
    env_peak = env.max()
    peak_idx = int(np.argmax(env))

    seg = np.where(env[:peak_idx + 1] >= 0.1 * env_peak)[0]
    rise_idx = int(seg[0]) if len(seg) else 0
    f["rise_time"] = (peak_idx - rise_idx) / sr

    after = np.where(env[peak_idx:] >= 0.1 * env_peak)[0]
    decay_idx = peak_idx + int(after[-1]) if len(after) else peak_idx
    f["decay_time"] = (decay_idx - peak_idx) / sr
    f["env_peak"] = env_peak
    f["env_t10"] = f["decay_time"]

    seg_env = env[peak_idx:decay_idx + 1]
    if len(seg_env) > 2:
        tt = np.arange(len(seg_env)) / sr
        slope = np.polyfit(tt, np.log(seg_env + 1e-12), 1)[0]
    else:
        slope = 0.0
    f["env_slope"] = float(slope)

    freqs, psd = normalized_psd(x, sr)
    centroid = float(np.sum(freqs * psd))
    f["dominant_freq"] = float(freqs[np.argmax(psd)])
    f["spectral_centroid"] = centroid
    f["bandwidth"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd)))
    f["spectral_entropy"] = float(-np.sum(psd * np.log2(psd + 1e-12)))

    lo = _band_energy(freqs, psd, 50, 500)
    mid = _band_energy(freqs, psd, 500, 2000)
    hi = _band_energy(freqs, psd, 2000, 4000)
    total = lo + mid + hi + 1e-12
    f["low_ratio"] = lo / total
    f["mid_ratio"] = mid / total
    f["high_ratio"] = hi / total

    return f


# ================== 单条信号梅尔频谱(dB) ==================
def compute_mel_db(x):
    """计算单条音频信号的梅尔频谱(dB), 与 extract_features 的预处理保持一致。"""
    x = preprocess(x)
    mel = librosa.feature.melspectrogram(
        y=x, sr=SR, n_mels=64, fmin=50, fmax=4000, n_fft=256, hop_length=64)
    return librosa.power_to_db(mel)


__all__ = ["extract_features", "extract_single_features", "compute_mel_db"]
