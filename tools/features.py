# -*- coding: utf-8 -*-
"""特征提取模块: 频谱、梅尔谱、MFCC、声学特征。"""

import numpy as np
import librosa
from scipy import stats
from scipy.signal import hilbert

from .dsp import SR, N_FFT, HOP

N_MELS = 64
N_MFCC = 20
N_CQT = 84
N_LOG_SPEC = 64


def compute_spectrogram(x, sr):
    D = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    return librosa.amplitude_to_db(D, ref=np.max)


def compute_mel(x, sr):
    mel = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=N_MELS, fmin=50, fmax=4000,
        n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(mel)


def compute_mfcc(x, sr):
    mfcc = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, fmax=4000)
    return mfcc.mean(axis=1)          # (20,)


def compute_cqt(x, sr):
    """常数Q变换谱图(对数频率轴, 50Hz~6.4kHz), dB。"""
    import warnings
    with warnings.catch_warnings():
        # CQT 高频 bin 在短帧下会有 n_fft 过大的警告, 属正常现象(敲击声主要看低频)
        warnings.simplefilter("ignore")
        cqt = np.abs(librosa.cqt(
            y=x, sr=sr, n_bins=N_CQT, bins_per_octave=12, fmin=50, hop_length=HOP))
    return librosa.amplitude_to_db(cqt, ref=np.max)


def compute_log_spec(x, sr, n_bins=N_LOG_SPEC, fmin=50, fmax=4000):
    """对数频率 STFT 谱图: 对线性 FFT 频谱按对数频率重采样, dB。

    比 CQT 更快, 频率轴对数分布(50Hz~4kHz), 低频细节放大。
    """
    D = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    log_freqs = np.geomspace(fmin, fmax, n_bins)
    log_spec = np.zeros((n_bins, D.shape[1]))
    for t in range(D.shape[1]):
        log_spec[:, t] = np.interp(log_freqs, freqs, D[:, t])
    return librosa.amplitude_to_db(log_spec, ref=np.max)


def compute_energy_envelope(x, sr):
    """短时能量包络(RMS 随时间), 一维。"""
    rms = librosa.feature.rms(
        y=x, frame_length=N_FFT, hop_length=HOP)[0]
    return rms


def compute_centroid_trace(x, sr):
    """谱质心随时间(Hz), 一维。"""
    cent = librosa.feature.spectral_centroid(
        y=x, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    return cent


def compute_spectral_flux(x, sr):
    """谱通量(频谱随时间变化率), 一维。"""
    D = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    flux = np.sqrt(np.sum(np.diff(D, axis=1) ** 2, axis=0))
    return np.concatenate([[0.0], flux])


def compute_features(x, sr):
    """提取关键声学特征(与训练特征对齐), 返回 dict。"""
    f = {}
    xn = np.asarray(x, dtype=np.float64)
    xn = xn - np.mean(xn)

    # 时域
    f["峰值"] = float(np.max(np.abs(xn)))
    f["RMS"] = float(np.sqrt(np.mean(xn ** 2)))
    f["波峰因子"] = f["峰值"] / (f["RMS"] + 1e-8)
    f["偏度"] = float(stats.skew(xn))
    f["峰度"] = float(stats.kurtosis(xn))
    f["过零率"] = float(librosa.feature.zero_crossing_rate(xn)[0].mean())

    # 包络瞬态(敲击起振/衰减, 与成熟度相关)
    env = np.abs(hilbert(xn))
    env = env / (env.max() + 1e-8)
    peak_idx = int(np.argmax(env))
    rise = np.where(env[:peak_idx + 1] >= 0.1)[0]
    rise_idx = int(rise[0]) if len(rise) else 0
    f["上升时间ms"] = (peak_idx - rise_idx) / sr * 1000
    after = np.where(env[peak_idx:] <= 0.1)[0]
    decay_idx = peak_idx + (int(after[0]) if len(after) else len(xn) - 1 - peak_idx)
    f["衰减时间ms"] = (decay_idx - peak_idx) / sr * 1000

    # 频域
    D = np.abs(librosa.stft(xn, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    f["谱质心Hz"] = float(librosa.feature.spectral_centroid(
        y=xn, sr=sr, n_fft=N_FFT, hop_length=HOP).mean())
    f["谱带宽Hz"] = float(librosa.feature.spectral_bandwidth(
        y=xn, sr=sr, n_fft=N_FFT, hop_length=HOP).mean())
    f["谱滚降Hz"] = float(librosa.feature.spectral_rolloff(
        y=xn, sr=sr, n_fft=N_FFT, hop_length=HOP, roll_percent=0.85).mean())
    f["谱平坦度"] = float(librosa.feature.spectral_flatness(
        y=xn, n_fft=N_FFT, hop_length=HOP).mean())
    psd = D / (D.sum(axis=0, keepdims=True) + 1e-12)
    f["谱熵"] = float(-np.sum(psd * np.log2(psd + 1e-12), axis=0).mean())
    f["谱对比度"] = float(librosa.feature.spectral_contrast(
        y=xn, sr=sr, n_fft=N_FFT, hop_length=HOP).mean())

    bands = [("低频比", 50, 500), ("中频比", 500, 2000), ("高频比", 2000, 4000)]
    energies = []
    for _, lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        energies.append(float(np.sum(D[m])))
    total = sum(energies) + 1e-12
    for (name, _, _), e in zip(bands, energies):
        f[name] = e / total
    f["主频Hz"] = float(freqs[int(np.argmax(np.mean(D, axis=1)))])
    return f
