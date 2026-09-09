# -*- coding: utf-8 -*-
"""代表性敲击选择: 从某部位 K 次敲击中选择最具代表性的 1 次。"""

import numpy as np
from scipy.stats import rankdata

from . import config
from .preprocess import (preprocess_signal, envelope, normalized_psd,
                         active_duration, signal_snr,
                         _minmax01, _mad, _pearson)
from .features import extract_single_features


def select_representative(sid, pos, knocks, sr):
    """从某部位 K 次敲击中选择最具代表性的 1 次(返回诊断明细 + 被选信号)。"""
    K = len(knocks)
    proc = []
    for k in range(K):
        x = knocks[k]
        xp = preprocess_signal(x, sr, normalize=True)
        xf = preprocess_signal(x, sr, normalize=False)
        env = envelope(xp)
        freqs, psd = normalized_psd(xf, sr)
        feat = extract_single_features(xf, sr)
        proc.append({"xp": xp, "env": env, "psd": psd, "feat": feat})

    peaks = np.array([p["feat"]["peak"] for p in proc])
    rms = np.array([p["feat"]["rms"] for p in proc])
    energies = np.array([p["feat"]["energy"] for p in proc])
    actives = np.array([active_duration(p["env"], sr) for p in proc])
    snrs = np.array([signal_snr(p["xp"], p["env"]) for p in proc])

    exc_peak = _minmax01(peaks)
    exc_rms = _minmax01(rms)
    exc_energy = _minmax01(energies)
    exc_active = _minmax01(actives)
    exc_snr = _minmax01(snrs)
    excitation = (config.W_EXC_PEAK * exc_peak + config.W_EXC_RMS * exc_rms +
                  config.W_EXC_ENERGY * exc_energy + config.W_EXC_ACTIVE * exc_active +
                  config.W_EXC_SNR * exc_snr)

    corr = np.zeros(K)
    env_sim = np.zeros(K)
    for i in range(K):
        c, e = [], []
        for j in range(K):
            if i == j:
                continue
            c.append(_pearson(proc[i]["xp"], proc[j]["xp"]))
            e.append(_pearson(proc[i]["env"], proc[j]["env"]))
        corr[i] = np.mean(c)
        env_sim[i] = np.mean(e)
    corr_clip = np.clip(_minmax01(corr), 0.0, 1.0)
    env_sim_clip = np.clip(_minmax01(env_sim), 0.0, 1.0)
    time_stab = 0.6 * corr_clip + 0.4 * env_sim_clip

    psd_mat = np.vstack([p["psd"] for p in proc])
    median_psd = np.median(psd_mat, axis=0)
    spec_sim = np.zeros(K)
    for i in range(K):
        a = proc[i]["psd"]
        spec_sim[i] = float(np.dot(a, median_psd) /
                            (np.linalg.norm(a) * np.linalg.norm(median_psd) + 1e-12))
    spec_stab = np.clip(spec_sim, 0.0, 1.0)

    feat_keys = ["dominant_freq", "spectral_centroid", "bandwidth",
                 "spectral_entropy", "env_t10"]
    feat_mat = np.array([[p["feat"][k] for k in feat_keys] for p in proc])
    med_feat = np.median(feat_mat, axis=0)
    scale = np.std(feat_mat, axis=0) + 1e-12
    dist = np.mean(np.abs((feat_mat - med_feat) / scale), axis=1)
    feat_stab = np.exp(-dist)

    recency = np.array(config.RECENCY_PRIOR[:K], dtype=np.float64)

    crests = np.array([p["feat"]["crest_factor"] for p in proc])
    z_peak = np.abs((peaks - np.median(peaks)) / _mad(peaks))
    z_crest = np.abs((crests - np.median(crests)) / _mad(crests))
    anomaly = np.zeros(K)
    anomaly += np.clip(z_peak - config.ANOM_PEAK_Z, 0.0, None)
    anomaly += np.clip(z_crest - config.ANOM_CREST_Z, 0.0, None)
    anomaly += np.clip(config.ANOM_SPEC_TH - spec_sim, 0.0, None) * 2.0

    total = (config.W_EXC * excitation + config.W_TIME * time_stab +
             config.W_SPEC * spec_stab + config.W_FEAT * feat_stab +
             config.W_REC * recency - config.W_ANOM * anomaly)

    selected_idx = int(np.argmax(total))
    ranks = rankdata(-total, method="min").astype(int)

    diagnostics = []
    for i in range(K):
        row = {
            "sample_id": sid, "position": pos, "knock_idx": i + 1,
            "exc_peak": exc_peak[i], "exc_rms": exc_rms[i],
            "exc_energy": exc_energy[i], "exc_active": exc_active[i],
            "exc_snr": exc_snr[i], "excitation_score": excitation[i],
            "time_corr": corr_clip[i], "time_env_sim": env_sim_clip[i],
            "time_stability_score": time_stab[i],
            "spec_cos_sim": spec_sim[i], "spectral_stability_score": spec_stab[i],
            "feat_dist": dist[i], "feature_stability_score": feat_stab[i],
            "recency_prior": recency[i], "anomaly_penalty": anomaly[i],
            "total_score": total[i], "rank": ranks[i],
            "selected": 1 if i == selected_idx else 0,
        }
        row.update(proc[i]["feat"])
        diagnostics.append(row)

    return {
        "diagnostics": diagnostics,
        "selected_knock": selected_idx + 1,
        "selected_signal": knocks[selected_idx],
        "selected_score": float(total[selected_idx]),
    }


__all__ = ["select_representative"]
