# -*- coding: utf-8 -*-
"""主窗口: 采集 + 敲击检测 + 保存 + 模型预测。"""

import os
import datetime

import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import QIODevice, QTimer, QThread, Qt
from PySide6.QtMultimedia import QAudioInput, QAudioFormat, QMediaDevices

from qt.audio import AudioBuffer
from qt.ui import build_ui, DEFAULT_DATA_DIR
import predict as predictor

from tools.dsp import SR, NOISE_K, N_FFT, denoise, normalize, envelope
from tools.features import (N_MELS, N_CQT, N_LOG_SPEC, compute_spectrogram,
                            compute_mel, compute_cqt, compute_log_spec,
                            compute_mfcc, compute_features,
                            compute_energy_envelope, compute_centroid_trace,
                            compute_spectral_flux)
from tools.deepspectrum import (compute_deep_spectrum, compute_feature_maps,
                                feature_maps_grid)

# 引入 whg/train.py 用于模型预测
try:
    import train as T
except Exception:
    T = None

# ================== 采集常量 ==================
KNOCK_LEN = 1024
PRE_PEAK = 128
POST_PEAK = KNOCK_LEN - PRE_PEAK
BUFFER_SIZE = SR * 2

KNOCK_NOISE_K = 5.0
MIN_KNOCK_INTERVAL = 0.3
HISTORY_FRAMES = 200


class TrainThread(QThread):
    """后台训练线程, 避免阻塞 UI。"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.error = None
        self.log_text = ""

    def run(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                self.module.main()
            self.log_text = buf.getvalue()
        except Exception as e:
            self.error = e
            self.log_text = buf.getvalue()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("声音采集 - 敲击触发 + 谱图/特征实时显示")
        self.resize(1000, 760)

        self.audio_input = None
        self.audio_buffer = None
        self.raw_samples = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.total_samples = 0

        self.knock_list = []
        self.last_knock_sample = -10 ** 9
        self.preview_list = []   # [(文件名, 信号数组)]
        self.preview_idx = -1
        self._ds_error = None    # Deep Spectrum 首次失败后不再重试

        n_freq = N_FFT // 2 + 1
        self.spec_history = np.zeros((n_freq, HISTORY_FRAMES))
        self.mel_history = np.zeros((N_MELS, HISTORY_FRAMES))
        self.cqt_history = np.zeros((N_CQT, HISTORY_FRAMES))
        self.log_spec_history = np.zeros((N_LOG_SPEC, HISTORY_FRAMES))

        build_ui(self)

        if T is not None:
            self.edit_train_dir.setText(getattr(T.config, "DATA_PATH", ""))
            self.edit_label_file.setText(getattr(T.config, "LABEL_PATH", ""))

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update)

    # ---------- 采集控制 ----------
    def start(self):
        fmt = QAudioFormat()
        fmt.setSampleRate(SR)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)

        dev = self.combo_device.currentData()
        if dev is None or dev.isNull():
            dev = QMediaDevices.defaultAudioInput()
        if dev is None or dev.isNull():
            QtWidgets.QMessageBox.warning(self, "错误", "未找到可用麦克风设备")
            return

        self.audio_buffer = AudioBuffer()
        self.audio_buffer.open(QIODevice.WriteOnly)
        self.audio_input = QAudioInput(dev, fmt, self)
        self.audio_input.start(self.audio_buffer)

        self.knock_list = []
        self.last_knock_sample = -10 ** 9

        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText(f"采集中... 0/{self.spin_knocks.value()}")

    def stop(self):
        self.timer.stop()
        if self.audio_input is not None:
            self.audio_input.stop()
            self.audio_input = None
        if self.audio_buffer is not None:
            self.audio_buffer.close()
            self.audio_buffer = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("已停止")

    def _browse_data_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择数据保存目录", self.edit_data_dir.text())
        if d:
            self.edit_data_dir.setText(d)

    def _browse_train_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择训练数据目录", self.edit_train_dir.text())
        if d:
            self.edit_train_dir.setText(d)

    def _browse_label_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择标签文件", self.edit_label_file.text(), "CSV (*.csv)")
        if f:
            self.edit_label_file.setText(f)

    def _train_model(self):
        if T is None:
            QtWidgets.QMessageBox.warning(self, "错误", "无法导入训练模块 train.py")
            return
        data_dir = self.edit_train_dir.text().strip()
        label_file = self.edit_label_file.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            QtWidgets.QMessageBox.warning(self, "错误", "训练数据目录无效")
            return
        if not label_file or not os.path.isfile(label_file):
            QtWidgets.QMessageBox.warning(self, "错误", "标签文件无效")
            return
        T.config.DATA_PATH = data_dir
        T.config.LABEL_PATH = label_file
        T.config.RUN_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        T.config.OUTPUT_DIR = os.path.join(
            T.PROJECT_ROOT, "results", "train", T.config.RUN_TIME)
        T.config.MODEL_PATH = os.path.join(T.config.OUTPUT_DIR, "model.pkl")
        T.config.VAL_RESULT_PATH = os.path.join(T.config.OUTPUT_DIR, "val_result.csv")
        self.btn_train.setEnabled(False)
        self.lbl_status.setText("模型训练中, 请稍候...")
        self.txt_train_log.setPlainText("训练中, 请稍候...\n")
        self._train_thread = TrainThread(T)
        self._train_thread.finished.connect(self._on_train_done)
        self._train_thread.start()

    def _on_train_done(self):
        self.btn_train.setEnabled(True)
        t = self._train_thread
        if getattr(t, "log_text", ""):
            self.txt_train_log.setPlainText(t.log_text)
        if t.error is not None:
            self.lbl_status.setText("训练失败")
            QtWidgets.QMessageBox.warning(self, "训练失败", str(t.error))
        else:
            self.lbl_status.setText("训练完成")
            QtWidgets.QMessageBox.information(
                self, "训练完成", f"模型已保存:\n{t.module.config.MODEL_PATH}")
            self._plot_train_metrics()

    def _plot_train_metrics(self):
        """训练完成后, 从 val_result.csv 画各样本预测分数散点(按真实类别着色)。"""
        import pyqtgraph as pg
        val_path = T.config.VAL_RESULT_PATH if T is not None else ""
        if not val_path or not os.path.isfile(val_path):
            return
        try:
            df = pd.read_csv(val_path)
            self.plot_train_metrics.clear()
            colors = {1: "g", 2: "y", 3: "r"}
            for cls in [1, 2, 3]:
                sub = df[df["true_class"] == cls]
                if len(sub):
                    name = T.CLASS_NAMES.get(cls, str(cls))
                    self.plot_train_metrics.plot(
                        sub.index.values, sub["score"].values, pen=None, symbol="o",
                        symbolBrush=colors[cls], symbolSize=8, name=name)
            for th in [T.config.THRESH_1_2, T.config.THRESH_2_3]:
                self.plot_train_metrics.addLine(
                    y=th, pen=pg.mkPen("gray", style=Qt.DashLine))
        except Exception as e:
            print(f"画训练指标失败: {e}")

    def _current_denoise_flags(self):
        return (self.chk_notch.isChecked(), self.chk_bandpass.isChecked(),
                self.chk_specsub.isChecked(), self.chk_gate.isChecked(),
                self.chk_median.isChecked(), self.chk_wiener.isChecked(),
                self.chk_smooth.isChecked())

    def _process(self, seg):
        """按当前降噪选项处理信号, 返回 (降噪后, 归一化后)。"""
        use_notch, use_bandpass, use_specsub, use_gate, use_median, use_wiener, use_smooth = \
            self._current_denoise_flags()
        x = denoise(seg, SR, use_notch, use_bandpass, use_specsub, use_gate,
                    self.spin_gate_k.value(), use_median, use_wiener, use_smooth)
        return x, normalize(x)

    # ---------- 数据更新 ----------
    def _append_samples(self, samples):
        n = len(samples)
        if n == 0:
            return
        if n >= BUFFER_SIZE:
            self.raw_samples = samples[-BUFFER_SIZE:].copy()
        else:
            self.raw_samples = np.concatenate([self.raw_samples[n:], samples])
        self.total_samples += n

    def _update(self):
        if self.audio_buffer is None:
            return
        data = self.audio_buffer.pop_all()
        if not data:
            return
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        arr = arr * self.spin_gain.value()
        self._append_samples(arr)

        # 电平监控
        recent = self.raw_samples[-KNOCK_LEN:]
        peak_level = float(np.max(np.abs(recent))) if len(recent) else 0.0
        self.bar_level.setValue(int(min(peak_level * 100, 100)))
        self.lbl_clip.setText("⚠ 削波!" if peak_level > self.spin_clip.value() else "")

        # 波形对比
        frame = self.raw_samples[-KNOCK_LEN:]
        x_raw = normalize(frame)
        _, x = self._process(frame)
        self.curve_raw.setData(x_raw)
        self.curve_clean.setData(x)

        # 时域图
        time_frame = self.raw_samples[-2048:]
        env = envelope(time_frame) if len(time_frame) else np.zeros(0)
        self.curve_time.setData(time_frame)
        self.curve_env.setData(env)

        # 频谱图
        spec = compute_spectrogram(x, SR)
        nf = spec.shape[1]
        self.spec_history = np.hstack([self.spec_history[:, nf:], spec])
        self.img_spec.setImage(self.spec_history.T, autoLevels=False, levels=(-80, 0))

        # 梅尔谱图
        mel = compute_mel(x, SR)
        nf2 = mel.shape[1]
        self.mel_history = np.hstack([self.mel_history[:, nf2:], mel])
        self.img_mel.setImage(self.mel_history.T, autoLevels=False, levels=(-80, 0))

        cqt = compute_cqt(x, SR)
        nf3 = cqt.shape[1]
        self.cqt_history = np.hstack([self.cqt_history[:, nf3:], cqt])
        self.img_cqt.setImage(self.cqt_history.T, autoLevels=False, levels=(-80, 0))

        log_spec = compute_log_spec(x, SR)
        nf4 = log_spec.shape[1]
        self.log_spec_history = np.hstack([self.log_spec_history[:, nf4:], log_spec])
        self.img_log_spec.setImage(self.log_spec_history.T, autoLevels=False, levels=(-80, 0))

        # MFCC + 特征
        mfcc = compute_mfcc(x, SR)
        self.curve_mfcc.setData(mfcc)
        feats = compute_features(x, SR)
        for name, lbl in self.feat_labels.items():
            v = feats.get(name, 0.0)
            lbl.setText(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")

        # 特征曲线
        values = np.array([feats.get(n, 0.0) for n in self.feat_names])
        vmin, vmax = values.min(), values.max()
        v_norm = (values - vmin) / (vmax - vmin + 1e-12)
        self.curve_feat.setData(v_norm)
        self.plot_feat.getAxis("bottom").setTicks(
            [[(i, n) for i, n in enumerate(self.feat_names)]])
        self.plot_feat.setXRange(0, len(self.feat_names) - 1)

        # 能量包络 / 谱质心随时间 / 谱通量
        self.curve_energy.setData(compute_energy_envelope(x, SR))
        self.curve_centroid.setData(compute_centroid_trace(x, SR))
        self.curve_flux.setData(compute_spectral_flux(x, SR))

        # 敲击检测
        knock = self._detect_knock()
        if knock is not None:
            self.knock_list.append(knock)
            n = len(self.knock_list)
            target = self.spin_knocks.value()
            self.lbl_status.setText(f"采集中... {n}/{target}")
            self.curve_clean.setData(knock)
            if n >= target:
                self._save_csv()
                self.stop()

    def _detect_knock(self):
        buf = self.raw_samples
        win = buf[-2048:]
        if len(win) < KNOCK_LEN:
            return None
        peak_abs = float(np.max(np.abs(win)))
        noise = float(np.percentile(np.abs(win), 20))
        thresh = max(noise * KNOCK_NOISE_K, self.spin_thresh.value())
        if peak_abs < thresh or peak_abs < self.spin_min_amp.value():
            return None
        if self.total_samples - self.last_knock_sample < MIN_KNOCK_INTERVAL * SR:
            return None

        p = int(np.argmax(np.abs(win)))
        global_peak = len(buf) - len(win) + p
        start = global_peak - PRE_PEAK
        end = start + KNOCK_LEN

        seg = np.zeros(KNOCK_LEN, dtype=np.float64)
        s0, e0 = max(start, 0), min(end, len(buf))
        if e0 > s0:
            seg[s0 - start: e0 - start] = buf[s0:e0]

        self.last_knock_sample = self.total_samples
        _, x = self._process(seg)
        return x

    # ---------- 保存 ----------
    def _save_csv(self):
        if not self.knock_list:
            return
        sid = self.spin_sid.value()
        pos = self.combo_pos.currentText()
        data_dir = self.edit_data_dir.text().strip() or DEFAULT_DATA_DIR
        os.makedirs(data_dir, exist_ok=True)

        mat = np.vstack(self.knock_list)
        csv_path = os.path.join(data_dir, f"{sid}_{pos}.csv")
        pd.DataFrame(mat).to_csv(csv_path, header=False, index=False)
        print(f"已保存: {csv_path}  shape={mat.shape}")

        # 谱图不再逐个落盘 PNG, 直接从刚保存的 CSV 加载并预览
        self._load_csv_to_preview([csv_path])

        self._predict_and_show()

    # ---------- 历史预览 ----------
    def _load_csv_to_preview(self, paths):
        """从 CSV 文件加载信号到预览列表(谱图按需计算, 不落盘 PNG)。"""
        self.preview_list = []
        for p in paths:
            try:
                df = pd.read_csv(p, header=None)
                arr = np.nan_to_num(df.values.astype(np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arr = arr[np.any(arr != 0, axis=1)]
                for row in arr:
                    r = row[:1024] if len(row) >= 1024 else np.pad(row, (0, 1024 - len(row)))
                    self.preview_list.append((os.path.basename(p), np.asarray(r, dtype=np.float64)))
            except Exception as e:
                print(f"加载 {p} 失败: {e}")
        if not self.preview_list:
            self.lbl_preview.setText("无有效信号")
            return
        self.preview_idx = 0
        self._show_preview()

    def _browse_preview(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择敲击数据 CSV(可多选)", self.edit_data_dir.text(), "CSV (*.csv)")
        if not paths:
            return
        self._load_csv_to_preview(paths)

    def _show_preview(self):
        if not self.preview_list or self.preview_idx < 0:
            return
        fname, sig = self.preview_list[self.preview_idx]
        self.lbl_preview.setText(f"{self.preview_idx + 1}/{len(self.preview_list)}: {fname}")
        x = normalize(sig)

        # 波形对比
        self.curve_raw.setData(sig)
        self.curve_clean.setData(x)

        # 时域图
        self.curve_time.setData(sig)
        self.curve_env.setData(envelope(sig))

        # 谱图(单条信号, 不滚动)
        self.img_spec.setImage(compute_spectrogram(x, SR).T, autoLevels=False, levels=(-80, 0))
        self.img_mel.setImage(compute_mel(x, SR).T, autoLevels=False, levels=(-80, 0))
        self.img_cqt.setImage(compute_cqt(x, SR).T, autoLevels=False, levels=(-80, 0))
        self.img_log_spec.setImage(compute_log_spec(x, SR).T, autoLevels=False, levels=(-80, 0))

        # MFCC + 特征
        self.curve_mfcc.setData(compute_mfcc(x, SR))
        feats = compute_features(x, SR)
        for name, lbl in self.feat_labels.items():
            v = feats.get(name, 0.0)
            lbl.setText(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
        values = np.array([feats.get(n, 0.0) for n in self.feat_names])
        vmin, vmax = values.min(), values.max()
        self.curve_feat.setData((values - vmin) / (vmax - vmin + 1e-12))

        # 能量包络 / 谱质心随时间 / 谱通量
        self.curve_energy.setData(compute_energy_envelope(x, SR))
        self.curve_centroid.setData(compute_centroid_trace(x, SR))
        self.curve_flux.setData(compute_spectral_flux(x, SR))

        # Deep Spectrum 特征
        self._update_deep_spectrum(x)

    def _prev_preview(self):
        if self.preview_list and self.preview_idx > 0:
            self.preview_idx -= 1
            self._show_preview()

    def _next_preview(self):
        if self.preview_list and self.preview_idx < len(self.preview_list) - 1:
            self.preview_idx += 1
            self._show_preview()

    # ---------- Deep Spectrum ----------
    def _update_deep_spectrum(self, x):
        """计算并显示 Deep Spectrum 特征向量 + CNN 中间层 feature maps。"""
        if self._ds_error is not None:
            return
        try:
            ds = compute_deep_spectrum(x, SR)
            self.curve_ds.setData(ds)
            fmaps = compute_feature_maps(x, SR, layer_name="layer3", n_channels=16)
            self.img_fmap.setImage(feature_maps_grid(fmaps).T, autoLevels=False, levels=(0, 1))
        except Exception as e:
            self._ds_error = str(e)
            self.curve_ds.setData([])
            print(f"Deep Spectrum 不可用: {e}")

    def _compare_deep_spectrum(self):
        """对预览列表中的信号计算 Deep Spectrum 特征并叠加对比(最多 20 条)。"""
        if self._ds_error is not None:
            QtWidgets.QMessageBox.warning(self, "提示", "Deep Spectrum 不可用")
            return
        if not self.preview_list:
            QtWidgets.QMessageBox.warning(self, "提示", "请先「批量预览」加载信号")
            return
        import pyqtgraph as pg

        self.plot_ds_compare.clear()
        n = len(self.preview_list)
        show_n = min(n, 20)
        for i in range(show_n):
            fname, sig = self.preview_list[i]
            try:
                ds = compute_deep_spectrum(normalize(sig), SR)
                color = pg.intColor(i, hues=show_n)
                self.plot_ds_compare.plot(ds, pen=pg.mkPen(color, width=1), name=f"{i + 1}:{fname}")
            except Exception as e:
                print(f"对比 {fname} 失败: {e}")
        self.lbl_preview.setText(f"DS 对比 {show_n}/{n} 条信号")

    # ---------- 预测 ----------
    def _predict_knocks(self, knocks):
        if not knocks:
            self.lbl_pred.setText("预测: 无数据")
            return
        model = predictor.load_model(T)
        if model is None:
            self.lbl_pred.setText("预测: 无模型")
            return
        try:
            pred_class = predictor.predict_class(model, knocks, T)
            name = T.CLASS_NAMES.get(pred_class, str(pred_class))
            self.lbl_pred.setText(f"预测: {name}({pred_class})")
        except Exception as e:
            self.lbl_pred.setText(f"预测失败: {e}")

    def _predict_and_show(self):
        self._predict_knocks(self.knock_list)

    def _predict_manual(self):
        """手动预测: 有采集数据直接用, 否则选 CSV 文件。"""
        if self.knock_list:
            self._predict_knocks(self.knock_list)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择敲击数据 CSV", self.edit_data_dir.text(), "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, header=None)
            arr = np.nan_to_num(df.values.astype(np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr[np.any(arr != 0, axis=1)]
            if arr.shape[0] == 0:
                QtWidgets.QMessageBox.warning(self, "提示", "文件内没有有效信号")
                return
            knocks = []
            for row in arr:
                r = row[:1024] if len(row) >= 1024 else np.pad(row, (0, 1024 - len(row)))
                knocks.append(np.asarray(r, dtype=np.float64))
            self._predict_knocks(knocks)
        except Exception as e:
            self.lbl_pred.setText(f"预测失败: {e}")
