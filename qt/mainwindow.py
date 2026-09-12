# -*- coding: utf-8 -*-
"""主窗口: 采集 + 敲击检测 + 保存 + 模型预测。"""

import os
import datetime

import numpy as np
import pandas as pd
import joblib
import threading

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QIODevice, QTimer, QThread, Qt, Signal, QSettings
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
    """后台训练线程, 避免阻塞 UI, 支持暂停/继续。"""

    progress = Signal(int, int, int, int, float, float)   # epoch, total, batch_done, batch_total, loss, acc

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.error = None
        self.log_text = ""
        self.pause_event = threading.Event()
        self.pause_event.set()   # 初始为运行状态

    def run(self):
        import io
        import inspect
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                try:
                    sig = inspect.signature(self.module.main).parameters
                except Exception:
                    sig = {}
                kwargs = {}
                if "progress_callback" in sig:
                    kwargs["progress_callback"] = (lambda ep, tot, bd, bt, loss, acc:
                                                   self.progress.emit(ep, tot, bd, bt, loss, acc))
                if "pause_event" in sig:
                    kwargs["pause_event"] = self.pause_event
                self.module.main(**kwargs)
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
        # 预测: 已加载模型
        self.loaded_model = None
        self.loaded_model_type = None    # 'lgbm' / 'dl'
        self.loaded_model_device = None
        # 训练曲线数据
        self._train_losses = []
        self._train_accs = []

        n_freq = N_FFT // 2 + 1
        self.spec_history = np.zeros((n_freq, HISTORY_FRAMES))
        self.mel_history = np.zeros((N_MELS, HISTORY_FRAMES))
        self.cqt_history = np.zeros((N_CQT, HISTORY_FRAMES))
        self.log_spec_history = np.zeros((N_LOG_SPEC, HISTORY_FRAMES))

        build_ui(self)

        if T is not None:
            self.edit_train_dir.setText(getattr(T.config, "DATA_PATH", ""))
            self.edit_label_file.setText(getattr(T.config, "LABEL_PATH", ""))

        # 记忆历史路径(模型/数据集/训练目录)
        self.settings = QSettings("reemoon", "durian_vidio")
        self._restore_settings()
        # 预测数据集目录默认使用训练数据目录
        if T is not None and not self.edit_pred_dir.text().strip():
            self.edit_pred_dir.setText(getattr(T.config, "DATA_PATH", ""))

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

    def _restore_settings(self):
        """恢复历史路径设置。"""
        for attr, key in [
            ("edit_model_path", "predict/model_path"),
            ("edit_pred_dir", "predict/data_dir"),
            ("edit_train_dir", "train/data_dir"),
            ("edit_label_file", "train/label_file"),
            ("edit_train_out", "train/out_dir"),
        ]:
            val = self.settings.value(key, "")
            if val and hasattr(self, attr):
                getattr(self, attr).setText(val)

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
            self.settings.setValue("train/data_dir", d)

    def _browse_label_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择标签文件", self.edit_label_file.text(), "CSV (*.csv)")
        if f:
            self.edit_label_file.setText(f)
            self.settings.setValue("train/label_file", f)

    def _browse_train_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择结果保存目录", self.edit_train_out.text())
        if d:
            self.edit_train_out.setText(d)
            self.settings.setValue("train/out_dir", d)

    def _train_model(self):
        if T is None:
            QtWidgets.QMessageBox.warning(self, "错误", "无法导入训练模块 train.py")
            return
        # 根据「模型」下拉框选择训练模块与框架
        module = T
        try:
            model_type = self.combo_model.currentData()
        except Exception:
            model_type = "lgbm"
        if model_type != "lgbm":
            try:
                from train import dl_train as DL
            except ImportError:
                import train.dl_train as DL
            module = DL
            # 把界面上的深度学习参数写入 dl_train 模块
            try:
                DL.MODEL_TYPE = model_type
                DL.EPOCHS = self.spin_dl_epochs.value()
                DL.BATCH_SIZE = self.spin_dl_batch.value()
                DL.LR = self.spin_dl_lr.value()
            except Exception:
                pass
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
        # 测试集比例
        try:
            T.config.TEST_SIZE = self.spin_test_size.value()
        except Exception:
            pass
        # 结果保存路径(用户可指定, 留空则按时间戳归档)
        out_dir = self.edit_train_out.text().strip()
        T.config.OUTPUT_DIR = out_dir or os.path.join(
            T.PROJECT_ROOT, "results", "train", T.config.RUN_TIME)
        os.makedirs(T.config.OUTPUT_DIR, exist_ok=True)
        # 模型文件名体现训练方法/框架
        if module is T:
            model_name = "model_lgbm.pkl"
        else:
            model_name = f"model_{getattr(module, 'MODEL_TYPE', 'cnn14')}.pt"
        T.config.MODEL_PATH = os.path.join(T.config.OUTPUT_DIR, model_name)
        T.config.VAL_RESULT_PATH = os.path.join(T.config.OUTPUT_DIR, "val_result.csv")
        # wandb 开关(仅深度学习训练生效)
        if module is not T:
            try:
                module.USE_WANDB = self.chk_use_wandb.isChecked()
            except Exception:
                pass
        self.btn_train.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("暂停训练")
        self.lbl_status.setText("模型训练中, 请稍候...")
        self.txt_train_log.setPlainText("训练中, 请稍候...\n")
        self.progress_train.setValue(0)
        self.progress_train_epoch.setValue(0)
        self._train_losses = []
        self._train_accs = []
        self.plot_train_loss.clear()
        self.plot_train_acc.clear()
        self._train_thread = TrainThread(module)
        self._train_thread.progress.connect(self._on_train_progress)
        self._train_thread.finished.connect(self._on_train_done)
        self._train_thread.start()

    def _on_train_done(self):
        self.btn_train.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停训练")
        t = self._train_thread
        if getattr(t, "log_text", ""):
            self.txt_train_log.setPlainText(t.log_text)
        if t.error is not None:
            self.lbl_status.setText("训练失败")
            QtWidgets.QMessageBox.warning(self, "训练失败", str(t.error))
        else:
            self.lbl_status.setText("训练完成")
            self.progress_train.setValue(100)
            QtWidgets.QMessageBox.information(
                self, "训练完成", f"模型已保存:\n{t.module.config.MODEL_PATH}")
            self._plot_train_metrics()

    def _on_train_progress(self, epoch, total, batch_done, batch_total, loss, acc):
        self.progress_train.setValue(int(epoch / max(total, 1) * 100))
        if batch_total:
            self.progress_train_epoch.setValue(int(batch_done / max(batch_total, 1) * 100))
        if acc is not None:
            self.lbl_status.setText(f"训练中... {epoch}/{total} (测试 acc {acc:.4f})")
            self._train_accs.append(acc)
            self._update_train_curves()
        else:
            self.lbl_status.setText(f"训练中... 第 {epoch}/{total} 轮, batch {batch_done}/{batch_total}")
            if loss is not None:
                self._train_losses.append(loss)
                self._update_train_curves()

    def _toggle_pause(self):
        """暂停/继续训练。"""
        t = getattr(self, "_train_thread", None)
        if t is None or not t.isRunning():
            return
        if t.pause_event.is_set():
            t.pause_event.clear()
            self.btn_pause.setText("继续训练")
            self.lbl_status.setText("训练已暂停")
        else:
            t.pause_event.set()
            self.btn_pause.setText("暂停训练")
            self.lbl_status.setText("训练继续中...")

    def _update_train_curves(self):
        """实时更新训练 loss 与 acc 曲线(增量 setData, 避免重绘卡顿)。"""
        if self._train_losses:
            self.curve_train_loss.setData(list(range(len(self._train_losses))), self._train_losses)
        if self._train_accs:
            self.curve_train_acc.setData(list(range(1, len(self._train_accs) + 1)), self._train_accs)

    def _plot_train_metrics(self):
        """训练完成后, 从 val_result.csv 画各样本预测散点(按真实类别着色)。

        兼容两种格式:
        - LightGBM 回归: 有 score 列, 画连续分数 + 阈值线;
        - 深度学习分类: 有 pred_class 列, 画预测类别。
        """
        import pyqtgraph as pg
        val_path = T.config.VAL_RESULT_PATH if T is not None else ""
        if not val_path or not os.path.isfile(val_path):
            return
        try:
            df = pd.read_csv(val_path)
            self.plot_train_metrics.clear()
            colors = {1: "g", 2: "y", 3: "r"}
            if "score" in df.columns:
                # LightGBM 回归: 连续分数散点 + 阈值线
                self.plot_train_metrics.setTitle("测试集各样本预测分数(按真实类别着色)")
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
            elif "pred_class" in df.columns:
                # 深度学习分类: 画预测类别
                self.plot_train_metrics.setTitle("测试集各样本预测类别(按真实类别着色)")
                for cls in [1, 2, 3]:
                    sub = df[df["true_class"] == cls]
                    if len(sub):
                        name = T.CLASS_NAMES.get(cls, str(cls))
                        self.plot_train_metrics.plot(
                            sub.index.values, sub["pred_class"].values, pen=None, symbol="o",
                            symbolBrush=colors[cls], symbolSize=8, name=name)
                self.plot_train_metrics.setYRange(0, 4)
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
    def _load_model_file(self):
        """加载用户选择的模型文件(LightGBM .pkl 或深度学习 .pt)。"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pkl *.pt)")
        if not path:
            return
        self.edit_model_path.setText(path)
        self.settings.setValue("predict/model_path", path)
        try:
            if path.endswith(".pt"):
                from predict import dl_inference as dl_inf
                import torch
                dev = torch.device("cpu") if self.chk_use_cpu.isChecked() else None
                self.loaded_model, self.loaded_model_device = dl_inf.load_dl_model(path, device=dev)
                self.loaded_model_type = "dl"
                self.lbl_model_status.setText(f"已加载深度学习模型({self.loaded_model_device}): {os.path.basename(path)}")
            else:
                self.loaded_model = joblib.load(path)
                self.loaded_model_type = "lgbm"
                self.lbl_model_status.setText(f"已加载 LightGBM 模型: {os.path.basename(path)}")
        except Exception as e:
            self.loaded_model = None
            self.loaded_model_type = None
            self.lbl_model_status.setText("模型加载失败")
            QtWidgets.QMessageBox.warning(self, "错误", f"模型加载失败:\n{e}")

    def _browse_pred_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择预测数据集目录", self.edit_pred_dir.text())
        if d:
            self.edit_pred_dir.setText(d)
            self.settings.setValue("predict/data_dir", d)

    def _browse_single_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择单个敲击数据 CSV", "", "CSV (*.csv)")
        if f:
            self.edit_single_file.setText(f)

    def _predict_single_file(self):
        """预测单个 CSV 文件: 样本级 + 信号级。"""
        path = self.edit_single_file.text().strip()
        if not path:
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择敲击数据 CSV", "", "CSV (*.csv)")
            if not f:
                return
            path = f
            self.edit_single_file.setText(path)
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "提示", "文件不存在")
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
            for row in arr[:20]:
                r = row[:1024] if len(row) >= 1024 else np.pad(row, (0, 1024 - len(row)))
                knocks.append(np.asarray(r, dtype=np.float64))
            pred = self._predict_knocks_value(knocks)
            if pred is None:
                self.lbl_pred.setText("预测: 无模型")
                return
            name = T.CLASS_NAMES.get(pred, str(pred))
            self.lbl_pred.setText(f"预测: {name}({pred})")
            # 信号级预测
            rows = []
            for i, x in enumerate(knocks):
                sp = self._predict_knocks_value([x])
                rows.append({"文件": os.path.basename(path),
                             "信号序号": i + 1,
                             "预测": sp if sp is not None else "-",
                             "预测名称": T.CLASS_NAMES.get(sp, "-") if sp is not None else "-"})
            txt = (f"单个文件预测: {os.path.basename(path)}\n"
                   f"样本级预测: {name}({pred})\n\n")
            self.txt_pred_result.setPlainText(
                txt + pd.DataFrame(rows).to_string(index=False))
        except Exception as e:
            self.lbl_pred.setText(f"预测失败: {e}")

    def _predict_knocks_value(self, knocks):
        """对多条敲击信号预测, 返回类别 int(1/2/3), 失败返回 None。"""
        if not knocks:
            return None
        if self.loaded_model is not None:
            try:
                if self.loaded_model_type == "dl":
                    from predict import dl_inference as dl_inf
                    return dl_inf.predict_dl(self.loaded_model, knocks,
                                             self.loaded_model_device)
                return predictor.predict_class(self.loaded_model, knocks, T)
            except Exception:
                return None
        # 回退: 自动加载最新 LightGBM 模型
        model = predictor.load_model(T)
        if model is None:
            return None
        try:
            return predictor.predict_class(model, knocks, T)
        except Exception:
            return None

    def _predict_knocks(self, knocks):
        pred = self._predict_knocks_value(knocks)
        if pred is None:
            if not knocks:
                self.lbl_pred.setText("预测: 无数据")
            else:
                self.lbl_pred.setText("预测: 无模型")
            return
        name = T.CLASS_NAMES.get(pred, str(pred))
        self.lbl_pred.setText(f"预测: {name}({pred})")

    def _predict_and_show(self):
        self._predict_knocks(self.knock_list)

    def _predict_manual(self):
        """手动预测: 数据集目录 > 采集数据 > 单个 CSV。"""
        # 1. 数据集目录批量预测
        pred_dir = self.edit_pred_dir.text().strip()
        if pred_dir and os.path.isdir(pred_dir):
            self._predict_directory(pred_dir)
            return
        # 2. 采集数据
        if self.knock_list:
            self._predict_knocks(self.knock_list)
            return
        # 3. 单个 CSV
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

    def _predict_directory(self, dir_path):
        """批量预测目录内所有 CSV, 按样本聚合, 对比真实标签并输出准确率。"""
        # 收集 CSV 文件(兼容 样本ID_位置.csv 与 位置.csv 两种命名)
        files = []
        for f in sorted(os.listdir(dir_path)):
            if not f.endswith(".csv") or "label" in f:
                continue
            stem = f[:-4]
            if "_" in stem and stem.split("_")[0].isdigit():
                files.append(f)
            elif stem.isdigit():
                files.append(f)
        if not files:
            all_csv = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
            if all_csv:
                msg = (f"目录内有 {len(all_csv)} 个 CSV, 但都不符合命名规则"
                       f"(需为 样本ID_位置.csv 或 位置.csv)\n例如: {all_csv[:3]}")
            else:
                msg = f"目录内没有 CSV 文件:\n{dir_path}"
            QtWidgets.QMessageBox.warning(self, "提示", msg)
            return

        def file_sid(f):
            """从文件名提取样本ID: 021_01.csv -> 21, 02.csv -> 0(单一样本)。"""
            stem = f[:-4]
            if "_" in stem and stem.split("_")[0].isdigit():
                return int(stem.split("_")[0])
            return 0

        # 读取真实标签(若存在 label.csv)
        label_map = {}
        label_path = os.path.join(dir_path, "label.csv")
        if os.path.isfile(label_path):
            try:
                lab = pd.read_csv(label_path, header=None)
                labels = pd.to_numeric(lab.iloc[:, 0], errors="coerce").dropna().astype(int).values
                sids = sorted({file_sid(f) for f in files})
                label_map = {sid: int(lb) for sid, lb in zip(sids, labels[:len(sids)])}
            except Exception:
                label_map = {}

        # 按样本聚合信号(同时记录每个文件的信号)
        from collections import defaultdict
        sample_knocks = defaultdict(list)
        file_knocks = {}
        for f in files:
            sid = file_sid(f)
            try:
                df = pd.read_csv(os.path.join(dir_path, f), header=None)
                arr = np.nan_to_num(df.values.astype(np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arr = arr[np.any(arr != 0, axis=1)]
                knocks = []
                for row in arr[:5]:   # 每文件最多 5 条信号
                    r = row[:1024] if len(row) >= 1024 else np.pad(row, (0, 1024 - len(row)))
                    knocks.append(np.asarray(r, dtype=np.float64))
                sample_knocks[sid].extend(knocks)
                file_knocks[f] = knocks
            except Exception:
                pass

        if not sample_knocks:
            QtWidgets.QMessageBox.warning(self, "提示", "目录内没有有效信号")
            return

        # 统计(样本级 + 信号级 + 每类)
        class_names = T.CLASS_NAMES
        sample_total = sample_correct = 0
        sig_total = sig_correct = 0
        sample_cls_total = {c: 0 for c in class_names}
        sample_cls_correct = {c: 0 for c in class_names}
        sig_cls_total = {c: 0 for c in class_names}
        sig_cls_correct = {c: 0 for c in class_names}

        rows = []
        signal_rows = []
        sids_sorted = sorted(sample_knocks.keys())
        n_samples = len(sids_sorted)
        self.progress_predict.setValue(0)
        for idx, sid in enumerate(sids_sorted):
            knocks = sample_knocks[sid]
            true = label_map.get(sid)
            pred = self._predict_knocks_value(knocks)
            mark = ""
            if pred is not None and true is not None:
                sample_total += 1
                sample_cls_total[true] = sample_cls_total.get(true, 0) + 1
                if pred == true:
                    sample_correct += 1
                    sample_cls_correct[true] = sample_cls_correct.get(true, 0) + 1
                    mark = "√"
                else:
                    mark = "×"
            sample_label = sid if sid != 0 else "/".join(
                os.path.splitext(f)[0] for f in files)
            # 信号级: 遍历该样本每个文件的每条信号, 记录预测结果
            for f, fknocks in file_knocks.items():
                if file_sid(f) != sid:
                    continue
                for i, x in enumerate(fknocks):
                    sp = self._predict_knocks_value([x])
                    if sp is not None:
                        signal_rows.append({
                            "样本": sample_label,
                            "文件": f,
                            "信号序号": i + 1,
                            "真实": true if true is not None else "-",
                            "预测": sp,
                            "预测名称": T.CLASS_NAMES.get(sp, str(sp)),
                        })
                        if true is not None:
                            sig_total += 1
                            sig_cls_total[true] = sig_cls_total.get(true, 0) + 1
                            if sp == true:
                                sig_correct += 1
                                sig_cls_correct[true] = sig_cls_correct.get(true, 0) + 1
            rows.append({"样本": sample_label,
                         "真实": true if true is not None else "-",
                         "预测": pred if pred is not None else "无法预测",
                         "预测名称": T.CLASS_NAMES.get(pred, "-") if pred else "-",
                         "正确": mark})
            self.progress_predict.setValue(int((idx + 1) / n_samples * 100))
            QtCore.QCoreApplication.processEvents()

        # 文件级预测(每个文件的信号单独预测)
        file_preds = []
        for f in files:
            knocks = file_knocks.get(f)
            if knocks:
                p = self._predict_knocks_value(knocks)
                if p is not None:
                    file_preds.append((f, p, T.CLASS_NAMES.get(p, str(p))))

        # 组装报告
        lines = ["========== 预测报告 =========="]
        lines.append(f"预测目录: {dir_path}")
        model_desc = getattr(self.loaded_model, "model_type", None)
        if not model_desc:
            model_desc = self.loaded_model_type or "自动"
        lines.append(f"模型: {model_desc} | 时间: "
                     f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if file_preds:
            lines.append("")
            lines.append("各文件预测:")
            for f, p, pname in file_preds:
                lines.append(f"  {f} -> {pname}({p})")
        lines.append("")
        if sample_total:
            lines.append(f"样本级准确率: {sample_correct}/{sample_total} = {sample_correct/sample_total:.2%}")
            for c in sorted(class_names):
                n = sample_cls_total[c]
                if n:
                    lines.append(f"  类别{c}({class_names[c]}): {sample_cls_correct[c]}/{n} = {sample_cls_correct[c]/n:.2%}")
        if sig_total:
            lines.append(f"信号级准确率: {sig_correct}/{sig_total} = {sig_correct/sig_total:.2%}")
            for c in sorted(class_names):
                n = sig_cls_total[c]
                if n:
                    lines.append(f"  类别{c}({class_names[c]}): {sig_cls_correct[c]}/{n} = {sig_cls_correct[c]/n:.2%}")
        if not label_map:
            lines.append("目录内无 label.csv, 无法计算准确率")
        lines.append("=" * 28)

        if rows:
            df_res = pd.DataFrame(rows)
            report_text = "\n".join(lines) + "\n\n" + df_res.to_string(index=False)
            self.txt_pred_result.setPlainText(report_text)
            # 保存所有结果到文档
            try:
                out_dir = os.path.join(T.PROJECT_ROOT, "results", "predict",
                                       datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)
                # 信号级结果(主 CSV, 每条信号一行)
                if signal_rows:
                    pd.DataFrame(signal_rows).to_csv(
                        os.path.join(out_dir, "predict_result.csv"),
                        index=False, encoding="utf-8-sig")
                # 样本级结果
                df_res.to_csv(os.path.join(out_dir, "sample_result.csv"),
                              index=False, encoding="utf-8-sig")
                with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fp:
                    fp.write(report_text)
                self.txt_pred_result.append(
                    f"\n\n预测结果已保存到:\n{out_dir}\n"
                    f"  - predict_result.csv ({len(signal_rows)} 条信号级结果)\n"
                    f"  - sample_result.csv (样本级结果)\n"
                    f"  - report.txt")
            except Exception as e:
                self.txt_pred_result.append(f"\n\n保存结果失败: {e}")
        else:
            self.txt_pred_result.setPlainText("\n".join(lines))

        # 单样本预测: 醒目标注预测结果
        if len(sample_knocks) == 1 and rows:
            r = rows[0]
            pred_val = r.get("预测")
            if isinstance(pred_val, int):
                self.lbl_pred.setText(f"预测: {T.CLASS_NAMES.get(pred_val, str(pred_val))}({pred_val})")
            return

        if sample_total:
            head = f"样本级 {sample_correct/sample_total:.2%}"
            if sig_total:
                head += f" | 信号级 {sig_correct/sig_total:.2%}"
            self.lbl_pred.setText(f"批量预测完成: {len(sample_knocks)} 个样本 | {head}")
        else:
            self.lbl_pred.setText(f"批量预测完成: {len(sample_knocks)} 个样本 | 无标签无法计算准确率")
