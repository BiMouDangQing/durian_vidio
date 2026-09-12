# -*- coding: utf-8 -*-
"""界面构建: 把所有控件挂到主窗口实例上。"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

import os

from PySide6 import QtCore, QtWidgets
from PySide6.QtMultimedia import QMediaDevices
import pyqtgraph as pg

from tools.dsp import NOISE_K
from tools.features import N_MFCC

# 界面默认值
KNOCK_THRESH = 0.15
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "captured"))


def build_ui(win: QtWidgets.QWidget):
    """构建主窗口界面, 所有控件挂到 win 上。

    win 可以是 QMainWindow(独立运行) 或 QWidget(嵌入前端页签)。
    """
    win.main_tabs = QtWidgets.QTabWidget()

    if isinstance(win, QtWidgets.QMainWindow):
        central = QtWidgets.QWidget()
        win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
    else:
        layout = QtWidgets.QVBoxLayout(win)
    layout.addWidget(win.main_tabs)

    # ================= 采集页 =================
    capture_page = QtWidgets.QWidget()
    cap_layout = QtWidgets.QVBoxLayout(capture_page)

    # 1. 控制栏
    ctrl = QtWidgets.QHBoxLayout()
    ctrl.addWidget(QtWidgets.QLabel("样本ID:"))
    win.spin_sid = QtWidgets.QSpinBox()
    win.spin_sid.setRange(1, 99999)
    win.spin_sid.setValue(1)
    ctrl.addWidget(win.spin_sid)
    ctrl.addWidget(QtWidgets.QLabel("部位:"))
    win.combo_pos = QtWidgets.QComboBox()
    win.combo_pos.addItems(["01", "02", "03", "04"])
    ctrl.addWidget(win.combo_pos)
    ctrl.addWidget(QtWidgets.QLabel("敲击数:"))
    win.spin_knocks = QtWidgets.QSpinBox()
    win.spin_knocks.setRange(1, 50)
    win.spin_knocks.setValue(5)
    ctrl.addWidget(win.spin_knocks)
    win.btn_start = QtWidgets.QPushButton("开始采集")
    win.btn_stop = QtWidgets.QPushButton("停止")
    win.btn_stop.setEnabled(False)
    ctrl.addWidget(win.btn_start)
    ctrl.addWidget(win.btn_stop)
    ctrl.addStretch(1)
    cap_layout.addLayout(ctrl)

    # 2. 输入设备选择
    dev_row = QtWidgets.QHBoxLayout()
    dev_row.addWidget(QtWidgets.QLabel("输入设备:"))
    win.combo_device = QtWidgets.QComboBox()
    for d in QMediaDevices.audioInputs():
        win.combo_device.addItem(d.description(), d)
    dev_row.addWidget(win.combo_device, stretch=1)
    cap_layout.addLayout(dev_row)

    # 3. 输出路径
    path_row = QtWidgets.QHBoxLayout()
    path_row.addWidget(QtWidgets.QLabel("数据目录:"))
    win.edit_data_dir = QtWidgets.QLineEdit(DEFAULT_DATA_DIR)
    path_row.addWidget(win.edit_data_dir, stretch=3)
    win.btn_data_dir = QtWidgets.QPushButton("浏览")
    path_row.addWidget(win.btn_data_dir)
    cap_layout.addLayout(path_row)

    # 历史预览
    pv_row = QtWidgets.QHBoxLayout()
    pv_row.addWidget(QtWidgets.QLabel("历史预览:"))
    win.btn_preview = QtWidgets.QPushButton("批量预览")
    pv_row.addWidget(win.btn_preview)
    win.btn_prev = QtWidgets.QPushButton("上一个")
    win.btn_next = QtWidgets.QPushButton("下一个")
    pv_row.addWidget(win.btn_prev)
    pv_row.addWidget(win.btn_next)
    win.btn_ds_compare = QtWidgets.QPushButton("DS 对比")
    pv_row.addWidget(win.btn_ds_compare)
    win.lbl_preview = QtWidgets.QLabel("未加载")
    pv_row.addWidget(win.lbl_preview, stretch=1)
    cap_layout.addLayout(pv_row)

    # 4. 降噪方法选择
    dn_row = QtWidgets.QHBoxLayout()
    dn_row.addWidget(QtWidgets.QLabel("降噪方法:"))
    win.chk_notch = QtWidgets.QCheckBox("陷波(工频)")
    win.chk_notch.setChecked(True)
    win.chk_bandpass = QtWidgets.QCheckBox("带通(50-700Hz)")
    win.chk_bandpass.setChecked(True)
    win.chk_specsub = QtWidgets.QCheckBox("谱减法")
    win.chk_specsub.setChecked(True)
    win.chk_gate = QtWidgets.QCheckBox("噪声门限")
    win.chk_gate.setChecked(True)
    win.chk_median = QtWidgets.QCheckBox("中值滤波")
    win.chk_median.setChecked(False)
    win.chk_wiener = QtWidgets.QCheckBox("维纳滤波")
    win.chk_wiener.setChecked(False)
    win.chk_smooth = QtWidgets.QCheckBox("平滑")
    win.chk_smooth.setChecked(False)
    dn_row.addWidget(win.chk_notch)
    dn_row.addWidget(win.chk_bandpass)
    dn_row.addWidget(win.chk_specsub)
    dn_row.addWidget(win.chk_gate)
    dn_row.addWidget(win.chk_median)
    dn_row.addWidget(win.chk_wiener)
    dn_row.addWidget(win.chk_smooth)
    dn_row.addStretch(1)
    cap_layout.addLayout(dn_row)

    # 5. 检测参数(可调)
    det_row = QtWidgets.QHBoxLayout()
    det_row.addWidget(QtWidgets.QLabel("输入增益:"))
    win.spin_gain = QtWidgets.QDoubleSpinBox()
    win.spin_gain.setRange(1.0, 100.0)
    win.spin_gain.setSingleStep(1.0)
    win.spin_gain.setDecimals(1)
    win.spin_gain.setValue(1.0)
    det_row.addWidget(win.spin_gain)
    det_row.addWidget(QtWidgets.QLabel("倍"))
    det_row.addWidget(QtWidgets.QLabel("敲击触发阈值:"))
    win.spin_thresh = QtWidgets.QDoubleSpinBox()
    win.spin_thresh.setRange(0.01, 1.0)
    win.spin_thresh.setSingleStep(0.01)
    win.spin_thresh.setDecimals(2)
    win.spin_thresh.setValue(KNOCK_THRESH)
    det_row.addWidget(win.spin_thresh)
    det_row.addWidget(QtWidgets.QLabel("最低有效幅度:"))
    win.spin_min_amp = QtWidgets.QDoubleSpinBox()
    win.spin_min_amp.setRange(0.01, 1.0)
    win.spin_min_amp.setSingleStep(0.01)
    win.spin_min_amp.setDecimals(2)
    win.spin_min_amp.setValue(0.05)
    det_row.addWidget(win.spin_min_amp)
    det_row.addWidget(QtWidgets.QLabel("噪声门限倍数:"))
    win.spin_gate_k = QtWidgets.QDoubleSpinBox()
    win.spin_gate_k.setRange(1.0, 10.0)
    win.spin_gate_k.setSingleStep(0.5)
    win.spin_gate_k.setDecimals(1)
    win.spin_gate_k.setValue(NOISE_K)
    det_row.addWidget(win.spin_gate_k)
    det_row.addWidget(QtWidgets.QLabel("削波提示阈值:"))
    win.spin_clip = QtWidgets.QDoubleSpinBox()
    win.spin_clip.setRange(0.5, 1.0)
    win.spin_clip.setSingleStep(0.01)
    win.spin_clip.setDecimals(2)
    win.spin_clip.setValue(0.98)
    det_row.addWidget(win.spin_clip)
    det_row.addStretch(1)
    cap_layout.addLayout(det_row)

    # 6. 电平 + 状态
    status_row = QtWidgets.QHBoxLayout()
    status_row.addWidget(QtWidgets.QLabel("输入电平:"))
    win.bar_level = QtWidgets.QProgressBar()
    win.bar_level.setRange(0, 100)
    win.bar_level.setValue(0)
    status_row.addWidget(win.bar_level, stretch=3)
    win.lbl_status = QtWidgets.QLabel("未开始")
    status_row.addWidget(win.lbl_status, stretch=2)
    win.lbl_clip = QtWidgets.QLabel("")
    win.lbl_clip.setStyleSheet("color:red;font-weight:bold")
    status_row.addWidget(win.lbl_clip)
    cap_layout.addLayout(status_row)

    # 7. 特征数值面板(顶部, 始终可见)
    feat_grid = QtWidgets.QGridLayout()
    win.feat_names = ["峰值", "RMS", "波峰因子", "偏度", "峰度", "过零率",
                      "上升时间ms", "衰减时间ms", "谱质心Hz", "谱带宽Hz",
                      "谱滚降Hz", "谱平坦度", "谱熵", "谱对比度",
                      "低频比", "中频比", "高频比", "主频Hz"]
    win.feat_labels = {}
    cols = 6
    for i, name in enumerate(win.feat_names):
        r, c = divmod(i, cols)
        lbl = QtWidgets.QLabel("--")
        lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        feat_grid.addWidget(QtWidgets.QLabel(f"{name}:"), r, c * 2)
        feat_grid.addWidget(lbl, r, c * 2 + 1)
        win.feat_labels[name] = lbl
    cap_layout.addLayout(feat_grid)

    # 8. 谱图标签页
    win.tabs = QtWidgets.QTabWidget()

    # 波形对比
    win.plot_wave = pg.PlotWidget(title="波形对比 (灰=原始, 青=降噪+归一化)")
    win.plot_wave.setYRange(-1.1, 1.1)
    win.plot_wave.addLegend()
    win.curve_raw = win.plot_wave.plot(pen=pg.mkPen("gray", width=1), name="原始")
    win.curve_clean = win.plot_wave.plot(pen=pg.mkPen("c", width=2), name="降噪+归一化")
    win.tabs.addTab(win.plot_wave, "波形对比")

    # 频谱图
    win.img_spec = pg.ImageItem(axisOrder="row-major")
    view_spec = pg.PlotItem(title="频谱图(dB)")
    view_spec.addItem(win.img_spec)
    win.plot_spec = pg.ImageView(view=view_spec)
    win.plot_spec.ui.histogram.hide()
    win.plot_spec.ui.roiBtn.hide()
    win.plot_spec.ui.menuBtn.hide()
    win.tabs.addTab(win.plot_spec, "频谱图")

    # 梅尔谱图
    win.img_mel = pg.ImageItem(axisOrder="row-major")
    view_mel = pg.PlotItem(title="梅尔谱图(dB)")
    view_mel.addItem(win.img_mel)
    win.plot_mel = pg.ImageView(view=view_mel)
    win.plot_mel.ui.histogram.hide()
    win.plot_mel.ui.roiBtn.hide()
    win.plot_mel.ui.menuBtn.hide()
    win.tabs.addTab(win.plot_mel, "梅尔谱图")

    # CQT 谱图(对数频率)
    win.img_cqt = pg.ImageItem(axisOrder="row-major")
    view_cqt = pg.PlotItem(title="CQT谱图(对数频率, dB)")
    view_cqt.addItem(win.img_cqt)
    win.plot_cqt = pg.ImageView(view=view_cqt)
    win.plot_cqt.ui.histogram.hide()
    win.plot_cqt.ui.roiBtn.hide()
    win.plot_cqt.ui.menuBtn.hide()
    win.tabs.addTab(win.plot_cqt, "CQT谱图")

    # 对数频谱图
    win.img_log_spec = pg.ImageItem(axisOrder="row-major")
    view_log = pg.PlotItem(title="对数频谱图(dB)")
    view_log.addItem(win.img_log_spec)
    win.plot_log_spec = pg.ImageView(view=view_log)
    win.plot_log_spec.ui.histogram.hide()
    win.plot_log_spec.ui.roiBtn.hide()
    win.plot_log_spec.ui.menuBtn.hide()
    win.tabs.addTab(win.plot_log_spec, "对数频谱图")

    # MFCC
    win.plot_mfcc = pg.PlotWidget(title="MFCC 均值(20 维)")
    win.curve_mfcc = win.plot_mfcc.plot(pen=pg.mkPen("m", width=2))
    win.plot_mfcc.setXRange(0, N_MFCC - 1)
    win.tabs.addTab(win.plot_mfcc, "MFCC")

    # 时域图
    win.plot_time = pg.PlotWidget(title="时域图 (最近 2048 点, 红=包络)")
    win.plot_time.addLegend()
    win.curve_time = win.plot_time.plot(pen=pg.mkPen("y", width=1), name="信号")
    win.curve_env = win.plot_time.plot(pen=pg.mkPen("r", width=2), name="包络")
    win.tabs.addTab(win.plot_time, "时域图")

    # 特征曲线
    win.plot_feat = pg.PlotWidget(title="特征曲线 (归一化, 看形状/相对大小)")
    win.curve_feat = win.plot_feat.plot(pen=pg.mkPen("g", width=2), symbol="o")
    win.tabs.addTab(win.plot_feat, "特征曲线")

    # 能量包络
    win.plot_energy = pg.PlotWidget(title="短时能量包络 (RMS 随时间)")
    win.curve_energy = win.plot_energy.plot(pen=pg.mkPen("y", width=2))
    win.tabs.addTab(win.plot_energy, "能量包络")

    # 谱质心随时间
    win.plot_centroid = pg.PlotWidget(title="谱质心随时间 (Hz)")
    win.curve_centroid = win.plot_centroid.plot(pen=pg.mkPen("c", width=2))
    win.tabs.addTab(win.plot_centroid, "谱质心随时间")

    # 谱通量
    win.plot_flux = pg.PlotWidget(title="谱通量 (频谱变化率)")
    win.curve_flux = win.plot_flux.plot(pen=pg.mkPen("m", width=2))
    win.tabs.addTab(win.plot_flux, "谱通量")

    # Deep Spectrum
    win.plot_ds = pg.PlotWidget(title="Deep Spectrum 特征 (ResNet18, 512 维)")
    win.curve_ds = win.plot_ds.plot(pen=pg.mkPen("c", width=2))
    win.plot_ds.setXRange(0, 511)
    win.tabs.addTab(win.plot_ds, "Deep Spectrum")

    # CNN 特征图(中间层 feature maps)
    win.img_fmap = pg.ImageItem(axisOrder="row-major")
    view_fmap = pg.PlotItem(title="CNN 特征图 (ResNet18 layer3, 前 16 通道)")
    view_fmap.addItem(win.img_fmap)
    win.plot_fmap = pg.ImageView(view=view_fmap)
    win.plot_fmap.ui.histogram.hide()
    win.plot_fmap.ui.roiBtn.hide()
    win.plot_fmap.ui.menuBtn.hide()
    win.tabs.addTab(win.plot_fmap, "CNN 特征图")

    # DS 多信号对比
    win.plot_ds_compare = pg.PlotWidget(title="Deep Spectrum 多信号对比 (512 维)")
    win.plot_ds_compare.addLegend()
    win.plot_ds_compare.setXRange(0, 511)
    win.tabs.addTab(win.plot_ds_compare, "DS 多信号对比")

    cap_layout.addWidget(win.tabs, stretch=1)
    win.main_tabs.addTab(capture_page, "采集")

    # ================= 训练页 =================
    train_page = QtWidgets.QWidget()
    train_layout = QtWidgets.QVBoxLayout(train_page)

    tr_row = QtWidgets.QHBoxLayout()
    tr_row.addWidget(QtWidgets.QLabel("训练数据目录:"))
    win.edit_train_dir = QtWidgets.QLineEdit()
    win.edit_train_dir.setPlaceholderText("训练集数据目录(含 sid_pos.csv)")
    tr_row.addWidget(win.edit_train_dir, stretch=3)
    win.btn_train_dir = QtWidgets.QPushButton("浏览")
    tr_row.addWidget(win.btn_train_dir)
    tr_row.addWidget(QtWidgets.QLabel("标签文件:"))
    win.edit_label_file = QtWidgets.QLineEdit()
    win.edit_label_file.setPlaceholderText("label.csv 路径")
    tr_row.addWidget(win.edit_label_file, stretch=2)
    win.btn_label_file = QtWidgets.QPushButton("浏览")
    tr_row.addWidget(win.btn_label_file)
    tr_row.addWidget(QtWidgets.QLabel("测试比例:"))
    win.spin_test_size = QtWidgets.QDoubleSpinBox()
    win.spin_test_size.setRange(0.1, 0.5)
    win.spin_test_size.setSingleStep(0.05)
    win.spin_test_size.setDecimals(2)
    win.spin_test_size.setValue(0.3)
    tr_row.addWidget(win.spin_test_size)
    tr_row.addWidget(QtWidgets.QLabel("模型:"))
    win.combo_model = QtWidgets.QComboBox()
    win.combo_model.addItem("LightGBM 传统", "lgbm")
    win.combo_model.addItem("PANNs CNN14", "cnn14")
    win.combo_model.addItem("PANNs CNN10", "cnn10")
    win.combo_model.addItem("ResNet18", "resnet18")
    tr_row.addWidget(win.combo_model)
    win.btn_train = QtWidgets.QPushButton("训练模型")
    tr_row.addWidget(win.btn_train)
    win.btn_pause = QtWidgets.QPushButton("暂停训练")
    win.btn_pause.setEnabled(False)
    tr_row.addWidget(win.btn_pause)
    tr_row.addStretch(1)
    train_layout.addLayout(tr_row)

    # 结果保存路径
    res_row = QtWidgets.QHBoxLayout()
    res_row.addWidget(QtWidgets.QLabel("结果保存路径:"))
    win.edit_train_out = QtWidgets.QLineEdit()
    win.edit_train_out.setPlaceholderText("留空则默认 results/train/<时间戳>/")
    res_row.addWidget(win.edit_train_out, stretch=3)
    win.btn_train_out = QtWidgets.QPushButton("浏览")
    res_row.addWidget(win.btn_train_out)
    res_row.addStretch(1)
    train_layout.addLayout(res_row)

    # 深度学习训练参数(仅对深度学习模型生效)
    dl_row = QtWidgets.QHBoxLayout()
    dl_row.addWidget(QtWidgets.QLabel("轮数:"))
    win.spin_dl_epochs = QtWidgets.QSpinBox()
    win.spin_dl_epochs.setRange(1, 500)
    win.spin_dl_epochs.setValue(60)
    dl_row.addWidget(win.spin_dl_epochs)
    dl_row.addWidget(QtWidgets.QLabel("批大小:"))
    win.spin_dl_batch = QtWidgets.QSpinBox()
    win.spin_dl_batch.setRange(0, 256)
    win.spin_dl_batch.setValue(0)
    win.spin_dl_batch.setSpecialValueText("自动")
    dl_row.addWidget(win.spin_dl_batch)
    dl_row.addWidget(QtWidgets.QLabel("学习率:"))
    win.spin_dl_lr = QtWidgets.QDoubleSpinBox()
    win.spin_dl_lr.setRange(0.000001, 0.1)
    win.spin_dl_lr.setDecimals(6)
    win.spin_dl_lr.setSingleStep(0.0001)
    win.spin_dl_lr.setValue(0.0001)
    dl_row.addWidget(win.spin_dl_lr)
    dl_row.addWidget(QtWidgets.QLabel("  "))
    win.chk_use_wandb = QtWidgets.QCheckBox("同步到 wandb")
    win.chk_use_wandb.setChecked(False)
    dl_row.addWidget(win.chk_use_wandb)
    dl_row.addStretch(1)
    train_layout.addLayout(dl_row)

    # 训练进度条(总进度 + 每轮进度)
    prog_row = QtWidgets.QHBoxLayout()
    prog_row.addWidget(QtWidgets.QLabel("总进度:"))
    win.progress_train = QtWidgets.QProgressBar()
    win.progress_train.setRange(0, 100)
    win.progress_train.setValue(0)
    prog_row.addWidget(win.progress_train, stretch=1)
    prog_row.addWidget(QtWidgets.QLabel("本轮:"))
    win.progress_train_epoch = QtWidgets.QProgressBar()
    win.progress_train_epoch.setRange(0, 100)
    win.progress_train_epoch.setValue(0)
    prog_row.addWidget(win.progress_train_epoch, stretch=1)
    train_layout.addLayout(prog_row)

    win.txt_train_log = QtWidgets.QTextEdit()
    win.txt_train_log.setReadOnly(True)
    train_layout.addWidget(win.txt_train_log, stretch=2)

    # 训练曲线(每个曲线单独一页)
    win.tabs_train_curves = QtWidgets.QTabWidget()
    win.plot_train_loss = pg.PlotWidget(title="训练 loss 曲线")
    win.curve_train_loss = win.plot_train_loss.plot(pen=pg.mkPen("r", width=2))
    win.tabs_train_curves.addTab(win.plot_train_loss, "Loss 曲线")
    win.plot_train_acc = pg.PlotWidget(title="测试集准确率曲线")
    win.curve_train_acc = win.plot_train_acc.plot(pen=pg.mkPen("g", width=2))
    win.tabs_train_curves.addTab(win.plot_train_acc, "准确率曲线")
    win.plot_train_metrics = pg.PlotWidget(title="训练指标: 测试集各样本预测分数(按真实类别着色)")
    win.plot_train_metrics.addLegend()
    win.tabs_train_curves.addTab(win.plot_train_metrics, "样本指标")
    train_layout.addWidget(win.tabs_train_curves, stretch=3)
    win.main_tabs.addTab(train_page, "训练")

    # ================= 预测页 =================
    predict_page = QtWidgets.QWidget()
    pred_layout = QtWidgets.QVBoxLayout(predict_page)

    # 模型加载
    model_row = QtWidgets.QHBoxLayout()
    model_row.addWidget(QtWidgets.QLabel("模型文件:"))
    win.edit_model_path = QtWidgets.QLineEdit()
    win.edit_model_path.setPlaceholderText("选择 LightGBM(.pkl) 或深度学习(.pt) 模型")
    model_row.addWidget(win.edit_model_path, stretch=3)
    win.btn_model_path = QtWidgets.QPushButton("加载模型")
    model_row.addWidget(win.btn_model_path)
    pred_layout.addLayout(model_row)
    win.lbl_model_status = QtWidgets.QLabel("未加载模型(将自动使用最新 LightGBM)")
    win.lbl_model_status.setStyleSheet("color:gray;")
    pred_layout.addWidget(win.lbl_model_status)

    # 数据集加载
    data_row = QtWidgets.QHBoxLayout()
    data_row.addWidget(QtWidgets.QLabel("数据集目录:"))
    win.edit_pred_dir = QtWidgets.QLineEdit()
    win.edit_pred_dir.setPlaceholderText("批量预测用(留空则用采集数据)")
    data_row.addWidget(win.edit_pred_dir, stretch=3)
    win.btn_pred_dir = QtWidgets.QPushButton("浏览")
    data_row.addWidget(win.btn_pred_dir)
    pred_layout.addLayout(data_row)

    # 单个文件预测
    single_row = QtWidgets.QHBoxLayout()
    single_row.addWidget(QtWidgets.QLabel("单个文件:"))
    win.edit_single_file = QtWidgets.QLineEdit()
    win.edit_single_file.setPlaceholderText("选择单个 CSV 文件预测")
    single_row.addWidget(win.edit_single_file, stretch=3)
    win.btn_single_file = QtWidgets.QPushButton("浏览")
    single_row.addWidget(win.btn_single_file)
    win.btn_predict_single = QtWidgets.QPushButton("预测该文件")
    single_row.addWidget(win.btn_predict_single)
    pred_layout.addLayout(single_row)

    # 设备选择
    dev_row = QtWidgets.QHBoxLayout()
    win.chk_use_cpu = QtWidgets.QCheckBox("使用 CPU 预测(较慢, 用于 GPU 被占用时)")
    win.chk_use_cpu.setChecked(False)
    dev_row.addWidget(win.chk_use_cpu)
    dev_row.addStretch(1)
    pred_layout.addLayout(dev_row)

    # 预测按钮 + 结果
    pred_row = QtWidgets.QHBoxLayout()
    pred_row.addWidget(QtWidgets.QLabel("成熟度预测:"))
    win.btn_predict = QtWidgets.QPushButton("预测")
    pred_row.addWidget(win.btn_predict)
    pred_row.addStretch(1)
    pred_layout.addLayout(pred_row)

    # 批量预测进度条
    win.progress_predict = QtWidgets.QProgressBar()
    win.progress_predict.setRange(0, 100)
    win.progress_predict.setValue(0)
    pred_layout.addWidget(win.progress_predict)

    win.lbl_pred = QtWidgets.QLabel("预测: --")
    win.lbl_pred.setAlignment(QtCore.Qt.AlignCenter)
    win.lbl_pred.setStyleSheet("font-weight:bold;font-size:48px;color:darkgreen")
    pred_layout.addWidget(win.lbl_pred, stretch=1)

    # 批量预测结果
    win.txt_pred_result = QtWidgets.QTextEdit()
    win.txt_pred_result.setReadOnly(True)
    win.txt_pred_result.setMaximumHeight(160)
    pred_layout.addWidget(win.txt_pred_result)
    win.main_tabs.addTab(predict_page, "预测")

    cmap = pg.colormap.get("magma") or "magma"
    win.img_spec.setColorMap(cmap)
    win.img_mel.setColorMap(cmap)
    win.img_cqt.setColorMap(cmap)
    win.img_log_spec.setColorMap(cmap)
    win.img_fmap.setColorMap(cmap)

    win.btn_start.clicked.connect(win.start)
    win.btn_stop.clicked.connect(win.stop)
    win.btn_data_dir.clicked.connect(win._browse_data_dir)
    win.btn_train_dir.clicked.connect(win._browse_train_dir)
    win.btn_label_file.clicked.connect(win._browse_label_file)
    win.btn_train_out.clicked.connect(win._browse_train_out)
    win.btn_train.clicked.connect(win._train_model)
    win.btn_pause.clicked.connect(win._toggle_pause)
    win.btn_model_path.clicked.connect(win._load_model_file)
    win.btn_pred_dir.clicked.connect(win._browse_pred_dir)
    win.btn_predict.clicked.connect(win._predict_manual)
    win.btn_single_file.clicked.connect(win._browse_single_file)
    win.btn_predict_single.clicked.connect(win._predict_single_file)
    win.btn_preview.clicked.connect(win._browse_preview)
    win.btn_prev.clicked.connect(win._prev_preview)
    win.btn_next.clicked.connect(win._next_preview)
    win.btn_ds_compare.clicked.connect(win._compare_deep_spectrum)
