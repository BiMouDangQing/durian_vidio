# -*- coding: utf-8 -*-
"""
audio_cleaner 桌面前端

PySide6 图形界面，单一窗口五个页签：
1. 音频清洗：配置并调用 audio_cleaner.py
2. 频谱图转换：将 CSV/TXT/NPY/WAV 等音频数字信号转换为频谱图与梅尔频谱，并批量预览
3. 声音采集：敲击触发录音 + 实时谱图/特征显示
4. 训练：榴莲成熟度分类模型训练
5. 预测：成熟度预测
"""

import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import spectrogram  # noqa: E402
from mainwindow import MainWindow as CaptureTool  # noqa: E402

# 兼容 PySide6 与 PyQt5
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    QT_LIB = "PySide6"
except ImportError:  # pragma: no cover
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        QT_LIB = "PyQt5"
    except ImportError as exc:
        raise ImportError(
            "未找到 PySide6 或 PyQt5，请先安装其一，例如：pip install PySide6"
        ) from exc


# 清洗脚本路径
CLEANER = ROOT / "audio_cleaner.py"


def current_python() -> str:
    """优先使用当前解释器，失败时退回 python。"""
    return sys.executable or "python"


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频数据处理工具 - 前端")
        self.resize(760, 560)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_clean_tab(), "音频清洗")
        self.tabs.addTab(self._build_spectrogram_tab(), "频谱图转换")

        # 声音采集 / 训练 / 预测（整合自 mainwindow.py 的采集工具）
        self.tool = CaptureTool()
        for _ in range(self.tool.main_tabs.count()):
            w = self.tool.main_tabs.widget(0)
            title = self.tool.main_tabs.tabText(0)
            self.tool.main_tabs.removeTab(0)
            self.tabs.addTab(w, title)

    # ------------------------------------------------------------------ #
    # 页签一：音频清洗
    # ------------------------------------------------------------------ #
    def _build_clean_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # 输入目录
        layout.addWidget(QtWidgets.QLabel("输入目录（音频）:"))
        row_in = QtWidgets.QHBoxLayout()
        self.edit_input = QtWidgets.QLineEdit()
        btn_input = QtWidgets.QPushButton("浏览…")
        btn_input.clicked.connect(lambda: self._pick_dir(self.edit_input))
        row_in.addWidget(self.edit_input)
        row_in.addWidget(btn_input)
        layout.addLayout(row_in)

        # 输出目录
        layout.addWidget(QtWidgets.QLabel("输出目录（结果）:"))
        row_out = QtWidgets.QHBoxLayout()
        self.edit_output = QtWidgets.QLineEdit()
        self.edit_output.setText(str(ROOT / "clean_result"))
        btn_output = QtWidgets.QPushButton("浏览…")
        btn_output.clicked.connect(lambda: self._pick_dir(self.edit_output))
        row_out.addWidget(self.edit_output)
        row_out.addWidget(btn_output)
        layout.addLayout(row_out)

        # 参数区
        form = QtWidgets.QFormLayout()
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(["iforest", "lof", "both"])
        form.addRow("检测方法:", self.combo_method)

        self.spin_contamination = QtWidgets.QDoubleSpinBox()
        self.spin_contamination.setRange(0.001, 0.5)
        self.spin_contamination.setSingleStep(0.01)
        self.spin_contamination.setValue(0.05)
        form.addRow("污染率:", self.spin_contamination)
        layout.addLayout(form)

        # 运行按钮
        self.btn_run = QtWidgets.QPushButton("开始清洗")
        self.btn_run.clicked.connect(self.run)
        layout.addWidget(self.btn_run)

        # 日志区
        layout.addWidget(QtWidgets.QLabel("日志:"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        return tab

    # ------------------------------------------------------------------ #
    # 页签二：频谱图转换
    # ------------------------------------------------------------------ #
    def _build_spectrogram_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # 文件选择按钮
        row_btns = QtWidgets.QHBoxLayout()
        btn_add_files = QtWidgets.QPushButton("添加文件…")
        btn_add_files.clicked.connect(self._add_spec_files)
        btn_add_dir = QtWidgets.QPushButton("添加目录…")
        btn_add_dir.clicked.connect(self._add_spec_dir)
        btn_remove = QtWidgets.QPushButton("移除选中")
        btn_remove.clicked.connect(self._remove_spec_selected)
        btn_clear = QtWidgets.QPushButton("清空")
        btn_clear.clicked.connect(self._clear_spec_list)
        for b in (btn_add_files, btn_add_dir, btn_remove, btn_clear):
            row_btns.addWidget(b)
        row_btns.addStretch()
        layout.addLayout(row_btns)

        # 文件列表
        self.list_spec = QtWidgets.QListWidget()
        layout.addWidget(self.list_spec)

        # 参数区
        form = QtWidgets.QFormLayout()
        self.spin_sr = QtWidgets.QSpinBox()
        self.spin_sr.setRange(1000, 384000)
        self.spin_sr.setValue(22050)
        form.addRow("采样率(文本/数组):", self.spin_sr)

        self.combo_kind = QtWidgets.QComboBox()
        self.combo_kind.addItem("两者都要", "both")
        self.combo_kind.addItem("频谱图 (STFT)", "stft")
        self.combo_kind.addItem("梅尔频谱", "mel")
        form.addRow("转换类型:", self.combo_kind)

        row_out = QtWidgets.QHBoxLayout()
        self.edit_spec_out = QtWidgets.QLineEdit()
        self.edit_spec_out.setText(str(ROOT / "spectrogram_out"))
        btn_spec_out = QtWidgets.QPushButton("浏览…")
        btn_spec_out.clicked.connect(lambda: self._pick_dir(self.edit_spec_out))
        row_out.addWidget(self.edit_spec_out)
        row_out.addWidget(btn_spec_out)
        form.addRow("输出目录:", row_out)
        layout.addLayout(form)

        # 转换按钮
        self.btn_spec_run = QtWidgets.QPushButton("开始转换")
        self.btn_spec_run.clicked.connect(self.run_spectrogram)
        layout.addWidget(self.btn_spec_run)

        # 预览导航
        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("上一张")
        self.btn_prev.clicked.connect(lambda: self._nav_image(-1))
        self.btn_next = QtWidgets.QPushButton("下一张")
        self.btn_next.clicked.connect(lambda: self._nav_image(1))
        self.lbl_img_info = QtWidgets.QLabel("尚未生成预览")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_img_info)
        nav.addStretch()
        layout.addLayout(nav)

        # 图片预览区
        self.img_label = QtWidgets.QLabel()
        self.img_label.setMinimumHeight(280)
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setText("转换完成后此处显示频谱图预览")
        layout.addWidget(self.img_label)

        # 日志区
        self.spec_log = QtWidgets.QPlainTextEdit()
        self.spec_log.setReadOnly(True)
        self.spec_log.setMaximumHeight(120)
        layout.addWidget(self.spec_log)

        self._images: list[str] = []
        self._img_index = -1

        return tab

    def _pick_dir(self, line_edit: QtWidgets.QLineEdit):
        start = line_edit.text().strip() or str(ROOT)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", start)
        if path:
            line_edit.setText(path)

    def _log(self, text: str):
        self.log.appendPlainText(text)

    def run(self):
        input_dir = self.edit_input.text().strip()
        if not input_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择输入目录。")
            return
        if not CLEANER.exists():
            QtWidgets.QMessageBox.warning(
                self, "提示", f"未找到清洗脚本：{CLEANER}\n请先创建 audio_cleaner.py。"
            )
            return

        cmd = [
            current_python(),
            str(CLEANER),
            "--input_dir", input_dir,
            "--output_dir", self.edit_output.text().strip() or str(ROOT / "clean_result"),
            "--method", self.combo_method.currentText(),
            "--contamination", str(self.spin_contamination.value()),
        ]

        self._log("执行命令: " + " ".join(cmd))
        self.btn_run.setEnabled(False)
        self.btn_run.setText("清洗中…")
        QtCore.QCoreApplication.processEvents()

        try:
            # 实时回显输出
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(ROOT),
            )
            for line in proc.stdout:
                self._log(line.rstrip())
                QtCore.QCoreApplication.processEvents()
            proc.wait()
            self._log(f"完成，退出码: {proc.returncode}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"运行出错: {exc}")
        finally:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("开始清洗")

    # ------------------------------------------------------------------ #
    # 频谱图转换：槽函数
    # ------------------------------------------------------------------ #
    def _add_spec_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择输入文件",
            str(ROOT),
            "支持格式 (*.csv *.txt *.dat *.tsv *.npy *.npz *.wav *.flac *.mp3 *.ogg *.m4a *.aac *.aiff *.au *.opus);;所有文件 (*)",
        )
        existing = {self.list_spec.item(i).text() for i in range(self.list_spec.count())}
        for f in files:
            if f not in existing:
                self.list_spec.addItem(f)

    def _add_spec_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", str(ROOT))
        if not path:
            return
        base = Path(path)
        existing = {self.list_spec.item(i).text() for i in range(self.list_spec.count())}
        added = 0
        for f in sorted(base.rglob("*")):
            if f.is_file() and spectrogram.is_supported(f) and str(f) not in existing:
                self.list_spec.addItem(str(f))
                added += 1
        self._spec_log(f"从目录添加了 {added} 个文件")

    def _remove_spec_selected(self):
        for item in self.list_spec.selectedItems():
            self.list_spec.takeItem(self.list_spec.row(item))

    def _clear_spec_list(self):
        self.list_spec.clear()

    def _spec_paths(self) -> list[str]:
        return [self.list_spec.item(i).text() for i in range(self.list_spec.count())]

    def _spec_log(self, text: str):
        self.spec_log.appendPlainText(text)

    def run_spectrogram(self):
        paths = self._spec_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(self, "提示", "请先添加输入文件。")
            return

        out_dir = self.edit_spec_out.text().strip() or str(ROOT / "spectrogram_out")
        kind = self.combo_kind.currentData()
        sr = self.spin_sr.value()

        self._images = []
        self._img_index = -1
        self._spec_log(f"开始转换 {len(paths)} 个文件，类型={kind}，采样率={sr}")
        self.btn_spec_run.setEnabled(False)

        results = []
        for i, p in enumerate(paths):
            self._spec_log(f"[{i + 1}/{len(paths)}] {Path(p).name}")
            r = spectrogram.process_file(p, out_dir, kind=kind, sr=sr)
            results.append(r)
            if r["ok"]:
                for png in r["outputs"].values():
                    self._images.append(png)
            else:
                self._spec_log(f"  失败: {r['error']}")
            QtWidgets.QCoreApplication.processEvents()

        ok = sum(1 for r in results if r["ok"])
        self._spec_log(f"完成：成功 {ok}/{len(results)}")

        # 生成汇总预览图
        if ok:
            preview_kind = "mel" if kind == "mel" else "stft"
            preview_path = spectrogram.render_preview(
                results, Path(out_dir) / "preview.png", kind=preview_kind
            )
            if preview_path:
                self._images.insert(0, preview_path)
                self._spec_log(f"预览图: {preview_path}")

        self.btn_spec_run.setEnabled(True)
        if self._images:
            self._show_image_index(0)

    def _nav_image(self, step: int):
        if not self._images:
            return
        self._show_image_index(self._img_index + step)

    def _show_image_index(self, index: int):
        n = len(self._images)
        if n == 0:
            self.lbl_img_info.setText("无图片")
            self.img_label.setText("无图片")
            return
        index = index % n
        self._img_index = index
        path = Path(self._images[index])
        self._show_image(str(path))
        self.lbl_img_info.setText(f"[{index + 1}/{n}] {path.name}")

    def _show_image(self, path: str):
        if not path or not Path(path).exists():
            self.img_label.setText("图片不存在")
            return
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            self.img_label.setText("无法加载图片")
            return
        pm = pm.scaled(
            self.img_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.img_label.setPixmap(pm)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
