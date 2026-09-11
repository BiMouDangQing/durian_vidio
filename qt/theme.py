# -*- coding: utf-8 -*-
"""界面主题: 浅蓝色 + 绿色配色, reemoon 品牌。

在 frontend.py / capture_app.py 的 main() 中通过 `app.setStyleSheet(QSS)` 应用。
"""

BRAND = "reemoon"

APP_TITLE = "reemoon · 榴莲声学成熟度分析工具"

QSS = """
/* ================= 全局 ================= */
* {
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QWidget {
    background-color: #eef6f9;
    color: #1d3b4a;
}

/* ================= 品牌横幅 ================= */
QFrame#brandBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #42a5f5, stop:0.5 #26c6da, stop:1 #66bb6a);
    border-radius: 6px;
}
QFrame#brandBar QLabel {
    background: transparent;
    color: #ffffff;
    font-size: 20px;
    font-weight: bold;
}

/* ================= 页签 ================= */
QTabWidget::pane {
    border: 1px solid #c3dfe8;
    background: #ffffff;
    border-radius: 4px;
}
QTabBar::tab {
    background: #d5eaf2;
    color: #2c5a6e;
    padding: 8px 22px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:hover { background: #c0e0ec; }
QTabBar::tab:selected {
    background: #43a047;
    color: #ffffff;
    font-weight: bold;
}

/* ================= 按钮 ================= */
QPushButton {
    background-color: #43a047;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 7px 16px;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover { background-color: #4caf50; }
QPushButton:pressed { background-color: #388e3c; }
QPushButton:disabled { background-color: #b5c4cb; color: #eceff1; }

/* ================= 输入控件 ================= */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #b7d5e0;
    border-radius: 4px;
    padding: 5px 8px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #42a5f5;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #4caf50;
}

/* ================= 日志 / 文本 ================= */
QPlainTextEdit, QTextEdit {
    background-color: #f6fbfd;
    border: 1px solid #c3dfe8;
    border-radius: 4px;
    padding: 4px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}

/* ================= 列表 ================= */
QListWidget {
    background-color: #ffffff;
    border: 1px solid #c3dfe8;
    border-radius: 4px;
}
QListWidget::item { padding: 3px; }
QListWidget::item:selected {
    background-color: #b3e5fc;
    color: #0d2b38;
}

/* ================= 进度条 ================= */
QProgressBar {
    border: 1px solid #b7d5e0;
    border-radius: 4px;
    background: #ffffff;
    text-align: center;
    height: 16px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #42a5f5, stop:1 #4caf50);
    border-radius: 3px;
}

/* ================= 复选框 ================= */
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 15px; height: 15px; }

/* ================= 分组框 ================= */
QGroupBox {
    border: 1px solid #c3dfe8;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #2c5a6e;
}
"""
