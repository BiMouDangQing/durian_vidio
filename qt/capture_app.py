# -*- coding: utf-8 -*-
"""声音采集工具入口。

用法:
    python qt/qt.py   (或双击 run_qt.bat)
"""

import os
import sys
import datetime


# ---------- 日志初始化(写文件, 同时保留控制台输出) ----------
def _setup_logging():
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"qt_{ts}.log")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    try:
        log_file = open(log_path, "a", encoding="utf-8")
    except Exception:
        return None

    out_streams = [log_file]
    err_streams = [log_file]
    if sys.stdout is not None:
        out_streams.append(sys.stdout)
    if sys.stderr is not None:
        err_streams.append(sys.stderr)
    sys.stdout = _Tee(*out_streams)
    sys.stderr = _Tee(*err_streams)

    def _excepthook(tp, val, tb):
        import traceback
        msg = "".join(traceback.format_exception(tp, val, tb))
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except Exception:
            pass
        if sys.__stderr__ is not None:
            try:
                sys.__stderr__.write(msg + "\n")
            except Exception:
                pass

    sys.excepthook = _excepthook
    return log_path


_LOG_PATH = _setup_logging()

# 先加载 torch(Deep Spectrum 依赖), 避免其 DLL 与 PyQt5 的初始化顺序冲突
try:
    import torch  # noqa: F401
except Exception:
    pass

# 确保 fqb 目录在 sys.path(以便 from qt.mainwindow import MainWindow)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# conda 环境下 pip 装的 PySide6 无法自动定位 Qt 插件, 显式设置 QT_PLUGIN_PATH
import PySide6 as _pyside6
_qt_plugins = os.path.join(os.path.dirname(_pyside6.__file__), "plugins")
if os.path.isdir(_qt_plugins):
    os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins)

from PySide6 import QtWidgets
from qt.mainwindow import MainWindow
from qt.theme import QSS, APP_TITLE


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.setWindowTitle(APP_TITLE)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
