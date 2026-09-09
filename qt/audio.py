# -*- coding: utf-8 -*-
"""录音缓冲(QIODevice 子类)。"""

from PySide6.QtCore import QIODevice


class AudioBuffer(QIODevice):
    """QAudioInput 的音频落点: Qt 音频线程调用 writeData 写入录音字节。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = bytearray()

    def writeData(self, data):
        self._data.extend(bytes(data))
        return len(data)

    def readData(self, maxlen):
        return bytes(0)

    def pop_all(self):
        """取出并清空已录字节, 返回 bytes。"""
        data = bytes(self._data)
        self._data.clear()
        return data
