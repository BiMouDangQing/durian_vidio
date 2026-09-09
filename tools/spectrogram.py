# -*- coding: utf-8 -*-
"""
音频数字信号 → 频谱图 / 梅尔频谱 转换与批量预览工具

支持输入格式：
- 文本数值：CSV / TXT / DAT（纯数值矩阵，可用 `--delimiter` / `--skip_header` / `--column` 控制）
- 数组文件：NPY / NPZ（取第一个数组）
- 音频文件：WAV / FLAC / MP3 等（自动读取文件自带采样率）

两个转换功能：
1. 频谱图（STFT 幅度谱，线性 Hz 频率轴）
2. 梅尔频谱（Mel 刻度频率轴）

支持批量处理，并在处理后自动生成一张汇总预览图（preview.png）。

注意：文本/数组文件不含采样率信息，需通过 `--sr`（默认 22050）指定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口后端，避免与 Qt 前端冲突
import matplotlib.pyplot as plt

# 中文字体支持（Windows 下优先使用微软雅黑 / 黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    HAS_LIBROSA = False

DEFAULT_SR = 22050
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".au", ".opus"}
TEXT_EXTS = {".csv", ".txt", ".dat", ".tsv", ".asc"}


# --------------------------------------------------------------------------- #
# 信号读取
# --------------------------------------------------------------------------- #
def _extract_column(data: np.ndarray, column: int | None) -> np.ndarray:
    """从二维数值矩阵中提取信号列。

    - 单列 → 直接返回
    - 指定 column → 取该列
    - 两列且第一列单调递增（常见 [索引/时间, 幅度] 格式）→ 取第二列
    - 其余情况 → 取第一列
    """
    data = np.asarray(data)
    if data.ndim == 1:
        return data.astype(np.float32)
    if data.shape[1] == 1:
        return data[:, 0].astype(np.float32)
    if column is not None:
        return data[:, column].astype(np.float32)
    if data.shape[1] == 2:
        first = data[:, 0]
        if np.all(np.diff(first) >= 0) and np.any(np.diff(first) > 0):
            return data[:, 1].astype(np.float32)
    return data[:, 0].astype(np.float32)


def load_signal(
    path: str | Path,
    sr: int = DEFAULT_SR,
    delimiter: str | None = None,
    skip_header: int = 0,
    column: int | None = None,
) -> tuple[np.ndarray, int]:
    """读取信号，返回 (signal, sample_rate)。

    - 音频文件：采样率取自文件本身
    - 文本 / 数组文件：采样率使用参数 `sr`
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in AUDIO_EXTS:
        if not HAS_LIBROSA:
            raise RuntimeError("处理音频文件需要安装 librosa")
        y, file_sr = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(y, dtype=np.float32), int(file_sr)

    if ext == ".npy":
        data = np.load(path)
        return _extract_column(data, column), int(sr)

    if ext == ".npz":
        arr = np.load(path)
        key = arr.files[0]
        return _extract_column(arr[key], column), int(sr)

    # 文本数值格式
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    data = np.nan_to_num(data)
    if data.ndim == 0:
        raise ValueError(f"未读取到有效数据：{path}")
    return _extract_column(data, column), int(sr)


def is_supported(path: str | Path) -> bool:
    ext = Path(path).suffix.lower()
    return ext in AUDIO_EXTS or ext in TEXT_EXTS or ext in {".npy", ".npz"}


# --------------------------------------------------------------------------- #
# 谱计算
# --------------------------------------------------------------------------- #
def compute_stft_db(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """计算 STFT 频谱图（幅度谱，单位 dB）。"""
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    return librosa.amplitude_to_db(S, ref=np.max)


def compute_mel_db(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """计算梅尔频谱（单位 dB）。"""
    M = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(M, ref=np.max)


# --------------------------------------------------------------------------- #
# 绘图与保存
# --------------------------------------------------------------------------- #
def render_single(
    signal: np.ndarray,
    sr: int,
    kind: str = "stft",
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
):
    """渲染单张频谱图，返回 matplotlib Figure。kind: 'stft' | 'mel'"""
    if kind == "mel":
        spec = compute_mel_db(signal, sr, n_fft, hop_length, n_mels)
        y_axis = "mel"
    else:
        spec = compute_stft_db(signal, sr, n_fft, hop_length)
        y_axis = "hz"

    fig, ax = plt.subplots(figsize=(8, 4))
    mappable = librosa.display.specshow(
        spec, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis, ax=ax
    )
    fig.colorbar(mappable, ax=ax, format="%+2.0f dB")
    return fig


def save_fig(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_preview(
    results: list[dict],
    out_path: str | Path,
    kind: str = "stft",
    cols: int = 4,
) -> str | None:
    """把批量转换结果拼成一张汇总预览图。results 为 process_file 返回的列表。"""
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        return None

    n = len(ok_results)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4), squeeze=False)
    for idx, r in enumerate(ok_results):
        ax = axes[idx // cols][idx % cols]
        png = r.get("outputs", {}).get(kind) or next(iter(r.get("outputs", {}).values()), None)
        if png and Path(png).exists():
            img = plt.imread(png)
            ax.imshow(img, aspect="auto")
        ax.set_title(Path(r["input"]).name, fontsize=8)
        ax.axis("off")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(f"批量预览 - {kind.upper()}", fontsize=12)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# 单文件 / 批量处理
# --------------------------------------------------------------------------- #
def process_file(
    in_path: str | Path,
    out_dir: str | Path,
    kind: str = "both",
    sr: int = DEFAULT_SR,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    **load_kwargs,
) -> dict:
    """处理单个文件，返回结果字典。kind: 'stft' | 'mel' | 'both'"""
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        signal, file_sr = load_signal(in_path, sr=sr, **load_kwargs)
        kinds = ["stft", "mel"] if kind == "both" else [kind]
        outputs: dict[str, str] = {}
        for k in kinds:
            fig = render_single(signal, file_sr, k, n_fft, hop_length, n_mels)
            png = out_dir / f"{in_path.stem}_{k}.png"
            save_fig(fig, png)
            outputs[k] = str(png)
        return {
            "input": str(in_path),
            "outputs": outputs,
            "ok": True,
            "error": None,
            "sr": int(file_sr),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "input": str(in_path),
            "outputs": {},
            "ok": False,
            "error": str(exc),
            "sr": None,
        }


def process_files(
    paths: list[str | Path],
    out_dir: str | Path,
    kind: str = "both",
    sr: int = DEFAULT_SR,
    cols: int = 4,
    preview: bool = True,
    **load_kwargs,
) -> tuple[list[dict], str | None]:
    """批量处理，返回 (结果列表, 汇总预览图路径或 None)。"""
    results = [process_file(p, out_dir, kind=kind, sr=sr, **load_kwargs) for p in paths]
    preview_path = None
    if preview:
        preview_kind = "mel" if kind == "mel" else "stft"
        preview_path = render_preview(results, Path(out_dir) / "preview.png", kind=preview_kind, cols=cols)
    return results, preview_path


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将 CSV/TXT/NPY/WAV 等音频数字信号转换为频谱图与梅尔频谱，并批量预览。"
    )
    parser.add_argument("inputs", nargs="+", help="输入文件路径（可多个）")
    parser.add_argument("--out_dir", default="./spectrogram_out", help="输出目录")
    parser.add_argument("--kind", default="both", choices=["stft", "mel", "both"], help="转换类型")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="文本/数组数据的采样率（默认 22050）")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--cols", type=int, default=4, help="预览图每行列数")
    parser.add_argument("--delimiter", default=None, help="文本分隔符（默认自动识别）")
    parser.add_argument("--skip_header", type=int, default=0, help="跳过的表头行数")
    parser.add_argument("--column", type=int, default=None, help="取第几列作为信号（默认自动识别）")
    args = parser.parse_args(argv)

    load_kwargs = {
        "delimiter": args.delimiter,
        "skip_header": args.skip_header,
        "column": args.column,
    }
    results, preview_path = process_files(
        args.inputs,
        args.out_dir,
        kind=args.kind,
        sr=args.sr,
        cols=args.cols,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        **load_kwargs,
    )

    ok = sum(1 for r in results if r["ok"])
    print(f"完成：成功 {ok}/{len(results)}")
    for r in results:
        if r["ok"]:
            for k, p in r["outputs"].items():
                print(f"  [OK] {Path(r['input']).name} -> {k}: {p}")
        else:
            print(f"  [FAIL] {Path(r['input']).name}: {r['error']}")
    if preview_path:
        print(f"预览图: {preview_path}")

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
