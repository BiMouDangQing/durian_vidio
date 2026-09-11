# reemoon · 榴莲声学成熟度分析工具

基于**敲击声信号**对榴莲成熟度进行三分类（1=生果 / 2=中间果 / 3=熟果）的桌面工具，包含声音采集、信号处理、模型训练（LightGBM + 深度学习）与成熟度预测，并提供友好的 PySide6 图形界面。

---

## 一、功能概览

| 模块 | 说明 |
|------|------|
| 音频清洗 | 调用清洗脚本去除离群/异常音频 |
| 频谱图转换 | 将 CSV/WAV 等数字信号转为频谱图、梅尔频谱并批量预览 |
| 声音采集 | 敲击触发录音 + 实时谱图/特征显示 |
| 训练 | LightGBM 传统模型 / PANNs CNN14 深度学习迁移学习（GPU） |
| 预测 | 加载模型 + 数据集批量预测，或对采集数据即时预测 |

---

## 二、环境要求

- Windows + NVIDIA GPU（可选，深度学习训练需 CUDA）
- conda（本项目环境已放在项目内 `.venv`，Python 3.11）
- 主要依赖：`torch 2.14.0+cu126`、`torchvision 0.29.0+cu126`、`librosa`、`lightgbm`、`PySide6`、`pyqtgraph`、`wandb`

依赖清单见 `requirements.txt`。

---

## 三、快速开始

### 1. 启动界面

双击 `durian_vidio.bat`（完整前端，5 个页签），或双击 `run_qt.bat`（仅采集工具）。

也可命令行启动：

```powershell
D:\model\model\vidio\.venv\python.exe qt\frontend.py
```

### 2. 训练

**方式一（图形界面）**：进入「训练」页签 → 设置数据目录/标签文件 → 选择训练方法（LightGBM / 深度学习 PANNs CNN14）→ 设置参数 → 点「训练模型」。

**方式二（命令行）**：

```powershell
# LightGBM 传统训练
D:\model\model\vidio\.venv\python.exe -m train.pipeline

# 深度学习训练（GPU）
D:\model\model\vidio\.venv\python.exe -m train.dl_train
```

### 3. 预测

进入「预测」页签 → 加载模型（`.pkl` 或 `.pt`）→ 选择数据集目录 → 点「预测」。

---

## 四、数据说明

- 数据目录：`D:\model\data\durian\音频数据\data_all`
- 文件命名：`{样本ID}_{部位}.csv`，样本 ID 21~335 共 315 个，每样本 4 个部位（01/02/03/04）
- 信号：每文件最多 5 条敲击，每条 1024 点 @16kHz
- 标签：`label.csv`（三级标签 1/2/3，按样本 ID 升序对应）

数据路径在 `train/config.py` 中配置（`DATA_PATH` / `LABEL_PATH`）。

---

## 五、目录结构

```
vidio/
├── qt/                 # PySide6 前端
│   ├── frontend.py     # 完整前端入口（5 页签）
│   ├── capture_app.py  # 采集工具入口
│   ├── mainwindow.py   # 采集/训练/预测逻辑
│   ├── ui.py           # 界面构建
│   └── theme.py        # 界面主题（浅蓝+绿）
├── train/              # 训练
│   ├── pipeline.py     # LightGBM 训练主流程
│   ├── dl_train.py     # PANNs CNN14 深度学习训练
│   └── ...
├── predict/            # 预测
│   ├── inference.py    # LightGBM 预测
│   ├── dl_inference.py # 深度学习预测
│   └── loader.py       # 模型自动加载
├── tools/              # 信号处理、频谱图、特征提取
├── md/                 # 项目文档
├── weights/            # 深度学习预训练权重（不入库）
├── results/            # 训练/预测产物（不入库）
├── durian_vidio.bat    # 一键启动完整前端
└── run_qt.bat          # 一键启动采集工具
```

---

## 六、深度学习训练说明

- 模型：PANNs CNN14（AudioSet 预训练迁移），针对 64ms 短敲击信号只对频率维池化
- 输入：梅尔谱图 64 bins × 25 帧
- 策略：冻结低层卷积，微调高层 + 三分类头；数据增强（加噪 + SpecAugment + mixup）
- 预训练权重：`weights/model.safetensors`（首次需下载，约 312 MB）

---

## 七、wandb 训练跟踪

勾选训练页签的「同步到 wandb」，训练记录（loss/acc/lr）会实时上传到 wandb.ai。

首次使用需登录：

```powershell
D:\model\model\vidio\.venv\python.exe -m wandb login
```

---

## 八、常见问题

- **界面启动报 Qt 插件错误**：已自动设置 `QT_PLUGIN_PATH`，若仍报错请确认 `.venv` 中 PySide6 完整安装。
- **GPU 不可用**：确认已安装 CUDA 版 torch（`torch 2.14.0+cu126`），`torch.cuda.is_available()` 返回 True。
- **无模型可预测**：先完成一次训练，或手动加载模型文件。
