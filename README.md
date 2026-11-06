# Dreamy

在 **Google Colab T4** 上训练的 Transformer 项目，基于
[lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)。

- **代码** 走 GitHub（本仓库），Colab 直接 clone
- **训练数据 & 模型权重** 走 Google Drive，不进仓库

## 目录结构

```
Dreamy/
├── README.md
├── requirements.txt          # 依赖（含锁定版本的 x-transformers）
├── colab_train.ipynb         # Colab 一键启动 notebook
├── .gitignore
├── src/
│   ├── config.py             # 路径/超参配置，读环境变量
│   ├── data.py               # 数据加载（从 Drive 读）
│   ├── model.py              # 模型构建
│   └── train.py              # 训练入口
└── scripts/
    └── smoke_test.py         # 本地/Colab 冒烟测试
```

## 本地开发（Windows，无 GPU）

```powershell
# 1. 建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 装依赖（CPU 版 torch 见下方说明）
pip install -r requirements.txt

# 3. 冒烟测试（不需要 GPU，也不需要数据）
python scripts\smoke_test.py
```

> 本机没有 NVIDIA GPU，`pip install torch` 默认装 CUDA 版会很大。开发机上装 CPU 版即可：
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Colab 上训练

1. 把训练数据传到 Google Drive，例如 `MyDrive/dreamy/data/`
2. 打开 `colab_train.ipynb`（或用 `File > Open notebook > GitHub` 直接从本仓库打开）
3. **运行时 → 更改运行时类型 → T4 GPU**
4. 依次运行各单元格

notebook 会：
- 挂载 Drive
- clone 本仓库
- 安装依赖（自动选 CUDA wheel）
- 校验 T4 的 sm_75 kernel 可用
- 启动训练，checkpoint 写回 Drive

## 数据放哪

数据不进仓库，通过环境变量指向 Drive：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DREAMY_DATA_DIR` | `/content/drive/MyDrive/dreamy/data` | 训练数据目录 |
| `DREAMY_OUT_DIR` | `/content/drive/MyDrive/dreamy/runs` | checkpoint / 日志输出 |

本地开发时指向本地目录即可。

## 依赖说明

`requirements.txt` 里 **pinned** 了关键版本，原因：

- `x-transformers==2.31.11` —— 上游 API 变动较快，锁定避免 Colab 上拉到的版本行为不同
- torch 在 Colab 上**不通过 requirements 装**，因为要按 CUDA 版本选择 wheel（见 notebook）

## T4 与 CUDA 版本

T4 是 Turing 架构（`sm_75`）。PyTorch 2.15 起不再发布 cu126 wheel，且 CUDA 13 系列要求
NVIDIA 驱动 ≥580。Colab 的 T4 驱动经常低于 580，所以**推荐 cu126 + torch 2.14.x**。

运行 `scripts/smoke_test.py` 会打印 `sm_75 -> OK` 来确认当前 torch 构建支持 T4。
