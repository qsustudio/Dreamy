# Dreamy

在 **Google Colab T4** 上从零训练的中文陪伴对话模型，基于
[lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)。

- **代码** 走 GitHub（本仓库），Colab 直接 clone
- **训练数据 & 模型权重** 走 Google Drive，不进仓库

## 目录结构

```
Dreamy/
├── README.md
├── requirements.txt              # 依赖（含锁定版本的 x-transformers）
├── colab_train.ipynb             # Colab 一键启动 notebook
├── .gitignore
├── configs/
│   └── mix.yaml                  # 数据配比 + 清洗参数
├── src/
│   ├── config.py                 # 路径/超参配置，读环境变量
│   ├── data.py                   # 数据加载（读 tokens.bin）
│   ├── model.py                  # 模型构建
│   ├── train.py                  # 训练入口
│   └── data_pipeline/            # 数据处理流水线
│       ├── schema.py             # 统一格式（对话 / 文档两种形态）
│       ├── clean.py              # 规范化、质量过滤、去重、脱敏
│       ├── mix.py                # 按配比采样
│       ├── report.py             # 健康度报告
│       └── tokenizer.py          # SentencePiece 训练 + 打包 tokens.bin
└── scripts/
    ├── build_dataset.py          # 一键：清洗 -> 配比 -> tokenizer -> tokens.bin
    └── smoke_test.py             # 本地/Colab 冒烟测试
```

## 数据流水线（从零训练必需）

从零训练和微调最大的区别：**没有现成的 tokenizer，而且通用语料占比必须高**。
所以数据要先经过一道流水线加工。

### 1. 按类别放原始数据

```
data/raw/
├── general/      # 通用中文语料（新闻/百科/小说），纯文本，一行一段
├── chat/         # 开放域闲聊
├── empathy/      # 情感支持 / 共情
├── depth/        # 深度内容（有观点、值得展开的话题）
├── persona/      # 人设 / 角色一致
└── safety/       # 安全边界（危机干预、拒绝）
```

支持 `.jsonl` / `.json` / `.txt`。程序会自动识别这些常见结构：
`turns/messages/conversations` 多轮数组、`instruction/output`、
`question/answer`、以及 `用户:` / `助手:` 交替的纯文本。

### 2. 一条命令构建

```bash
python scripts/build_dataset.py --tokenize
```

产出：

```
data/processed/
├── cleaned/<类别>.jsonl    # 各类清洗结果，可抽查
├── mixed.jsonl             # 按配比混合后的数据
├── tokenizer.model         # SentencePiece 词表
├── tokens.bin              # 扁平 token 流，训练直接读
└── meta.json               # 词表大小等，训练脚本据此对齐模型
```

### 3. 看健康度报告

每次构建会打印报告：各类**目标占比 vs 实际占比**、清洗丢弃原因、
长度分布、预计 T4 训练时长、以及告警。

**这一步别跳过。** 配比失衡和复读数据是训练崩掉最常见的原因，
报告就是为了在开训前把这些暴露出来。

### 配比的默认值

```yaml
general: 0.50    # 通用语料：从零训练学语言的底子
chat:    0.20    # 开放域闲聊
empathy: 0.15    # 情感支持（"陪伴感"的核心）
depth:   0.07    # 深度内容
persona: 0.05    # 人设一致
safety:  0.03    # 安全边界
```

`general` 低于 35% 报告会告警 —— 从零训练时语言能力不稳，多半是这个原因。

配比不是理论最优，调法见下。

### 怎么调配比

1. 用默认值先跑一次，看报告里的实际占比
2. 某类数据不够时会自动欠采，报告里会有偏差告警
3. 要严格按配比（总条数受最稀缺类别限制）：`--strict`
4. 想指定总量：`--total 200000`

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

数据不进仓库（`.gitignore` 已屏蔽 `data/`），通过环境变量指向 Drive：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DREAMY_DATA_DIR` | `/content/drive/MyDrive/dreamy/data` | 训练数据目录（需含 `tokens.bin` + `meta.json`） |
| `DREAMY_OUT_DIR` | `/content/drive/MyDrive/dreamy/runs` | checkpoint / 日志输出 |

本地开发时指向本地目录即可。

> `tokens.bin` 是二进制，别用 Drive 网页上传大文件——直接在 Colab 里跑
> `build_dataset.py` 生成更快，因为原始语料本来就在 Drive 上。

## 训练规模化前先看这个

从零训练要认清一件事：**T4 免费档撑不起真正的通用语言模型**。

粗略估算（MFU ≈ 20%）：100M 参数模型在 T4 上约 50k tokens/s，
8 小时一个 session 能跑约 1.5B token。而 Chinchilla 最优是 20 token/参数，
100M 模型需要约 2B token 才算「训练充分」。

所以现实的做法是：
- 模型规模控制在 **50M~150M**
- 数据以「够用」为目标，先把流程和配比跑通
- 用 2~3 epoch，而不是追求 Chinchilla 最优

## 依赖说明

`requirements.txt` 里 **pinned** 了关键版本，原因：

- `x-transformers==2.31.11` —— 上游 API 变动较快，锁定避免 Colab 上拉到的版本行为不同
- torch 在 Colab 上**不通过 requirements 装**，因为要按 CUDA 版本选择 wheel（见 notebook）
- `sentencepiece` 只在构建数据时需要（训练 tokenizer），但小，直接列为必需

## T4 与 CUDA 版本

T4 是 Turing 架构（`sm_75`）。PyTorch 2.15 起不再发布 cu126 wheel，且 CUDA 13 系列要求
NVIDIA 驱动 ≥580。Colab 的 T4 驱动经常低于 580，所以**推荐 cu126 + torch 2.14.x**。

运行 `scripts/smoke_test.py` 会打印 `sm_75 -> OK` 来确认当前 torch 构建支持 T4。
