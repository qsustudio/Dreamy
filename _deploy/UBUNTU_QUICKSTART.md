# Ubuntu + RTX 4090 快速上手

目标：在实验室的 Ubuntu 4090 机器上，10 小时窗口内训完 Dreamy-110M。

## 0. 先决条件

- NVIDIA 驱动 **525+**（`nvidia-smi` 能正常输出）
- Python **3.10+**（`python3 --version`）
- 能访问 GitHub 和 HuggingFace（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）
- 磁盘：数据 + checkpoint 约 **20 GB**

## 1. 拉代码

```bash
git clone https://github.com/qsustudio/Dreamy.git
cd Dreamy
```

## 2. 装环境 + 自检

```bash
bash _deploy/setup_4090.sh
```

这个脚本会：
- 检查 GPU / 驱动版本
- 建 `.venv`，装 torch 2.14.1+cu126（4090 = sm_89 支持）
- 装项目依赖
- **自检**：确认 CUDA 可用、sm_89 kernel 在构建里、bf16 可用、
  110M 模型能前向+反向、报告显存占用

看到 `自检通过` 就可以继续。

## 3. 准备数据

### 方式 A：本机下载（推荐）

避免从开发机传 13GB。HF 下载速度通常比上传快。

```bash
source .venv/bin/activate

# 下载约 13GB（8 个 Skypile + 3 个维基 + LCCC + 中文指令）
python scripts/download_data.py --preset full

# 转成统一目录结构
python scripts/prepare_sources.py --stage both

# 清洗 + 配比 + 训练 tokenizer + 打包
python scripts/build_dataset.py --stage both --tokenize
```

最后一步会打印**数据健康度报告**，重点看：
- 各类实际占比 vs 目标占比
- 总 token 数（目标 2.0B）
- 告警项

### 方式 B：从开发机传

如果开发机已经构建好，只需传这几个文件到 `data/processed/`：

```
tokens.bin          预训练 token 流（2B token ≈ 4GB）
val_tokens.bin      验证集
meta.json           词表信息
tokenizer.model     SentencePiece 词表
sft_tokens.npy      SFT 数据
sft_labels.npy
sft_meta.json
```

传完直接跳到第 4 步。

## 4. 冒烟训练（必做，5 分钟）

正式开跑前先确认 bf16 稳定、显存够、吞吐符合预期。

```bash
python src/train.py --stage pretrain --max-steps 200
```

看三个数字：
- **loss** 应该在下降，不出现 NaN
- **显存** 应该在 20GB 以内（可加 `--batch-size` 调）
- **吞吐** 报告里的 step 耗时 —— 用它倒推 10 小时能训多少 token

如果 200 步正常，再启动正式训练。

## 5. 正式训练

```bash
# 预训练（主要耗时，约 7-8 小时）
python src/train.py --stage pretrain

# 断线/中断后继续
python src/train.py --stage pretrain --resume auto

# SFT（从预训练最优 checkpoint 出发，约 20-30 分钟）
python src/train.py --stage sft --init-from <输出目录>/checkpoints/pretrain_best.pt
```

`checkpoint` 默认每 2000 步存一次到 `runs/checkpoints/`，
可用 `--ckpt-every` 调整。

## 6. 时间预算参考

| 阶段 | 预计 |
| --- | --- |
| 数据下载 | 15-20 分钟 |
| 数据清洗 + 打包 | 40-60 分钟（CPU，不占 GPU） |
| 环境安装 | 5 分钟 |
| 冒烟训练 | 5 分钟 |
| **预训练** | **约 7-8 小时** |
| SFT | 20-30 分钟 |
| 评测 | 30 分钟 |

⚠️ **数据准备是 CPU 活，不占 GPU**。如果 10 小时窗口很紧，
建议提前在别的机器上做好数据，只把 `tokens.bin` 传过来。

## 常见问题

**`CUDA out of memory`**
调小 micro batch：`--batch-size 8`，必要时加 `--grad-accum 2`
保持有效 batch 不变（8 × 1024 × 2 = 16,384）。

**训练特别慢**
先确认 `自检` 里 bf16 是"支持"。如果回退到 fp16 会慢一些。
另外检查 `nvidia-smi` 里 GPU 利用率是否接近 100%。

**数据报告里某类占比严重偏低**
说明该类数据不够。要么补数据源（`configs/mix.yaml` 里调来源），
要么调低该类目标权重。别硬凑，报告里的告警就是给你看的。

**HuggingFace 下载卡住**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
也可以用 `hf_transfer`（已装）加速。
