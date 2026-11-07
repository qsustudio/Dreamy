# Dreamy v1.0 训练记录（可复现）

> 2026-11-07 首轮训练。模型权重在下载时损坏、且实例已释放导致不可恢复。
> 本文档记录**完整可复现的配置**，用于重训。
> 所有数据与代码仍在开发机上完好。

## 一、最终结果（首轮）

| 项目 | 值 |
| --- | --- |
| 模型 | 110.18M 参数 |
| 预训练 | 38,000 步 / 0.62B token，最终 loss **4.55**（ppl 94） |
| SFT | 16,000 步 / 2 epoch，loss 约 **3.5** |
| 身份效果 | ✅ "我是Dreamy，由清夏开发训练的一个中文模型。" |
| 生成质量 | ✅ 采样解码下连贯、无复读、能收尾 |
| 硬件 | RTX 4090 24GB，驱动 580，约 6.5 小时 |

## 二、数据（产物在开发机 `data/processed/`）

| 文件 | 大小 | 内容 |
| --- | --- | --- |
| `tokens.bin` | 2,032 MiB | 1,065,628,342 token 预训练 |
| `val_tokens.bin` | 20.6 MiB | 10,806,604 token 验证 |
| `sft_tokens.npy` | 64.8 MiB | (33,229, 1024) uint16 |
| `sft_labels.npy` | 129.6 MiB | (33,229, 1024) int32，-100 = 不计 loss |
| `sft_val_*.npy` | 2.0 MiB | 333 条 |
| `tokenizer.model` | 754,677 B | SentencePiece BPE，词表 32768 |
| `meta.json` | 756 B | vocab_size=32768, seq_len=1024 |

**预训练配比（实际）**：
general 60.2% / chat 20.5% / knowledge 14.5% / empathy 3.9% / safety 1.0%

**数据来源**（全部 HF 公开已清洗数据集）：
- `opencsg/chinese-fineweb-edu`（Skypile）→ general
- `wanng/wikipedia-zh-mnbvc` → knowledge
- `silver/lccc` → chat
- `Mxode/Chinese-Instruct` → empathy / safety / SFT

**SFT 配比**：chat 31.9% / empathy 26.6% / depth 21.2% / opinion 10.6% /
boundary 5.3% / followup 4.2% / identity 3.1%（1,272 条，上采样 8 倍，关去重）

## 三、复现步骤

```bash
# 0. 拉代码 + 装环境
git clone https://github.com/qsustudio/Dreamy.git
cd Dreamy
bash _deploy/setup_4090.sh

# 1. 传数据到 data/processed/
#    tokens.bin / val_tokens.bin / meta.json / tokenizer.model
#    sft_tokens.npy / sft_labels.npy / sft_val_*.npy / sft_meta.json

# 2. 预训练（5.3 小时）
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python -u src/train.py --stage pretrain --max-steps 38000 \
    --batch-size 4 --grad-accum 4 \
    --eval-every 0 --ckpt-every 1000 > train.log 2>&1 &

# 3. SFT（1 小时）
nohup python -u src/train.py --stage sft \
    --init-from runs/checkpoints/pretrain_last.pt \
    --epochs 2 --max-steps 16000 \
    --eval-every 0 --ckpt-every 2000 > sft.log 2>&1 &

# 4. 导出 + 校验（见 scripts/export_model.py）
python scripts/export_model.py
```

## 四、关键参数（不要改）

| 参数 | 值 | 原因 |
| --- | --- | --- |
| micro batch | **4** | 8 会 OOM（24GB 装不下 8192×1024 的激活） |
| grad accum | **4** | 有效 batch = 16384 token |
| `--eval-every` | **0** | eval 用 batch 32 会 OOM，必须关 |
| AMP | **bf16** | 4090 原生支持，不用 GradScaler |
| 预训练 lr | 3e-4 | warmup 2%，余弦到 3e-5 |
| SFT lr | 5e-5 | warmup 3%，余弦到 5e-6 |

**推理/解码参数（实测调出来的）**：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| temperature | **0.75** | |
| top_p | **0.9** | |
| repetition_penalty | **1.25** | 消除复读的关键 |
| 解码方式 | **必须采样** | 贪心必定无限复读 |

## 五、踩过的坑

1. **`arch in get_arch_list()` 误判 4090**：CUDA 二进制兼容是「X.y 的 kernel 能跑在
   X.z (z>=y)」，sm_89 跑 sm_86 即可。已改成实测 matmul 判断。
2. **eval batch 32 OOM**：训练 batch 4 没事，eval 翻 8 倍就爆。用 `--eval-every 0` 绕开。
3. **`python ... > log &` 看起来卡住**：Python 块缓冲，必须 `python -u`。
4. **SFT 数据 dedup 吃掉上采样**：identity 类已加进 `NO_DEDUP_CATEGORIES`。
5. **权重下载损坏**：大小正确但内容出错（32/129 块坏）。
   → **传输后必须校验 MD5，2GB 文件建议分块校验**。

## 六、下次必须做

- [ ] 导出后立刻算 MD5，下载完立刻核对
- [ ] 大文件分 50MB 块传输 + 逐块 MD5
- [ ] 权重下载成功后**先备份两份**再释放实例
