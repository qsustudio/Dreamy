# Dreamy v2 训练方案（RTX 5090 32G / ≤10 小时）

> 目标：在一张 5090 上、**10 小时以内**、把 32G 显存和 1792GB/s 带宽榨干，
> 训出比 v1.0 明显更好的中文陪伴对话基座。
> 日期：2026-11-08 上午执行。

---

## 一、一页结论

| 项目 | v1.0（4090 实际跑完） | **v2（5090 计划）** | 倍数 |
| --- | --- | --- | --- |
| 参数量 | 110.1M | **127.0M**（dim 832 / depth 12 / heads 13） | 1.15× |
| 训练 token | 0.62B | **2.40B** | **3.9×** |
| D/N（训练充分度） | 5.7（严重欠训练） | **18.9（接近 Chinchilla 20）** | 3.3× |
| 上下文 | 1024 | 1024（不变，够 8-12 轮对话） | — |
| 词表 | 32768 | 32768（**不重训 tokenizer**） | — |
| 注意力 | 手写（物化 S×S 矩阵） | Flash/SDPA（`attn_flash=True`） | — |
| 词表外其他 | — | qk_norm、zero_init_branch、dropout 0 | — |
| 训练时长 | ~5.3h | **9.0h 训练 + 1h 余量** | — |

**这一轮的收益主要来自数据，不是参数。**
v1.0 只喂了 0.62B token，D/N=5.7，模型远没训练充分 —— 这才是 v1 质量的天花板。
参数从 110M 到 127M 只值 ~0.04 nat，而 token 从 0.62B 到 2.4B 值 ~0.3 nat。

---

## 二、为什么是 127M / 2.4B（不是 300M，也不是 145M）

### 2.1 硬约束

训练时长固定 10h ⇒ 能花掉的算力固定：

```
C = 有效峰值 × 时间 = 6 · N · D · f
```

- 5090 BF16 dense 峰值 = **209.5 TFLOPS**（Tensor Core 419 sparse / 2，FP32 104.8 × 2）
- f = 注意力额外开销系数 = `1 + 2·S·dim·depth/N`
  （业界常用的 6ND 只算线性层；S=1024 时注意力还要多花 15~18%，**不修正会高估 15% 的 token 量**）

### 2.2 三档配置实测推算

| 配置 | 参数 | f | FLOP/token | MFU 25% | MFU 30% | MFU 35% | 2.4B 耗时(30%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A `768/12/12` | 110.1M | 1.17 | 0.774 GF | 67.7k tok/s | 81.3k | 94.8k | 8.2h |
| **B `832/12/13`** | **127.0M** | **1.16** | **0.885 GF** | **59.2k** | **71.1k** | **82.9k** | **9.4h** |
| C `896/12/14` | 145.0M | 1.15 | 1.002 GF | 52.3k | 62.7k | 73.2k | 10.6h ✗ |

**B 是唯一在整个 25%~40% MFU 区间都能吃掉 80% 以上数据的配置。**

### 2.3 明天怎么两分钟内定死

上机后先跑 200 步实测（`scripts/bench_throughput.py`），把实测 tok/s 换算成 MFU：

```
MFU = tok/s × 6 × N × f / 209.5e12
```

| 实测 MFU | 选这个 | 2.4B 预计耗时 |
| --- | --- | --- |
| < 28% | A `768/12/12`（110.1M） | 8.8~9.8h |
| **28% ~ 35%** | **B `832/12/13`（127.0M）** | **8.0~9.4h** |
| > 35% | C `896/12/14`（145.0M） | 9.1h |

三档之间的 loss 差距在 0.05 nat 以内（<5% 困惑度），**不要在这上面纠结**；
真正决定成败的是「2.4B token 能不能在 10h 内训完」。

---

## 三、数据配比（实测池子 + 重复轮数）

### 3.1 最终数据配比（11 类，已含新下载的 4.98GB）

| 类别 | 来源 | 清洗后可用 | 本配比预算 | 重复轮数 |
| --- | --- | --- | --- | --- |
| general | opencsg/chinese-fineweb-edu | **1.69B** | 1.09B | 0.65× |
| chat | silver/lccc（短浅闲聊） | 0.222B | 0.288B | 1.30× |
| **chat_v2** | shareAI/ShareGPT-Chinese-90k | 0.084B | 0.120B | 1.43× |
| knowledge | wanng/wikipedia-zh-mnbvc | 0.156B | 0.096B | 0.62× |
| **knowledge_v2** | wikimedia/wikipedia 20231101.zh | 0.54B | 0.312B | 0.58× |
| depth | Mxode/Chinese-Instruct 深度分流 | 0.099B | 0.072B | 0.73× |
| **depth_v2** | wangrui6/Zhihu-KOL | 0.39B | 0.240B | 0.62× |
| empathy | Mxode/Chinese-Instruct 情绪分流 | 0.042B | 0.048B | 1.14× |
| **empathy_v2** | XuShihao6715/counseling-* ×4 | 0.067B | 0.096B | 1.43× |
| safety | Mxode/Chinese-Instruct 安全分流 | 0.011B | 0.014B | 1.31× |
| **safety_v2** | Skepsun/cvalues_rlhf | 0.012B | 0.022B | 1.80× |
| **合计** | | **3.42B** | **2.40B** | **0.70×** |

聚合占比：**general 45.5% / chat 17% / knowledge 17% / depth 13% / empathy 6% / safety 1.5%**

> **这一轮几乎每个类别都只训一遍**（0.58~1.8 轮），不再靠重复硬凑。
> 上半部分是原有数据，标 **粗体** 的是 2026-11-08 凌晨新下载并接入的 4.98GB。

**关键判断：数据不但够，还有富余。** 可用 3.42B，目标 2.40B，用不完的部分留给下一轮。

> general 的 1.69B 是「全部 12 个原始分片都清洗完」的估计值：
> 已清洗 6 个分片得到 1.368B 字符，原始→清洗保留率 83%，剩 6 个分片同理。
>
> chat 的原始池其实**已经基本挖干了**：清洗出 872 万条对话，而原始是 1883 万行
> （LCCC 一问一答约占 941 万条 pair），已经是 93% 的转化率。
> 所以 chat 的缺口不是「没洗完」，而是「raw 里就没有更多」——
> 想降重复轮数只能换更好的对话数据源，多洗没用。

**关键判断：数据基本够，不需要靠大量重复硬凑。**
平均重复 1.1 轮，最狠的 empathy 3.4 轮 —— 都远在 4 轮无损线以内
（Muennighoff et al. 2023：≤4 epochs 近似等價于新数据）。

### 3.2 配比本身（`configs/mix.yaml` → `pretrain_v2`）

```yaml
general:      0.455   # 语言 + 世界知识地基
chat:         0.12    # LCCC 闲聊（短浅，靠量）
chat_v2:      0.05    # ShareGPT 中文：真实用户多轮
knowledge:    0.04    # 旧维基（已耗尽）
knowledge_v2: 0.13    # 官方维基 20231101
depth:        0.03    # 旧深度讨论
depth_v2:     0.10    # 知乎高赞长回答
empathy:      0.02    # 旧情绪数据（keyword 分流）
empathy_v2:   0.04    # 真实心理咨询多轮 —— 陪伴能力的核心
safety:       0.006   # 旧安全数据
safety_v2:    0.009   # CValues 安全偏好正例
max_epochs:   4.0     # 单类重复硬上限，超了就报警要求补数据
```

**设计依据：**
1. **general 45.5%** 是语言能力的底座，不能低。v1 实际跑到 60% 是因为别的类
   提前耗尽、general 被迫兜底 —— 这次改成稀缺类超采样，缺口不再甩给 general。
2. **chat 17% + knowledge 17% + depth 13%** 是"对话为主 + 有深度"的三根支柱。
3. **empathy 6%** 单独拎出来，因为陪伴是差异化能力；而且这次终于有了**真实咨询对话**
   （不再是从 instruct 里按关键词捞出来的）。
4. **depth 加回来**（v1 权重是 0，等于把 15 万条深度讨论数据白扔了）。
5. 旧数据源（无 `_v2` 后缀）保留但占比压低，让位给质量更高的新数据。

### 3.3 代码已支持（本次改动）

- `build_dataset_streaming.py` 新增 `--mix-key pretrain_v2`、`--max-epochs`、`--plan-only`
- **超采样**：某类不够预算时循环重复整类，而不是降低占比
- **重复上限**：超过 `max_epochs` 截断，缺口按剩余产能分摊，分摊也分轮做（不会把接收方也顶爆）
- **续清洗防重复**：`--reuse-shards` 时会跳过已处理过的原始文件
  （旧版会从第一个文件重读，直接把数据洗成重复的 —— 这是个隐藏 bug，已修）
- `meta.json` 现在记录 `epochs_used`，能直接看出每类重复了几轮

---

## 四、今晚要做的事

### 4.0 时间预算

| 步骤 | 预计耗时 | 能不能并行 |
| --- | --- | --- |
| 补数据下载（可选） | 0.5~2h | 可以和清洗并行 |
| 清洗 + 打包 2.4B token | **2~4h**（MinHash 是瓶颈） | 只能顺序 |
| 传数据到 5090 机器 | 15~40min | — |

> 清洗是今晚最贵的一步，**越早开始越好**。可以先不清 depth/general 增量，
> 用 `--plan-only` 先把预算表跑出来看数。

### 4.1 先自检预算（秒级，不清洗）

```bash
python scripts/report_pools.py --include-raw
```

### 4.2 正式构建（已启动，跑 ~2h）

```bash
python -u scripts/build_dataset_streaming.py \
  --stage pretrain \
  --mix-key pretrain_v2 \
  --reuse-shards \
  --clean-only general,depth,chat_v2,empathy_v2,depth_v2,knowledge_v2,safety_v2 \
  --tokenize \
  --max-chars 3890000000 \
  --out-dir data/processed
```

- `--max-chars 3890000000` = 38.9 亿字符 ≈ **2.40B token**
- `--clean-only`：老类别（chat/knowledge/empathy/safety）已经洗完了，直接复用分片；
  只清洗需要补量的 general/旧 depth 和全部 7 个新类别
- **不要加 `--retrain-tokenizer`**：词表必须和 v1 一致（32768）
- 输出 `tokens.bin` ≈ **4.8GB**，`val_tokens.bin` ≈ 48MB
- 旧的 v1 数据集已备份到 `data/processed/_v1_backup/`

> **为什么必须写 `--clean-only`**：`--reuse-shards` 续清洗要靠「最后写出的分片对应
> 哪个原始文件」来猜进度。general 的原始目录没变过，猜得准；但 **chat 的原始目录
> 后来被重新切分过**（清洗时是 4 个大文件，现在是 95 个），文件名对不上会从头重洗一遍，
> 直接把数据洗成重复的。所以老类别一律不续洗。
> 本次改动已经加了 `cleaned/<类>/progress.json` 精确记录进度，以后不会再有这个问题。

跑完检查 `data/processed/meta.json`：
`num_tokens` 应 ≈ 2.4B，`epochs_used` 应全部 ≤ 3.5，`mix_actual` 应贴近 `mix_target`。

> **时间不够时的降级**：`--max-chars` 是可以往下调的。
> 3.89e9 → 2.40B token（要补洗 3 个 general 分片，约 2-3h）
> 3.40e9 → 2.10B token（只要补洗 2 个，省 40min，D/N 降到 16.5，仍健康）
> 下限别低于 3.25e9（2.0B token），再低 127M 就吃不满了。
> 真正耗时的是 general 的 MinHash 近重复（约 6ms/篇），不是 tokenize。

### 4.2b 如果今晚真下载了新数据

新数据**不要**直接丢进 `data/raw/chat/`（会触发上面那个续清洗问题）。
放成新目录，当成新类别：

```bash
# 1) 新数据放进新目录
data/raw/chat_new/part-000.jsonl

# 2) configs/mix.yaml 的 pretrain_v2.categories 里加一行，从原类别挪一点权重过来
#    chat: 0.19  ->  chat: 0.13 / chat_new: 0.06

# 3) 构建时把新类别加进 --clean-only
python -u scripts/build_dataset_streaming.py --stage pretrain --mix-key pretrain_v2 \
  --reuse-shards --clean-only general,depth,chat_new --tokenize --max-chars 3890000000
```

新数据越多，`epochs_used` 里 chat/empathy 的重复轮数越低，效果越好。

### 4.3 可以顺手清理

```bash
# 旧的混合版（1.94GB），v2 构建不再使用
rm data/processed/pretrain_mixed.jsonl
```

---

## 五、明早的执行顺序

### 步骤 0：环境（一次性，~20 分钟）

```bash
# 5090 = sm_120，必须 CUDA 13 + torch 2.14.1；cu126 完全不支持 Blackwell
bash _deploy/setup_5090.sh
```

脚本会自动检测 compute capability 并选对 CUDA 版本，然后打印
`torch.cuda.get_arch_list()` 和一次真实 matmul 自检。
**如果自检不通过就别往下走**，回来找我。

### 步骤 1：传数据

`tokens.bin`(4.8G) + `val_tokens.bin`(48M) + `tokenizer.model` + `meta.json` + SFT 三个文件。

### 步骤 2：200 步实测（关键，别跳过）

```bash
python -u scripts/bench_throughput.py \
  --dim 832 --depth 12 --heads 13 --head-dim 64 --seq-len 1024 \
  --batch-size 16 --grad-accum 2 \
  --steps 60 --warmup 10 --sweep 8,16,24,32 \
  --peak-tflops 209.5 --hours 10 --reserve-hours 1
```

它直接打印：实测 tok/s、反推 MFU、每档 micro-batch 的显存/吞吐、10h 能训多少 token、
以及**可以直接复制的训练命令**（`--max-steps` 已经按 10h 算好）。
拿这个结果对照第二节 2.3 的表决定用 A / B / C —— 如果要换尺寸，重跑一次
bench 换 `--dim/--depth/--heads` 即可。

### 步骤 3：正式训练

```bash
# 用全新 out-dir，避免踩到旧 run 残留的 model_config.json
python -u src/train.py --stage pretrain \
  --out-dir runs_v2 \
  --dim 832 --depth 12 --heads 13 --head-dim 64 --seq-len 1024 \
  --batch-size 16 --grad-accum 2 \
  --lr 4e-4 --warmup-ratio 0.014 \
  --max-steps <bench 给的值> \
  --eval-every 2000 --eval-batch-size 8 --eval-batches 64 \
  --ckpt-every 2000
```

> - `--heads` 就算漏传也不会错：`src/model.py` 会按 `dim/head_dim` 自动纠正并提示
> - **用全新 `--out-dir`**：架构解析优先级是 `ckpt → out_dir/model_config.json → 命令行`，
>   旧的 `runs/` 里如果留着 110M 的配置，会被当成默认值（虽然 CLI 会覆盖，但没必要踩）
> - `--ckpt-every 2000` ≈ 每 11 分钟存一次，掉电最多损失 11 分钟
> - `--resume auto` 挂了直接重跑同一条命令，**不用再传尺寸**（按 ckpt 里的架构自动重建）

### 步骤 4：SFT（约 20-30 分钟）

```bash
# SFT 数据由另一个脚本处理（它的编排逻辑和预训练不同：按条数、要构造 labels）
python -u scripts/build_dataset.py --stage sft --tokenize
python -u src/train.py --stage sft --out-dir runs_v2 \
  --init-from runs_v2/checkpoints/pretrain_last.pt \
  --max-steps <2-3 epoch 对应步数> --lr 5e-5
```

> SFT 只有 3300 万 token，5090 上 2-3 epoch 也就十几分钟。
> **`--init-from` 会自动按 pretrain ckpt 里的架构重建模型**，尺寸参数一个都不用传。
> 模型尺寸如果和 pretrain 不一致会直接报错并列出差异，不会错位加载。

**时间分配**：pretrain 9.0h + SFT 0.4h + 存盘/重启余量 0.6h = 10.0h。

---

## 六、把 5090 榨干的清单

### 已经做的

| 手段 | 收益 | 状态 |
| --- | --- | --- |
| `attn_flash=True`（SDPA `is_causal`） | **2.3×**（v1 的 MFU 只有 15%，因为它物化了 [B,H,S,S]） | ✓ v1.1 |
| `qk_norm` + `zero_init_branch` | 允许更高 LR，收敛更稳 | ✓ v1.1 |
| dropout = 0 | 欠训练阶段 dropout 纯拖慢 | ✓ v1.1 |
| bf16（不是 fp16） | 免 GradScaler，黑维尔 bf16 原生 | ✓ |

### 明天要做的

| 手段 | 预期收益 | 备注 |
| --- | --- | --- |
| **micro-batch 尽量大**（16→24→32 扫一遍） | 3~8% | flash 后注意力不物化，32G 能开很大 |
| **fused AdamW** | 3~6% | 小模型上优化器占比不小 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 防碎片 OOM | 已在 train.py 里设置 |
| `allow_bf16_reduced_precision_reduction` | 1~3% | bf16 GEMM 规约更快 |
| 关掉每步 `synchronize()`，用 CUDA Event 计时 | 1~2% | — |

### 不要做的

- ❌ **梯度检查点**：32G 显存放得下 127M + batch 16，开 checkpointing 只会慢 30%
- ❌ **`torch.compile`**：sm_120 上还不稳，可能吃 10 分钟编译还不返工，收益 10% 不值得冒险
- ❌ **fp8**：5090 支持，但 x-transformers 没接，改造风险远大于收益
- ❌ **加大 seq_len**：用户明确不需要长上下文，而且注意力是 O(S²)，
  1024→2048 在 127M 上要多吃 30% 算力

---

## 七、风险与回退

| 风险 | 概率 | 影响 | 应对 |
| --- | --- | --- | --- |
| cu130/torch 装不上或 sm_120 跑不起来 | 中 | 致命 | `setup_5090.sh` 会自动降到 cu128 + torch 2.11；实在不行换 cu128 三件套 |
| 实测 MFU < 25%（租赁卡降频） | 中 | 少训 30% 数据 | 换 A 档 110M，或者接受 1.9B token |
| 4e-4 学习率发散 | 低 | 浪费时间 | qk_norm 兜着；前 1000 步盯 loss，异常就 3e-4 重开 |
| 清洗没跑完 | 中 | 数据不够 | 用 `--max-chars` 缩小目标（下限 2.0B token = 3.25B 字符） |
| 数据传不上去 | 低 | 卡住 | 提前分卷（`scripts/export_model.py` 有现成的 MD5+分卷逻辑可复用） |
| 训练中途挂掉 | 中 | 损失 ≤11min | `--ckpt-every 2000` + `--resume auto` |

---

## 八、这一轮不做什么（避免范围失控）

- ❌ 不做多轮长上下文 / 思维链 —— 用户明确不需要
- ❌ 不换 tokenizer 词表 —— 收益 ~1%，风险是整条数据链重刷
- ❌ 不做 MoE —— 127M 规模下专家路由只有害无益
- ❌ 不追求 300M+ 参数 —— 10 小时算力只能喂 2.4B token，
  300M 会变成 D/N=8 的欠训练，效果反而不如 127M

---

## 九、数据补给：已下载并接入（2026-11-08 凌晨）

### 9.1 带宽这关是怎么过的

| 时间 | 线路 | 实测 | 结论 |
| --- | --- | --- | --- |
| 23:40 | huggingface.co 直连 | **完全不通**（WinError 10060） | 下不了 |
| 23:40 | hf-mirror.com | 0.15 MB/s | 4.98GB 要 9.2 小时 |
| 23:50（开代理后） | huggingface.co 直连 | **≈10 MB/s** | **4.98GB 用 15.9 分钟** |

结论：**网络是唯一瓶颈，通了就应该下。** 之前按 0.15MB/s 判断"不值得等"是错的。

### 9.2 实际下载 + 转换结果

`scripts/download_v2.py --preset core --endpoint hf` → **4.98GB / 15.9 分钟**，9 个源全部 gated=false。

`scripts/prepare_sources_v2.py --stage both` 把它们转成 Dreamy 类别（**输出到 `<类别>_v2/`，
和旧分片物理隔离，不会覆盖老数据**）：

| 新类别 | 来源 | 条数 | 原始字符 | 估 token |
| --- | --- | --- | --- | --- |
| chat_v2 | shareAI/ShareGPT-Chinese-English-90k | 111,357 | 163.9M | 0.084B |
| empathy_v2 | XuShihao6715/counseling-{cpsdd,psycheval,soulchat-r1,sage} | 84,913 | 130.9M | 0.067B |
| depth_v2 | wangrui6/Zhihu-KOL | 1,002,295 | 756.1M | 0.39B |
| knowledge_v2 | wikimedia/wikipedia 20231101.zh | 1,328,085 | 1050.5M | 0.54B |
| safety_v2 | Skepsun/cvalues_rlhf | 79,157 | 22.9M | 0.012B |

**SFT 阶段也顺带升级了**：新数据同时写进 `data/raw_sft/`（chat +111k、empathy +83k、
depth +997k、boundary +78k），明晚 SFT 会用上真实咨询对话和真实用户多轮。

### 9.3 新增/修改的代码

| 文件 | 作用 |
| --- | --- |
| `scripts/download_v2.py` | 核实过的 9 源下载器，断点续传 + 官方/镜像自动切换 |
| `scripts/prepare_sources_v2.py` | 5 类 record adapter（ShareGPT 的 `{human,assistant}`、咨询的 `client/counselor` 角色、知乎大写 `INSTRUCTION/RESPONSE`、CValues 的 `pos_resp`、维基文档 `text`） |
| `configs/mix.yaml` | `pretrain_v2` 扩成 11 个类别，见 §3 |
| `build_dataset_streaming.py` | 本轮新增 `--clean-only` / `--max-epochs` / 超采样 / `progress.json` 精确续清洗 |

被拒的候选（126GB 实为语音数据、`gated=auto` 拿不到的、LCCC 换皮等）记录在
[docs/v2_data_sources.md](v2_data_sources.md) 里，省得下次重复调研。

---

## 附：关键数字速查

```
5090: 170 SM / 21760 CUDA core / 2.41GHz / 1792 GB/s / 32GB GDDR7 / 575W
      BF16 dense = 209.5 TFLOPS
      sm_120 → 必须 CUDA ≥ 12.8，推荐 CUDA 13 + torch 2.14.1

127.0M (832/12/13/64) : f=1.161, 6Nf=0.885 GF/token
2.40B token           : = 3.89B 字符（中文字/token ≈ 0.617）
32,768 tok/step       : 2.4B → 73,242 步
Chinchilla 参考        : D/N=20 最优；本方案 D/N=18.9
```
