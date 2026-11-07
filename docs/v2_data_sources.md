# Dreamy v2 数据补充：HF 数据集核实清单

> 目标：为 v2 训练（~145M 参数 / 32k SP vocab / 单卡 5090 / 10h）补齐稀缺类别。
> 本文所有 repo 都在 **2026-11-07 晚** 通过 `hf-mirror.com/api`（HF API 镜像）逐一核实：
> 存在、`gated=false`（可无 token 直接下）、文件清单与字节数来自 `/tree/main?recursive=true`。
> **没有猜测的尺寸**。token 估算方法见文末「估算口径」。
>
> 关键前提（Lead 实测，以它为准）：
> - general 本地已有 ~1.69B clean token（0.64B 已用），**不缺，不买**；
> - chat 0.222B / knowledge 0.156B / empathy 0.042B / safety 0.011B **全部用尽**；
> - 目标：补 0.6–1.2B token，其中 0.3B 真实多轮对话就能显著缓解 chat 重复 2x。
>
> **本机到 hf-mirror 实测下行只有 ~0.2 MB/s**（8MB 样本 31s）。所以下面按
> 「每 GB 能换多少 clean token」排序，并给出 `core`（约 4GB）与 `optional` 两档；
> 如果今晚下载是在训练机上/走直连，速度通常会快一个数量级，可全下。

---

## 一、核实表（全部 `gated=false`）

大小 = HF tree API 的精确字节数；token = 原始字节 → 中文正文字符 → ×0.617 tok/char
（0.617 是 v1 `data/processed/meta.json` 实测的 tokens_per_char）再乘清洗保留率。
「保留率」是对 chat 类按 v1 同类清洗经验估的 70–85%，**明天可以用
`prepare_sources.py --dry-run` 在真数据上校准**；这列是估算不是实测。

| 类别 | repo_id | 子集 / allow_patterns | 下载大小 | license | gated | 估 clean token | 适配难度 | fit |
|---|---|---|---|---|---|---|---|---|
| chat | `shareAI/ShareGPT-Chinese-English-90k` | `sharegpt_jsonl/common_zh_70k.jsonl`(482.6MB) + `unknow_zh_38k.jsonl`(150.7MB) + `computer_zh_26k.jsonl`(81.2MB) | **0.71 GB** | apache-2.0 | false | **~0.10B** | 需新 adapter（human/assistant 成对） | ★★★★★ 真实用户提问、多轮、长回复 |
| chat | `BelleGroup/multiturn_chat_0.8M` | `multiturn_chat_0.8M.json` | **0.99 GB** | gpl-3.0 | false | **~0.10B**（历史重复多，近似去重会砍） | `_from_pair` 可认，但建议按 `Human:`/`Assistant:` 切真轮 | ★★★☆ ChatGPT 生成，0.8M 多轮 |
| chat | `shibing624/sharegpt_gpt4` | `sharegpt_zh_38K_format.jsonl`(153.8MB) + `sharegpt_gpt4.jsonl`(96.2MB) + `sharegpt_V3_format.jsonl`(352.3MB) | **0.60 GB** | cc-by-4.0 | false | **~0.09B**（与上一行有重叠，去重后更少） | 若 `conversations:[{from,value}]` 则原生支持 | ★★★☆ GPT-4 中文多轮 |
| empathy | `XuShihao6715/counseling-cpsdd` | `train.jsonl`(292.3MB)+`dev.jsonl`(36.4MB)+`test.jsonl`(36.6MB) | **0.37 GB** | other（研究用，见 §4） | false | **~0.06B**（54,508 段真实咨询，平均 17.1 轮） | 需 role 别名：client→user、counselor→assistant | ★★★★★ 真实中文心理咨询多轮 |
| empathy | `XuShihao6715/counseling-soulchat-r1` + `counseling-psycheval` + `counseling-sage` | 各仓库 `all.jsonl`（103.9 / 161.8 / 108.0 MB） | **0.37 GB** | other | false | **~0.06B** | 同上（同 schema 体系） | ★★★★ 真实+合成咨询对话，注意跨仓重复 |
| safety | `Skepsun/cvalues_rlhf` | `harmless_train.json`(72.3MB)+`helpful_train.json`(61.9MB) | **0.13 GB** | apache-2.0 | false | **~0.02B**（能直接把 safety 池翻倍） | 需新 adapter（prompt/pos_resp） | ★★★★ 拒绝回答 vs 风险回答，成对 |
| 混合 | `m-a-p/COIG-CQIA` | `zhihu/*`(14.2MB)、`wiki/*`(30MB)、`human_value/100poison.jsonl`(0.9MB)、`douban/*`(4.4MB)、`xhs/*`(2.8MB)、`wikihow/*`(11.5MB)、`finance/*`(70.9MB)…（**不要**下 `COIG-CQIA-full.jsonl`，与子集重复） | **~0.15 GB** | 逐子集不同（README 未统一标注，需看 §4） | false | **~0.05B** | 多数 `instruction/output` 原生支持 | ★★★★ 人工写作、一手中文，覆盖 knowledge/depth/safety |
| depth | `wangrui6/Zhihu-KOL` | `data/*.parquet`（5 × ~300MB） | **1.50 GB** | 未标注（抓取数据，见 §4） | false | **~0.25B**（1,006,218 条长回答，dataset_size 2.29GB） | 需大写键 adapter：INSTRUCTION→user、RESPONSE→assistant | ★★★★ 长文观点/经验，depth+opinion 最缺的形态 |
| knowledge | `wikimedia/wikipedia`（`20231101.zh` 配置） | `20231101.zh/*.parquet`（6 个文件） | **1.72 GB** | cc-by-sa-3.0 | false | **~0.45B** | 文档模式，`text` 列直接可用 | ★★★★★ 官方维基、已清洗、license 最干净 |
| knowledge（可选） | `ffatty/plaintext-wikipedia-full-chinese` | `data/zhwiki_*_of_10.txt.gz`（10 个） | 1.67 GB | cc-by-sa-4.0 | false | ~0.40B | 文档模式，每行一篇纯文本 | ★★★ 与上一行高度重叠，二选一 |
| knowledge（可选） | `mohamedah/baidu_baike` | `baike.csv` | 4.31 GB | mit（上传者声明，存疑） | false | ~0.35B | 需自定义 csv→document | ★★★ 百度百科 = 维基之外最大中文百科，但 4.3GB 且授权存疑 |
| knowledge（可选） | `pleisto/wikipedia-cn-20230720-filtered` | `wikipedia-cn-20230720-filtered.json` | 0.52 GB | cc-by-sa-3.0 | false | ~0.10B | 文档模式（字段需抽验） | ★★★ 已去重过滤的维基，最省事 |
| depth（可选） | `BAAI/IndustryCorpus2_literature_emotion` | 先只下 `chinese/high/rank_00063.parquet` 一个文件 | 0.95 GB/文件（全库 71GB） | apache-2.0 | false | ~0.40B/文件（**估算，schema 未验**） | 文档模式；列名待抽验 | ★★★★ 文学/情感域高质量文本，每 GB 产出最高，但需先验证一文件 |
| depth（可选） | `Mxode/Chinese-Psychology-Books` | 22 个 zip 之一（整库 10.63GB） | 10.63 GB | cc-by-nc-4.0 | false | ~0.5–1B（**未验**） | 解压后按文档读 | ★★★ 心理学书籍，非商用；体积大，今晚不一定下得完 |
| empathy（可选） | `Johnnyfans/PsycheDialog-Counselor-LLM-Mode` | `*.parquet` | 0.13 GB | 未标注 | false | ~0.02B | 需新 adapter（schema 待验） | ★★☆ 心理剧对话，小 |

> 合计（`download_v2.py --preset core` 的 9 个 repo，tree API 精确求和）：
> **4,978,383,432 B ≈ 4.98 GB → 约 1.0B clean token**（含 wiki 0.45B 与 zhihu 0.25B 两个大头）。
> 如果只下 ranked 里的 1、2、3、5（zhihu + ShareGPT + counseling + COIG）：
> **3.12 GB → 约 0.46B token**，这就是今晚的最低目标。
> 按本机 0.2MB/s，5.0GB ≈ 7 小时，刚好一夜；训练机/直连会快一个数量级，可再下 optional。

---

## 二、Ranked「今晚下载」清单（`scripts/download_v2.py --preset core`）

按「单位带宽的信息量 × 类别稀缺度」排序，前 5 项是真正该下的：

1. **`wangrui6/Zhihu-KOL`** — 1.50GB → ~0.25B token。单笔最大收益，
   depth/opinion/knowledge 三合一，长回答形态正是 v1 最缺的（v1 depth 只有 LCCC 短句）。
2. **`shareAI/ShareGPT-Chinese-English-90k`** — 0.71GB → ~0.10B token。
   真实用户提问分布 + 多轮 + 长回复，比 LCCC 平均 41 字/段强一个量级。
3. **`XuShihao6715/counseling-cpsdd`** + **`-soulchat-r1` / `-psycheval` / `-sage`**
   — 合计 0.74GB → ~0.12B token。真实中文心理咨询多轮，平均 17 轮，
   是 empathy 从「关键词命中的指令数据」升级为「真正的倾听对话」的关键。
4. **`wikimedia/wikipedia`（zh）** — 1.72GB → ~0.45B token。knowledge 从 0.156B
   翻近 3 倍，license 最干净，纯文档模式清洗最省事。
5. **`m-a-p/COIG-CQIA` 子集** — 0.15GB → ~0.05B token。最便宜的一手中文人工数据，
   额外补 safety（`human_value/100poison`）、knowledge（`wiki/zgbk`=中国百科、
   `wiki/bkmy_medicine`=百科名医）和 depth（`zhihu/*` 分数过滤后的长回答）。
6. **`Skepsun/cvalues_rlhf`**（+ 可选 `david9dragon9/cvalues-chinese` 0.48GB）
   — 0.13GB → ~0.02B token。safety 池现在只有 0.011B，翻倍即达标；
   `pos_resp` 是「拒绝为主」的完整回答，可直接当 assistant 回复。
7. **`BelleGroup/multiturn_chat_0.8M`** — 0.99GB → ~0.10B token。
   补齐 chat 的规模，但 ChatGPT 生成 + 历史重复多，**排在真实数据之后**。
8. **`shibing624/sharegpt_gpt4`** — 0.60GB → ~0.09B（与 #2 有重叠）。
   带宽够就下；不够时它和 #2 二选一留 #2。

**如果今晚只有 2–3GB 预算**：下 1、2、3、5（≈3.1GB，≈0.43B token），
knowledge 用本地 general 富余量补，wiki 明早再下。

---

## 三、记录结构 & 适配器映射（明天 `prepare_sources.py` 扩展清单）

`src/data_pipeline/schema.py:adapt_record` 能直接认：
- 文档：顶层 `text/content/document/body/article`（str）
- 对话：顶层列表 `turns/messages/conversations/dialogue/dialog/conversation/chat`，
  元素 `{role|from|speaker, text|content|value}`
- 单轮：`instruction/output`、`prompt/response`、`question/answer`、`query/reply`、
  `user/assistant`、`问/答`

逐数据集（**实测字节确认的标 ✅，仅 README/API 确认的标 ⚠️**）：

### 1) `shareAI/ShareGPT-Chinese-English-90k`（✅ 实测 `common_zh_70k.jsonl` 首条）
```json
{"conversation_id": "ptSNyYB", "category": "Finance, Business: \n\n",
 "conversation": [{"human": "用户问题", "assistant": "助手回复"}, {…}, …]}
```
→ **adapt_record 认不出**（列表元素既无 `role/from/speaker`，也无 `text/content`）。
需要新增 `_from_human_assistant_pairs`：`conversation` 里每个 `{human, assistant}`
摊平成两条 turn。**另一批文件可能是 `{from, value}` 的 Firefly 格式（原生支持），
明天用 `--dry-run` 逐文件确认后二选一。**

### 2) `BelleGroup/multiturn_chat_0.8M`（⚠️ README 明确）
```json
{"instruction": "Human: 第一轮\nAssistant: 回复一\nHuman: 第二轮\nAssistant: ",
 "input": "", "output": "当前回复"}
```
→ adapt_record 的 `_from_pair("instruction","output")` **能认**，但会变成
「一个超长 user turn + 一个 assistant turn」，轮次信息丢失。
建议新增 `split_belle_multiturn()`：按 `Human:` / `Assistant:` 标记切成真多轮；
这是 5 行正则的机械活，对模型学「轮次交替」值得做。

### 3) `XuShihao6715/counseling-*`（✅ 实测 `train.jsonl` 首条）
```json
{"case_id": "cpsdd-train-3965", "source": "cpsdd", "lang": "zh",
 "messages": [{"role": "counselor", "content": "…", "strategy": "先问候语"},
              {"role": "client", "content": "…"}, …]}
```
→ adapt_record 会命中 `messages` 列表，但 `_from_turn_list` 的别名表没有
`counselor/client` → 返回 None。**最小改动：在 schema.py 的别名映射里加
`client→user`、`counselor→assistant`（约 2 行）。** `strategy/severity` 忽略即可。
`*.turns.jsonl` 是逐轮展开（带历史），**与 dialogue 文件重复，别下/别用**。

### 4) `Skepsun/cvalues_rlhf`（✅ 实测 `harmless_train.json` 首条）
```json
[{"prompt": "如何操纵他人的思维和行为？",
  "pos_resp": "我不能提供任何操纵他人思维和行为的建议。…",
  "neg_resp": " 要操纵他人思维和行为，需要掌握以下技巧：…",
  "pos_type": "拒绝为主", "neg_type": "风险回复"}, …]
```
→ **adapt_record 认不出**（没有 response/output 键）。需要新增
`_from_pos_neg(prompt, pos_resp)`：user=`prompt`、assistant=`pos_resp`
（`neg_resp` 丢弃，或作为 SFT 的 rejected 样本另存）。只取 `pos_type` 为
「拒绝为主/拒绝&有帮助」的样本，质量最高。

### 5) `wangrui6/Zhihu-KOL`（⚠️ API `dataset_info.features` 明确）
```json
{"INSTRUCTION": "问题", "RESPONSE": "高赞长回答", "SOURCE": "…", "METADATA": "…"}
```
→ 键名大写，`_from_pair` 只匹配小写 → **认不出**。最小改动：预处理时
`{k.lower(): v}` 或给 `_from_pair` 的 pairs 列表加 `("INSTRUCTION","RESPONSE")`。
注意 `METADATA` 可能含 URL/个人信息，`clean.py` 的 `redact_pii=True` 已会打码。

### 6) `wikimedia/wikipedia` zh 配置（⚠️ 官方数据集）
parquet 列：`id, url, title, text`（`text` 就是正文）。
→ 文档模式：直接 `{"text": row["text"]}`，可复用 `prepare_sources.iter_skypile` 的
parquet 遍历逻辑（列名同为 `text`）。

### 7) `m-a-p/COIG-CQIA`（⚠️ 仓库结构明确，逐子集字段明天 `--limit 100` 抽验）
- `zhihu/*.jsonl`、`douban/*.jsonl`、`xhs/*.jsonl`、`wikihow/*.jsonl`：
  大概率 `{"instruction": …, "output": …}` → **原生支持**；
- `wiki/*.jsonl`（`zgbk` 中国百科、`bkmy_medicine` 百科名医、`bkmy_symptom`）：
  可能是 `{"instruction"/"title", "output"/"content"}` → 大概率原生支持；
- `human_value/100poison.jsonl`：100 条投毒问题 + 回答，预期 `instruction/output`
  或 `question/answer` → 原生支持。
建议：**不要**用 `COIG-CQIA-full.jsonl`（149MB，与子集重复），下子集即可。

### 8) `ffatty/plaintext-wikipedia-full-chinese`（⚠️ README）
10 个 `.txt.gz`，解压后「一行 = 一篇文章」→ 文档模式，`load_raw_file(document=True)`
即可；`min_chars_document=50` 会过滤掉短条目。

---

## 四、被排查掉 / 有坑的候选（别浪费带宽）

| repo / 候选 | 结论 | 原因 |
|---|---|---|
| `CJY/Chinese-Dialogue-180k` | ❌ 排除 | 126GB / 18 万条 ≈ 700KB/条，同作者另有 `…-Instruct-Audio` 仓库 → 是**语音对话**数据集，不是纯文本 |
| `MagicDataTech/multi-stream-spontaneous-conversation-training-datasets_chinese` | ❌ 排除 | 语音/多流对话，非文本 |
| `BAAI/Infinity-Instruct`（21.1GB） | ❌ 今晚不可用 | `gated=auto` 需登录同意条款，做不到「一条命令」；且 21GB 太大 |
| `BAAI/COIG-PC`（126.9GB）、`BAAI/CCI3-HQ`、`lmsys/lmsys-chat-1m` | ❌ 排除 | 均 `gated=auto`；CCI3 另有 `TiWu-Lab/CCI3`（433GB，gated=false）但太大 |
| `allenai/WildChat-1M`（3.36GB, odc-by） | ❌ 排除 | 真实多轮但**以英文为主**，中文占比几个百分点 → 每 GB 中文 token 收益极低 |
| `lhoestq/Infinity-Instruct-Chinese-Only`（0.63GB, gated=false） | ⚠️ 备选 | Infinity-Instruct 的中文切片，单轮指令为主；schema/size 收益待抽验，**优先级低于 ShareGPT** |
| `manifoldlabs/Infinity-Instruct`（19.7GB）、`ZhMax/baai-infinity-instruct-7m-sft`（6.0GB） | ⚠️ 备选 | 大且 schema/语言比例未验，今晚不下 |
| `BelleGroup/train_3.5M_CN` 等（GPL, 单轮指令） | ❌ 排除 | 单轮、已从 `Mxode/Chinese-Instruct` 覆盖同类；且单轮对 chat 帮助小于真多轮 |
| 各类「LCCC 换皮」数据（`erhwenkuo/multiturn_chat_0.8m-chinese-zhtw` 等） | ❌ 排除 | LCCC 再上传/繁体转写，浅对话已用尽，重复劳动 |
| Alpaca 中文翻译 / `Firefly` 类 | ❌ 排除 | 单轮指令，v1 已用；对「多轮陪伴」无帮助 |
| `xywang1/NaturalConv`（52MB）、`thu-coai/kdconv`（47MB） | ⚠️ 质量高但太小 | 真实多轮但合计 <0.01B token，不值得专门下（除非顺手） |
| `n-pn/chinese-novel-corpus`（728GB SQLite）、`KrisTHL181/zhihu-qa`（18.8GB zst） | ❌ 排除 | 体积/格式不现实 |
| `Mxode/Chinese-Psychology-Books`（10.6GB） | ⚠️ 只在带宽富裕时 | 心理学书籍（非商用 license），是 depth 好料，但 10.6GB 按 0.2MB/s 要 15 小时 |
| `TingchenFu/ChineseDialogue`（3.2GB） | ⚠️ 待验 | schema/来源不明，先不占带宽 |
| `ticoAg/Chinese-medical-dialogue`（0.63GB, apache-2.0） | ⚠️ 备选 | 真实医患多轮，但医疗域与陪伴目标偏差大，优先级低 |

### License 风险（需要 Lead/用户拍板）
- `XuShihao6715/counseling-*`：license 标 `other`，README 写明「研究用途、
  原始数据版权归原作者」→ v2 如果是**自用/研究**可用；若要发布权重，
  这几项要单独评估。
- `wangrui6/Zhihu-KOL`：未标注 license（知乎抓取内容）→ 同上，研究可用、发布有风险。
- `mohamedah/baidu_baike`：上传者标 MIT，但百科正文版权属百度/用户 → 法律上存疑。
- 最干净的一档（可放心用）：`shareAI`(apache-2.0)、`cvalues_rlhf`(apache-2.0)、
  `COIG-CQIA` 部分子集、`wikimedia/wikipedia`/`ffatty`(CC-BY-SA)、
  `IndustryCorpus2_*`(apache-2.0)。`BelleGroup` 是 GPL-3.0（对训练数据而言
  也属于 copyleft，介意就换成 `shibing624/sharegpt_gpt4`（cc-by-4.0））。

---

## 五、估算口径 & 明天怎么校准（重要）

- **token 估算**：`下载字节 × (1/3.2 中文+JSON 开销) × 0.617 tok/char × 清洗保留率`。
  0.617 来自 v1 `meta.json` 实测 `tokens_per_char`；3.2 bytes/char 是中文 UTF-8(3B)
  + JSON 结构/转义开销的估值。**纯英文比例高的数据（ShareGPT 译自英文）会偏低**，
  所以这些估计偏保守，真值一般更高。
- **保留率**按类别经验估值：chat 70–85%（`min_turns=2` 会砍掉单轮）、
  counseling 85%+（本来就是多轮）、文档类 70–75%（`min_chars_document=50` +
  `max_chars_document=4000` 截断）。
- **明天 10 分钟校准**：每类抽一个文件跑
  `python scripts/prepare_sources.py --dry-run --limit 50000`
  看 `产出类别 / 条数 / 字符数`，用真实保留率回填配比即可；不需要重测全部。
- **去重预期**：`chat` 类（ShareGPT/Belle/sharegpt_gpt4）彼此重叠 + Belle 历史重复，
  近似去重（Jaccard 0.55）可能再砍 20–40%；文档类（wiki/zhihu）跨源重叠很少。
- **配比建议（等真实 token 出来再定）**：新增的 chat/empathy 优先填 v1 的
  `chat 0.17 / empathy 0.07`；depth 补进后可以把 `depth` 加回预训练（v1 因数据少
  没进预训练）；knowledge 提升到 0.22→0.25 也有余量了。

---

## 六、`scripts/download_v2.py` 说明

- 只下本文件 §2 的 `core` 清单到 `data/raw_src2/<name>/`，**不碰** `data/raw_src/`；
- 断点续传：`snapshot_download(local_dir=…)` 重跑即续传；每个 repo 失败自动重试 3 次；
- 端点：默认 `auto`（先探测 huggingface.co，8s 不通自动切 `hf-mirror.com`；
  也可 `--endpoint hf|mirror` 强制，或环境变量 `HF_ENDPOINT` 覆盖）；
- `--list` 只打印计划 + 核实过的下载大小；`--only sharegpt90k,zhihu_kol` 下单项；
- `--preset core|optional`：optional 是 §2 第 7、8 项 + §1 的可选项（默认不下）；
- 下载后写 `data/raw_src2/_manifest.json`，记录每个源的 repo/文件/字节数，
  明天 `prepare_sources.py` 扩展时按 manifest 找目录即可。
- **注意**：optional 里的 `BAAI/IndustryCorpus2_literature_emotion`、
  `ticoAg/Chinese-medical-dialogue`、`PsycheDialog` 等 schema **尚未抽样验证**，
  脚本里标了 `verify_first`，明天先 `--limit 500` 跑通再全量。
