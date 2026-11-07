"""流式数据构建：清洗 -> 配比 -> tokenizer -> tokens.bin

和旧版的区别：**全程不把整类数据读进内存**。

为什么必须这样：全量数据有 2180 万条 / 45 亿字符，
一次性读进 Python 需要 25GB 以上内存，16GB 机器直接 OOM。
这里改成「按分片处理 + 边读边写」，常驻内存控制在几百 MB。

流程::

    1. 清洗各类 -> cleaned/<类别>/part-NNNN.jsonl（分片落盘，只保留计数）
    2. 按字符预算决定各类取多少
    3. 抽样训练 tokenizer
    4. 流式读分片 -> 编码 -> 按哈希切 train/val -> 分批写 tokens.bin

用法::

    python scripts/build_dataset.py --stage pretrain --tokenize
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "data_pipeline"))

from clean import CleanConfig, CleanStats, Deduper, clean_conversations  # noqa: E402
from mix import normalize_weights                                       # noqa: E402
from report import format_report                                        # noqa: E402
from schema import Conversation, dump_jsonl, load_raw_file, read_jsonl  # noqa: E402
from tokenizer import (                                                 # noqa: E402
    IGNORE_INDEX, ROLE_TO_TOKEN, SPECIAL_TOKENS, END_TOKEN,
    train_tokenizer, write_meta,
)

LENIENT_CATEGORIES = {"safety", "boundary"}
STAGE_DIRS = {"pretrain": "raw", "sft": "raw_sft"}
SHARD_SIZE = 20000          # 每片多少条
IGNORE = IGNORE_INDEX


# ---------------------------------------------------------------- 工具
def load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_files(raw_dir: str, category: str) -> list[str]:
    d = os.path.join(raw_dir, category)
    if not os.path.isdir(d):
        return []
    return [
        os.path.join(d, n) for n in sorted(os.listdir(d))
        if os.path.isfile(os.path.join(d, n)) and not n.startswith(".")
    ]


def build_clean_config(q: dict, document: bool, lenient: bool) -> CleanConfig:
    return CleanConfig(
        min_turns=q.get("min_turns", 2),
        max_turns=q.get("max_turns", 40),
        max_chars=q.get("max_chars", 6000),
        min_chars_per_turn=q.get("min_chars_per_turn", 1),
        max_repeat_ngram_ratio=q.get("max_repeat_ngram_ratio", 0.15),
        repeat_ngram_n=q.get("repeat_ngram_n", 5),
        max_punct_ratio=q.get("max_punct_ratio", 0.5),
        min_unique_char_ratio=q.get("min_unique_char_ratio", 0.05),
        exact_dedup=q.get("exact_dedup", True),
        near_dedup=q.get("near_dedup", True),
        near_dedup_similarity=q.get("near_dedup_similarity", 0.55),
        min_chars_for_near_dedup=q.get("min_chars_for_near_dedup", 80),
        redact_pii=q.get("redact_pii", True),
        lenient=lenient,
        document=document,
        min_chars_document=q.get("min_chars_document", 50),
        max_chars_document=q.get("max_chars_document", 4000),
    )


def shard_stats(shards: list[str]) -> tuple[int, int]:
    """统计已有分片里的 (字符数, 条数)。"""
    chars = 0
    items = 0
    for sh in shards:
        for conv in read_jsonl(sh):
            chars += conv.total_chars
            items += 1
    return chars, items


def last_consumed_source(shards: list[str]) -> str | None:
    """最后一个已有分片对应的原始文件名。

    续清洗时必须知道上次读到哪个原始文件了 —— 否则会从第一个文件
    重新读一遍，把已经清洗过的数据再洗一次写一遍，数据直接重复。
    """
    if not shards:
        return None
    with open(shards[-1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            return (rec.get("meta") or {}).get("source")
    return None


def progress_path(shard_dir: str) -> str:
    return os.path.join(shard_dir, "progress.json")


def read_progress(shard_dir: str) -> set[str]:
    """读「已经清洗过哪些原始文件」的进度清单。

    没有这份清单就必须靠猜（见 resume_files 的注释），而猜错两次的代价
    是数据重复或者漏数据，所以清洗时一定把它写下来。
    """
    path = progress_path(shard_dir)
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).get("processed_files") or [])
    except (OSError, json.JSONDecodeError):
        return set()


def write_progress(shard_dir: str, processed: set[str], chars: int) -> None:
    try:
        with open(progress_path(shard_dir), "w", encoding="utf-8") as f:
            json.dump({"processed_files": sorted(processed),
                       "cleaned_chars": int(chars)}, f, ensure_ascii=False)
    except OSError:
        pass          # 进度文件写不了不该让整次清洗失败


def resume_files(files: list[str], existing: list[str]) -> tuple[list[str], int]:
    """续清洗时跳过已经写过分片的原始文件。

    优先用 progress.json 里的精确清单。**没有清单时只能退化成猜**：
    假设「最后一个分片的 source 就是最后读到的原始文件」，整份跳过。
    这个假设只在原始文件目录没有重新切分过时成立 —— 重新切分过的话
    文件名对不上，会从头重洗一遍（数据重复）。所以对老类别请配
    --clean-only 明确指定要清洗哪些类。
    """
    last = last_consumed_source(existing)
    if last is None:
        return files, 0
    names = [os.path.basename(f) for f in files]
    if last not in names:
        return files, 0
    idx = names.index(last) + 1
    return files[idx:], idx


def list_existing_shards(shard_dir: str) -> list[str]:
    if not os.path.isdir(shard_dir):
        return []
    return [os.path.join(shard_dir, f) for f in sorted(os.listdir(shard_dir))
            if f.endswith(".jsonl")]


def fmt_chars(n: float) -> str:
    """把字符数格式化成人能读的短串（小规模测试时也不会全显示 0.0M）。"""
    n = float(n)
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"


# ---------------------------------------------------------------- 1. 清洗
def clean_category(raw_dir, cat, ccfg, out_dir, is_doc,
                   budget_chars: int | None = None,
                   reuse: bool = False):
    """流式清洗一个类别，分片落盘。返回 (shard_paths, stats, kept_chars)。

    budget_chars: 该类需要的字符预算。达到后立即停止清洗 ——
                  这是节省大量时间的核心（不需要把整个类别都洗完）。
    reuse: 复用已存在的分片（中断后继续时用）。
    """
    files = collect_files(raw_dir, cat)
    shard_dir = os.path.join(out_dir, "cleaned", cat)
    os.makedirs(shard_dir, exist_ok=True)

    existing = list_existing_shards(shard_dir)
    existing_chars, existing_items = ((shard_stats(existing) if existing else (0, 0)))
    done_files: set[str] = read_progress(shard_dir)

    if reuse and existing:
        if budget_chars is None or existing_chars >= budget_chars:
            st = CleanStats()
            st.kept = existing_items
            print(f"      复用已有 {len(existing)} 个分片 / "
                  f"{existing_chars/1e6:.0f}M 字符"
                  f"{'（已达预算）' if budget_chars else ''}")
            return existing, st, existing_chars
        # 续清洗必须先跳过已经处理过的原始文件，否则会重复清洗 → 数据重复
        if done_files:
            remain = [f for f in files if os.path.basename(f) not in done_files]
            skipped = len(files) - len(remain)
        else:
            remain, skipped = resume_files(files, existing)
        if not remain:
            st = CleanStats()
            st.kept = existing_items
            print(f"      原始文件已全部处理完（{len(files)} 个），"
                  f"保留 {existing_items:,} 条 / {existing_chars/1e6:.0f}M 字符")
            return existing, st, existing_chars
        print(f"      已有 {len(existing)} 个分片 / {existing_chars/1e6:.0f}M 字符，"
              f"未达预算 {budget_chars/1e6:.0f}M；"
              f"跳过前 {skipped} 个原始文件，从第 {skipped+1}/{len(files)} 个续清洗")
        files = remain
    else:
        # 全新清洗：清掉旧分片
        for f in os.listdir(shard_dir):
            if f.endswith(".jsonl"):
                os.remove(os.path.join(shard_dir, f))
        existing = []
        existing_chars, existing_items = 0, 0
        done_files = set()

    st = CleanStats()
    deduper = Deduper(ccfg.near_dedup_similarity)
    shards: list[str] = list(existing)
    buf: list[Conversation] = []
    kept_chars: list[int] = [existing_chars]
    stop = [False]

    def flush():
        nonlocal buf
        if not buf:
            return
        path = os.path.join(shard_dir, f"part-{len(shards):04d}.jsonl")
        dump_jsonl(buf, path)
        shards.append(path)
        buf = []
        # 每落一个分片就记一次进度：中途 Ctrl-C / 断电也能精确续上
        write_progress(shard_dir, done_files, kept_chars[0])

    def raw_iter():
        for fp in files:
            yield from load_raw_file(
                fp,
                meta={"category": cat, "source": os.path.basename(fp)},
                document=is_doc,
            )
            done_files.add(os.path.basename(fp))

    for conv in clean_conversations(raw_iter(), ccfg, deduper, st):
        buf.append(conv)
        kept_chars[0] += conv.total_chars
        if len(buf) >= SHARD_SIZE:
            flush()
            # 达到预算就停，不再浪费 CPU 清洗用不到的数据
            if budget_chars is not None and kept_chars[0] >= budget_chars:
                print(f"      已达预算 {budget_chars/1e6:.0f}M 字符，停止清洗"
                      f"（已读入 {st.seen:,} 条）")
                stop[0] = True
                break
        if st.seen % 2_000_000 == 0 and st.seen:
            print(f"      {st.seen:,} 读入 / {st.kept:,} 保留")
    flush()
    write_progress(shard_dir, done_files, kept_chars[0])
    return shards, st, kept_chars[0]


# ---------------------------------------------------------------- 3. tokenizer
def tokenizer_corpus(shard_map, budgets_chars, total_sample_chars):
    """各类按预算比例抽一段数据，用于训练 tokenizer。"""
    total_budget = sum(budgets_chars.values()) or 1
    for cat, shards in shard_map.items():
        share = budgets_chars.get(cat, 0) / total_budget
        want = int(total_sample_chars * share)
        got = 0
        for sh in shards:
            for conv in read_jsonl(sh):
                yield conv
                got += conv.total_chars
                if got >= want:
                    break
            if got >= want:
                break


# ---------------------------------------------------------------- 4. 主流程
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 流式数据构建")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "mix.yaml"))
    p.add_argument("--stage", default="pretrain", choices=["pretrain"])
    p.add_argument("--mix-key", default=None,
                   help="用 configs/mix.yaml 里的哪个段落（默认同 --stage）。"
                        "例如 --stage pretrain --mix-key pretrain_v2")
    p.add_argument("--max-epochs", type=float, default=None,
                   help="单类数据最多重复几轮（默认取配置里的 max_epochs）")
    p.add_argument("--plan-only", action="store_true",
                   help="只清洗 + 打印预算表，不训练 tokenizer、不打包")
    p.add_argument("--clean-only", default=None,
                   help="逗号分隔，只清洗这些类别，其余只复用已有分片。"
                        "老类别续清洗有重复数据风险时用它把范围钉死，"
                        "例如 --clean-only general,depth")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--tokenize", action="store_true")
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--reuse-shards", action="store_true",
                   help="复用已存在的 cleaned 分片（中断后继续）")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--max-chars", type=int, default=None,
                   help="目标总字符预算（覆盖配置）")
    args = p.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    q = cfg.get("quality", {})
    tk = cfg.get("tokenizer", {})
    stage = args.stage
    mix_key = args.mix_key or stage
    stage_cfg = cfg.get(mix_key) or {}
    if not stage_cfg.get("categories"):
        print(f"配置里没有 {mix_key} 段", file=sys.stderr)
        return 1

    weights = normalize_weights(stage_cfg["categories"])
    doc_cats = set(stage_cfg.get("document_categories") or [])
    seq_len = int(stage_cfg.get("packing", {}).get("seq_len", 1024))
    max_epochs = float(
        args.max_epochs if args.max_epochs is not None
        else stage_cfg.get("max_epochs", 1.0)
    )
    val_ratio = float(q.get("val_ratio", 0.01))
    seed = int(cfg.get("seed", 42))

    data_root = args.raw_dir or os.path.join(ROOT, "data")
    raw_dir = os.path.join(data_root, STAGE_DIRS[stage])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("Dreamy 数据构建（流式，低内存）")
    print("=" * 78)
    print(f"原始数据: {raw_dir}")
    print(f"输出    : {out_dir}")

    # ---- 1. 清洗 ----
    print("\n[1/4] 清洗（分片落盘，按预算提前停止）")
    print("-" * 78)

    # 如果给了总字符预算，就能提前算出每类需要多少，
    # 从而在清洗阶段就停手，不必把用不到的数据也处理一遍。
    upfront = None
    if args.max_chars:
        upfront = {k: int(args.max_chars * weights[k]) for k in weights}
        print(f"  总预算 {args.max_chars/1e6:.0f}M 字符"
              f"（≈ {args.max_chars*0.62/1e9:.2f}B token）")
        for k in weights:
            print(f"    {k:<10} {upfront[k]/1e6:>9.0f}M 字符")

    shard_map: dict[str, list[str]] = {}
    stats_by_cat: dict[str, CleanStats] = {}
    avail_items: dict[str, int] = {}
    avail_chars: dict[str, int] = {}

    for cat in weights:
        if not collect_files(raw_dir, cat) and not list_existing_shards(
                os.path.join(out_dir, "cleaned", cat)):
            print(f"  {cat}: 无文件")
            shard_map[cat] = []
            stats_by_cat[cat] = CleanStats()
            avail_items[cat] = 0
            avail_chars[cat] = 0
            continue
        clean_only = (set(x.strip() for x in args.clean_only.split(",") if x.strip())
                      if args.clean_only else None)
        if args.plan_only or (clean_only is not None and cat not in clean_only):
            # 只用现成分片，不洗新数据。
            # 老类别（chat 等）的原始文件目录后来被重新切分过，
            # 靠文件名猜"洗到哪了"会重复，所以干脆不续洗。
            shards = list_existing_shards(os.path.join(out_dir, "cleaned", cat))
            have_c, have_i = (shard_stats(shards) if shards else (0, 0))
            why = "plan-only" if args.plan_only else "不在 --clean-only 里"
            print(f"  {cat}: [{why}] 跳过清洗，"
                  f"复用 {len(shards)} 个分片 / {have_c/1e6:.0f}M 字符")
            shard_map[cat] = shards
            st = CleanStats()
            st.kept = have_i
            stats_by_cat[cat] = st
            avail_items[cat] = have_i
            avail_chars[cat] = have_c
            continue
        ccfg = build_clean_config(q, document=cat in doc_cats,
                                  lenient=cat in LENIENT_CATEGORIES)
        print(f"  {cat}: 清洗中 ...")
        shards, st, kept_chars = clean_category(
            raw_dir, cat, ccfg, out_dir, cat in doc_cats,
            budget_chars=upfront[cat] if upfront else None,
            reuse=args.reuse_shards,
        )
        shard_map[cat] = shards
        stats_by_cat[cat] = st
        avail_items[cat] = st.kept
        avail_chars[cat] = kept_chars
        print(f"      -> 读入 {st.seen:,}，保留 {st.kept:,} 条 / "
              f"{kept_chars/1e6:.0f}M 字符，{len(shards)} 个分片")

    total_avail_chars = sum(avail_chars.values())
    if total_avail_chars == 0:
        print("没有可用数据", file=sys.stderr)
        return 1

    # ---- 2. 预算 ----
    print("\n[2/4] 按字符预算分配")
    print("-" * 78)
    # 预算总量 = 各类可用之和（尽量用满）；某类不足时其他类不会超额
    budget_total = args.max_chars or total_avail_chars
    # 和 v1 的关键区别：预算直接按配比算，**不再 min 到可用量**。
    # 稀缺类别用「重复训练」补齐，而不是让 general 顶上空缺 ——
    # v1 就是这么把 general 从 50% 顶到 60% 的，配比全歪了。
    budgets = {k: float(budget_total) * weights[k] for k in weights}
    eps = max(1.0, float(budget_total) * 1e-4)
    capped: dict[str, float] = {}
    for k in weights:
        avail = float(avail_chars.get(k, 0))
        if avail <= 0:
            capped[k] = 0.0
            print(f"  [warn] {k:<10} 无可用数据，占比 {weights[k]:.1%} 将分摊给其他类")
            continue
        cap = avail * max_epochs
        if budgets[k] > cap:
            capped[k] = cap
            print(f"  [warn] {k:<10} 需 {fmt_chars(budgets[k])} 字符 = "
                  f"{budgets[k]/avail:.1f} 轮 > 上限 {max_epochs:.1f} 轮，"
                  f"截断到 {fmt_chars(cap)}")
        else:
            capped[k] = budgets[k]
    shortfall = float(budget_total) - sum(capped.values())
    if shortfall > eps:
        # 缺口按「还剩多少产能」分摊，通常全部落到 general 上。
        # 必须分轮分摊：一轮就可能把接收方也顶到上限，剩下还要继续找下家。
        for _ in range(10):
            headroom = {
                k: max(0.0, float(avail_chars.get(k, 0)) * max_epochs - capped[k])
                for k in weights
            }
            hsum = sum(headroom.values())
            if shortfall <= eps or hsum <= 0:
                break
            for k in weights:
                give = min(shortfall * headroom[k] / hsum, headroom[k])
                capped[k] += give
            new_shortfall = float(budget_total) - sum(capped.values())
            if new_shortfall >= shortfall - eps:
                break
            shortfall = new_shortfall
        if shortfall > eps:
            print(f"  [warn] 缺口 {fmt_chars(shortfall)} 字符无处分摊，"
                  f"总量会低于目标（需要补数据）", file=sys.stderr)
        else:
            gained = [f"{k}+{fmt_chars(capped[k] - float(budget_total)*weights[k])}"
                      for k in weights
                      if capped[k] > float(budget_total) * weights[k] + eps]
            print("  缺口已按剩余产能分摊给：" + "、".join(gained))
    budgets = capped

    for k in weights:
        avail = float(avail_chars.get(k, 0))
        ep = budgets[k] / avail if avail else 0.0
        print(f"  {k:<10} 可用 {fmt_chars(avail):>9}  预算 {fmt_chars(budgets[k]):>9}"
              f"  重复 {ep:>4.2f} 轮")
    tot = sum(budgets.values())
    print(f"  {'合计':<10} 可用 {fmt_chars(total_avail_chars):>9}  "
          f"预算 {fmt_chars(tot):>9} "
          f"（≈ {tot*0.617/1e9:.2f}B token）")

    if args.plan_only:
        print("\n[plan-only] 只做规划，不训练 tokenizer、不打包。")
        return 0

    # ---- 3. tokenizer ----
    tok_path = os.path.join(out_dir, "tokenizer.model")
    if args.tokenize:
        print("\n[3/4] 训练 tokenizer")
        print("-" * 78)
        if os.path.isfile(tok_path) and not args.retrain_tokenizer:
            print(f"  复用已有 tokenizer: {tok_path}")
        else:
            sample_chars = int(tk.get("max_chars", 150_000_000))
            tok_path = train_tokenizer(
                tokenizer_corpus(shard_map, budgets, sample_chars),
                out_dir,
                vocab_size=tk.get("vocab_size", 32768),
                model_type=tk.get("model_type", "bpe"),
                character_coverage=tk.get("character_coverage", 0.9995),
                max_chars=sample_chars,
            )
    else:
        print("\n[3/4] 跳过 tokenizer")
        tok_path = tok_path if os.path.isfile(tok_path) else None

    # ---- 4. 混合 + 打包 ----
    if not args.tokenize:
        print("\n[4/4] 跳过打包（加 --tokenize 启用）")
        return 0

    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    vocab = sp.get_piece_size()
    ids_of = {n: sp.piece_to_id(n) for n in SPECIAL_TOKENS}
    for n, i in ids_of.items():
        if i == sp.unk_id():
            raise SystemExit(f"特殊 token {n} 不在词表里")

    print("\n[4/4] 混合并打包 token 流")
    print("-" * 78)

    rng = random.Random(seed)
    CHUNK = 2_000_000
    buf: list[int] = []
    cur_target: list[str | None] = [None]
    written = {"train": 0, "val": 0}
    n_chars = 0
    n_conv = 0
    n_doc = 0
    taken_items: dict[str, int] = {}
    taken_chars: dict[str, int] = {}

    ftrain = open(os.path.join(out_dir, "tokens.bin"), "wb")
    fval = open(os.path.join(out_dir, "val_tokens.bin"), "wb")

    def flush(target):
        if not buf:
            return
        fh = ftrain if target == "train" else fval
        np.asarray(buf, dtype=np.uint16).tofile(fh)
        written[target] += len(buf)
        buf.clear()

    def emit(toks: list[int], target: str):
        # 换目标文件前先把旧缓冲刷出去，否则 train/val 会串
        if cur_target[0] is not None and target != cur_target[0] and buf:
            flush(cur_target[0])
        cur_target[0] = target
        buf.extend(toks)
        if len(buf) >= CHUNK:
            flush(target)

    epochs_used: dict[str, float] = {}
    try:
        for cat in weights:
            budget = budgets[cat]
            shards = shard_map[cat]
            avail = float(avail_chars.get(cat, 0))
            if budget <= 0 or not shards:
                epochs_used[cat] = 0.0
                continue
            acc = 0.0
            ep = 0
            # 预算可能超过「可用量 × max_epochs」（分摊后偶发），
            # 所以必须给轮数硬上限，否则会无限循环下去。
            max_rounds = max(1, int(math.ceil(max_epochs)) + 1)
            # 不足预算时循环重复整个类别（每轮重新打乱分片顺序），
            # 直到凑满预算或达到轮数上限。
            while acc < budget and ep < max_rounds:
                ep += 1
                order = list(shards)
                rng.shuffle(order)
                gained = 0.0
                for sh in order:
                    if acc >= budget:
                        break
                    items = list(read_jsonl(sh))   # 一次只读一个分片
                    rng.shuffle(items)
                    for conv in items:
                        if acc >= budget:
                            break
                        is_val = (zlib.crc32(conv.text.encode("utf-8")) % 10000
                                  < val_ratio * 10000)
                        target = "val" if is_val else "train"

                        if conv.is_document:
                            toks = sp.encode(conv.text, out_type=int)
                            toks.append(ids_of[END_TOKEN])
                            n_doc += 1
                        else:
                            toks = []
                            for t in conv.turns:
                                toks.append(ids_of[ROLE_TO_TOKEN.get(t.role, "<|user|>")])
                                toks.extend(sp.encode(t.text, out_type=int))
                            toks.append(ids_of[END_TOKEN])
                            n_conv += 1

                        emit(toks, target)
                        acc += conv.total_chars
                        gained += conv.total_chars
                        n_chars += conv.total_chars
                        taken_items[cat] = taken_items.get(cat, 0) + 1
                        taken_chars[cat] = taken_chars.get(cat, 0) + conv.total_chars
                if gained <= 0:
                    break
            epochs_used[cat] = acc / avail if avail else 0.0
            print(f"  {cat:<10} 写入 {acc/1e6:>9.1f}M 字符 / "
                  f"{ep} 轮（{epochs_used[cat]:.2f} epoch）")
        if cur_target[0] is not None:
            flush(cur_target[0])
    finally:
        ftrain.close()
        fval.close()

    total_tokens = written["train"] + written["val"]
    actual = {k: (taken_chars.get(k, 0) / n_chars if n_chars else 0) for k in weights}

    meta = {
        "vocab_size": vocab,
        "num_tokens": total_tokens,
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "num_conversations": n_conv,
        "num_documents": n_doc,
        "num_chars": n_chars,
        "tokens_per_char": total_tokens / n_chars if n_chars else 0,
        "special_token_ids": ids_of,
        "stage": stage,
        "seq_len": seq_len,
        "mix_key": mix_key,
        "mix_target": weights,
        "mix_actual": actual,
        "max_epochs": max_epochs,
        "epochs_used": epochs_used,
    }
    write_meta(os.path.join(out_dir, "meta.json"), meta)

    # ---- 报告 ----
    if not args.no_report:
        class _M:
            pass
        m = _M()
        m.target = weights
        m.actual = actual
        m.available_chars = avail_chars
        m.taken_chars = taken_chars
        m.total_chars = n_chars
        m.available = avail_items
        m.taken = taken_items
        m.total = sum(taken_items.values())
        m.deviations = lambda: {k: actual[k] - weights[k] for k in weights}
        print()
        print(format_report(m, stats_by_cat, total_tokens,
                            n_params=110_180_000, seq_len=seq_len))

    print(f"  词表      : {vocab:,}")
    print(f"  train token: {written['train']:,}")
    print(f"  val token  : {written['val']:,}")
    print(f"  token/字符 : {meta['tokens_per_char']:.2f}")
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
