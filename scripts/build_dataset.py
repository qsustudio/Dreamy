"""一键构建训练数据：清洗 -> 配比混合 -> 切分验证集 -> tokenizer -> 打包

用法::

    # 第一阶段：预训练数据（data/raw/<类别>/ -> tokens.bin）
    python scripts/build_dataset.py --stage pretrain --tokenize

    # 第二阶段：SFT 数据（data/raw_sft/<类别>/ -> sft_tokens.npy）
    python scripts/build_dataset.py --stage sft --tokenize

    # 两个阶段一起
    python scripts/build_dataset.py --stage both --tokenize

    # 只看清洗和配比报告，不打包
    python scripts/build_dataset.py --stage pretrain

目录约定::

    data/
    ├── raw/                      # 预训练原始数据
    │   ├── general/              (中文文章，纯文本)
    │   ├── chat/                 (多轮闲聊)
    │   ├── narrative/            (故事叙事)
    │   ├── knowledge/            (知识解释)
    │   ├── empathy/              (情绪交流)
    │   ├── depth/                (深度讨论)
    │   ├── persona/              (人格一致)
    │   └── safety/               (安全边界)
    ├── raw_sft/                  # SFT 原始数据
    │   ├── chat/ empathy/ depth/
    │   └── opinion/ followup/ boundary/
    └── processed/                # 输出
        ├── tokenizer.model
        ├── tokens.bin / val_tokens.bin        (预训练)
        ├── sft_tokens.npy / sft_labels.npy    (SFT)
        └── meta.json / sft_meta.json
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

# 让 src/ 下的包可被 import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "data_pipeline"))

from clean import CleanConfig, CleanStats, Deduper, clean_conversations  # noqa: E402
from mix import mix_datasets                                            # noqa: E402
from report import format_report                                        # noqa: E402
from schema import Conversation, dump_jsonl, load_raw_file              # noqa: E402
from schema import read_jsonl
from tokenizer import (                                                 # noqa: E402
    pack_sft_arrays, pack_tokens, train_tokenizer, write_meta,
)

# 这些类别不按长度规则砍（短但关键，如危机干预话术）
LENIENT_CATEGORIES = {"safety", "boundary"}

STAGE_DIRS = {"pretrain": "raw", "sft": "raw_sft"}


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


def run_clean(raw_dir: str, out_dir: str, stage_cfg: dict, q: dict,
              reuse: bool = False):
    """清洗各类数据，返回 (pools, stats_by_cat)。

    reuse=True 时直接读 data/processed/cleaned/<类别>.jsonl，
    跳过耗时的清洗（调配比时可以省 20 分钟以上）。
    """
    categories = list(stage_cfg["categories"].keys())
    doc_cats = set(stage_cfg.get("document_categories") or [])

    pools: dict[str, list[Conversation]] = {}
    stats_by_cat: dict[str, CleanStats] = {}

    for cat in categories:
        cleaned_path = os.path.join(out_dir, "cleaned", f"{cat}.jsonl")
        if reuse and os.path.isfile(cleaned_path):
            kept = list(read_jsonl(cleaned_path))
            pools[cat] = kept
            stats_by_cat[cat] = CleanStats()   # 复用时不重算丢弃统计
            print(f"  {cat}: 复用 cleaned/{cat}.jsonl（{len(kept):,} 条）")
            continue

        files = collect_files(raw_dir, cat)
        if not files:
            print(f"  {cat}: 无文件")
            pools[cat] = []
            stats_by_cat[cat] = CleanStats()
            continue

        is_doc = cat in doc_cats
        ccfg = build_clean_config(q, document=is_doc,
                                  lenient=cat in LENIENT_CATEGORIES)
        print(f"  {cat}: {len(files)} 个文件{'（文档类）' if is_doc else ''}")

        def raw_iter(files=files, cat=cat, is_doc=is_doc):
            for fp in files:
                yield from load_raw_file(
                    fp,
                    meta={"category": cat, "source": os.path.basename(fp)},
                    document=is_doc,
                )

        st = CleanStats()
        deduper = Deduper(ccfg.near_dedup_similarity)
        kept = list(clean_conversations(raw_iter(), ccfg, deduper, st))
        pools[cat] = kept
        stats_by_cat[cat] = st
        print(f"      -> 读入 {st.seen:,}，保留 {st.kept:,}")

        if kept:
            dump_jsonl(kept, cleaned_path)

    return pools, stats_by_cat


def split_val(convs: list[Conversation], val_ratio: float, seed: int):
    """按「整条数据」切分验证集。

    绝对不能按 token 窗口随机切 —— 那样同一条对话会一半在 train、
    一半在 val，验证集 loss 会虚低，等于没有验证集。
    """
    if val_ratio <= 0 or len(convs) < 2:
        return convs, []
    idx = list(range(len(convs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(convs) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [c for i, c in enumerate(convs) if i not in val_idx]
    val = [convs[i] for i in sorted(val_idx)]
    return train, val


def get_model_cfg():
    try:
        from config import ModelConfig
        return ModelConfig()
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 数据构建")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "mix.yaml"))
    p.add_argument("--stage", default="pretrain",
                   choices=["pretrain", "sft", "both"])
    p.add_argument("--raw-dir", default=None,
                   help="数据根目录（其下含 raw/ 和 raw_sft/），默认 <项目>/data")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--total", type=int, default=None, help="目标条数，覆盖配置")
    p.add_argument("--strict", action="store_true", help="严格按配比")
    p.add_argument("--tokenize", action="store_true",
                   help="训练 tokenizer 并打包成可训练格式")
    p.add_argument("--retrain-tokenizer", action="store_true",
                   help="强制重训 tokenizer（默认复用已存在的）")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--reuse-cleaned", action="store_true",
                   help="复用 cleaned/<类别>.jsonl，跳过清洗（调配比时省大量时间）")
    p.add_argument("--unit", default="chars", choices=["chars", "items"],
                   help="配比按字符（默认，推荐）还是按条数分配")
    args = p.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    q = cfg.get("quality", {})
    tk = cfg.get("tokenizer", {})
    val_ratio = float(q.get("val_ratio", 0.01))
    seed = int(cfg.get("seed", 42))

    stages = ["pretrain", "sft"] if args.stage == "both" else [args.stage]

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.model")

    print("=" * 78)
    print("Dreamy 数据构建")
    print("=" * 78)
    print(f"阶段        : {', '.join(stages)}")
    print(f"输出目录    : {args.out_dir}")

    for stage in stages:
        stage_cfg = cfg.get(stage)
        if not stage_cfg or not stage_cfg.get("categories"):
            print(f"\n[跳过] 配置里没有 {stage} 段的配比")
            continue

        data_root = args.raw_dir or os.path.join(ROOT, "data")
        raw_dir = os.path.join(data_root, STAGE_DIRS[stage])
        print("\n" + "=" * 78)
        print(f"阶段: {stage}    原始数据: {raw_dir}")
        print("=" * 78)
        if not os.path.isdir(raw_dir):
            print(f"  [!] 目录不存在，跳过。请把数据放到 {raw_dir}/<类别>/")
            continue

        # ---- 1. 清洗 ----
        print("\n[1/4] 清洗")
        print("-" * 78)
        pools, stats_by_cat = run_clean(raw_dir, args.out_dir, stage_cfg, q, reuse=args.reuse_cleaned)

        if sum(len(v) for v in pools.values()) == 0:
            print(f"\n{stage}: 没有可用数据，跳过")
            continue

        # ---- 2. 混合 ----
        print("\n[2/4] 按配比混合")
        print("-" * 78)
        total = args.total if args.total is not None else stage_cfg.get("total")
        mix = mix_datasets(pools, stage_cfg["categories"], total=total,
                           strict=args.strict, seed=seed, unit=args.unit)
        print(f"  取用 {mix.total:,} 条")
        dump_jsonl(mix.conversations,
                   os.path.join(args.out_dir, f"{stage}_mixed.jsonl"))

        # ---- 3. 切分验证集（按整条数据）----
        print("\n[3/4] 切分验证集（按整条对话，避免泄漏）")
        print("-" * 78)
        train_convs, val_convs = split_val(mix.conversations, val_ratio, seed)
        print(f"  train {len(train_convs):,} / val {len(val_convs):,} "
              f"(val_ratio={val_ratio})")

        # ---- 报告 ----
        seq_len = int(stage_cfg.get("packing", {}).get("seq_len", 1024))
        est_tokens = int(sum(c.total_chars for c in mix.conversations) * 1.25)
        mc = get_model_cfg()
        if mc:
            vocab = int(tk.get("vocab_size", 16384))
            # 每层 ≈ 4*dim^2 (attention) + 9*dim^2 (SwiGLU 三矩阵) = 13*dim^2
            n_params = (vocab * mc.dim + mc.max_seq_len * mc.dim
                        + mc.depth * 13 * mc.dim * mc.dim)
        else:
            n_params = 36_000_000
        if not args.no_report:
            print()
            print(format_report(mix, stats_by_cat, est_tokens,
                                n_params=n_params, seq_len=seq_len))

        # ---- 4. tokenizer / 打包 ----
        if not args.tokenize:
            print("\n[4/4] 跳过打包（加 --tokenize 启用）")
            continue

        print("\n[4/4] tokenizer + 打包")
        print("-" * 78)
        if os.path.isfile(tokenizer_path) and not args.retrain_tokenizer:
            print(f"  复用已有 tokenizer: {tokenizer_path}")
        else:
            tokenizer_path = train_tokenizer(
                train_convs, args.out_dir,
                vocab_size=tk.get("vocab_size", 16384),
                model_type=tk.get("model_type", "bpe"),
                character_coverage=tk.get("character_coverage", 0.9995),
                max_chars=tk.get("max_chars", 150_000_000),
            )

        if stage == "pretrain":
            meta = pack_tokens(
                train_convs, tokenizer_path,
                os.path.join(args.out_dir, "tokens.bin"),
            )
            meta.update({
                "stage": "pretrain",
                "seq_len": seq_len,
                "num_samples": len(train_convs),
                "mix_target": mix.target,
                "mix_actual": mix.actual,
            })
            write_meta(os.path.join(args.out_dir, "meta.json"), meta)
            if val_convs:
                pack_tokens(val_convs, tokenizer_path,
                            os.path.join(args.out_dir, "val_tokens.bin"))
            print(f"  词表        : {meta['vocab_size']:,}")
            print(f"  train token : {meta['num_tokens']:,}")
            print(f"  token/字符  : {meta['tokens_per_char']:.2f}")
            print(f"  验证集      : {os.path.join(args.out_dir, 'val_tokens.bin')}")
            print("  输出        : tokens.bin / val_tokens.bin / meta.json")

        else:  # sft
            meta = pack_sft_arrays(
                train_convs, tokenizer_path, seq_len=seq_len,
                out_tokens_path=os.path.join(args.out_dir, "sft_tokens.npy"),
                out_labels_path=os.path.join(args.out_dir, "sft_labels.npy"),
            )
            if val_convs:
                pack_sft_arrays(
                    val_convs, tokenizer_path, seq_len=seq_len,
                    out_tokens_path=os.path.join(args.out_dir, "sft_val_tokens.npy"),
                    out_labels_path=os.path.join(args.out_dir, "sft_val_labels.npy"),
                )
            meta.update({
                "stage": "sft",
                "seq_len": seq_len,
                "mix_target": mix.target,
                "mix_actual": mix.actual,
            })
            write_meta(os.path.join(args.out_dir, "sft_meta.json"), meta)
            print(f"  样本数      : {meta['sft_samples']:,} x {meta['sft_seq_len']}")
            print(f"  监督 token  : {meta['num_supervised_tokens']:,} "
                  f"({meta['supervised_ratio']:.1%}，只有 assistant 部分)")
            print("  输出        : sft_tokens.npy / sft_labels.npy / sft_meta.json")

    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
