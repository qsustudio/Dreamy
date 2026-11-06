"""一键构建训练数据：清洗 -> 配比混合 -> (可选) tokenizer -> tokens.bin

用法::

    # 只清洗 + 混合 + 报告（不建 tokenizer）
    python scripts/build_dataset.py

    # 全流程，产出可直接训练的 tokens.bin
    python scripts/build_dataset.py --tokenize

    # 指定数据量 / 严格配比
    python scripts/build_dataset.py --total 200000 --strict --tokenize

目录约定::

    data/
    ├── raw/                     <- 你自己往里放原始数据
    │   ├── general/             (通用中文语料)
    │   ├── chat/                (开放域闲聊)
    │   ├── empathy/             (情感支持)
    │   ├── depth/               (深度内容)
    │   ├── persona/             (人设/角色)
    │   └── safety/              (安全边界)
    └── processed/               <- 脚本输出
        ├── cleaned/<cat>.jsonl
        ├── mixed.jsonl
        ├── tokenizer.model
        ├── tokens.bin
        └── meta.json
"""
from __future__ import annotations

import argparse
import os
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
from tokenizer import pack_tokens, train_tokenizer, write_meta          # noqa: E402

DEFAULT_CATEGORIES = ["general", "chat", "empathy", "depth", "persona", "safety"]

# 这些类别不按长度规则砍 —— 短但关键（危机干预话术往往就一两句）
LENIENT_CATEGORIES = {"safety"}


def load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_files(raw_dir: str, category: str) -> list[str]:
    d = os.path.join(raw_dir, category)
    if not os.path.isdir(d):
        return []
    files = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and not name.startswith("."):
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 数据构建")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "mix.yaml"))
    p.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--total", type=int, default=None, help="目标总条数，覆盖配置")
    p.add_argument("--strict", action="store_true",
                   help="严格按配比（总条数受最稀缺类别限制）")
    p.add_argument("--tokenize", action="store_true",
                   help="训练 tokenizer 并打包 tokens.bin")
    p.add_argument("--no-report", action="store_true", help="不打印报告")
    args = p.parse_args(argv)

    t_start = time.time()
    cfg = load_config(args.config)
    weights: dict[str, float] = cfg.get("categories", {})
    q = cfg.get("quality", {})
    tk = cfg.get("tokenizer", {})
    pk = cfg.get("packing", {})

    if not weights:
        print("配置里没有 categories 配比", file=sys.stderr)
        return 1

    categories = list(weights.keys())
    doc_categories = set(cfg.get("document_categories", []) or [])
    print("=" * 78)
    print("Dreamy 数据构建")
    print("=" * 78)
    print(f"原始数据目录: {args.raw_dir}")
    print(f"输出目录    : {args.out_dir}")
    print(f"类别        : {', '.join(categories)}")
    if doc_categories:
        print(f"文档类（纯文本）: {', '.join(sorted(doc_categories))}")

    # ---------------- 1. 清洗 ----------------
    print("\n[1/4] 清洗")
    print("-" * 78)
    pools: dict[str, list[Conversation]] = {}
    stats_by_cat: dict[str, CleanStats] = {}

    for cat in categories:
        files = collect_files(args.raw_dir, cat)
        if not files:
            print(f"  {cat}: 无文件（{os.path.join(args.raw_dir, cat)}）")
            pools[cat] = []
            stats_by_cat[cat] = CleanStats()
            continue

        print(f"  {cat}: {len(files)} 个文件")
        is_doc = cat in doc_categories
        lenient = cat in LENIENT_CATEGORIES
        ccfg = CleanConfig(
            min_turns=q.get("min_turns", 2),
            max_turns=q.get("max_turns", 30),
            max_chars=q.get("max_chars", 3000),
            min_chars_per_turn=q.get("min_chars_per_turn", 1),
            max_repeat_ngram_ratio=q.get("max_repeat_ngram_ratio", 0.15),
            repeat_ngram_n=q.get("repeat_ngram_n", 5),
            max_punct_ratio=q.get("max_punct_ratio", 0.5),
            min_unique_char_ratio=q.get("min_unique_char_ratio", 0.05),
            exact_dedup=q.get("exact_dedup", True),
            near_dedup=q.get("near_dedup", True),
            near_dedup_similarity=q.get("near_dedup_similarity", 0.55),
            redact_pii=q.get("redact_pii", True),
            lenient=lenient,
            document=is_doc,
            min_chars_document=q.get("min_chars_document", 50),
            max_chars_document=q.get("max_chars_document", 4000),
        )

        def raw_iter(files=files, cat=cat, is_doc=is_doc):
            for fp in files:
                yield from load_raw_file(
                    fp,
                    meta={"category": cat, "source": os.path.basename(fp)},
                    document=is_doc,
                )

        st = CleanStats()
        deduper = Deduper(ccfg.near_dedup_similarity)
        kept: list[Conversation] = []
        for conv in clean_conversations(raw_iter(), ccfg, deduper=deduper, stats=st):
            kept.append(conv)
        pools[cat] = kept
        stats_by_cat[cat] = st
        print(f"      -> 读入 {st.seen:,}，保留 {st.kept:,}")

        # 落盘清洗结果，方便抽查
        if kept:
            dump_jsonl(kept, os.path.join(args.out_dir, "cleaned", f"{cat}.jsonl"))

    if sum(len(v) for v in pools.values()) == 0:
        print("\n没有任何可用数据。请把原始数据按类别放进 "
              f"{args.raw_dir}/<类别>/ 后重试。", file=sys.stderr)
        print("\n支持的格式: .jsonl / .json / .txt（「用户:」「助手:」交替）",
              file=sys.stderr)
        return 1

    # ---------------- 2. 混合 ----------------
    print("\n[2/4] 按配比混合")
    print("-" * 78)
    total = args.total if args.total is not None else cfg.get("total")
    mix = mix_datasets(pools, weights, total=total, strict=args.strict,
                       seed=cfg.get("seed", 42))
    print(f"  取用 {mix.total:,} 条")
    dump_jsonl(mix.conversations, os.path.join(args.out_dir, "mixed.jsonl"))
    print(f"  已写入 {os.path.join(args.out_dir, 'mixed.jsonl')}")

    # ---------------- 3. 报告 ----------------
    seq_len = int(pk.get("seq_len", 512))
    est_tokens = sum(c.total_chars for c in mix.conversations) * 1.6  # 中文约 1.2~1.8 token/字

    # 参数量估算：embedding + depth * 每层(约 12*dim^2)
    try:
        from config import ModelConfig
        mc = ModelConfig()
        vocab = int(tk.get("vocab_size", 16000))
        n_params = (vocab * mc.dim + mc.max_seq_len * mc.dim
                    + mc.depth * 12 * mc.dim * mc.dim)
    except Exception:
        n_params = 6_500_000

    if not args.no_report:
        print()
        print(format_report(mix, stats_by_cat, int(est_tokens),
                            n_params=n_params, seq_len=seq_len))

    # ---------------- 4. tokenizer ----------------
    if args.tokenize:
        print("\n[3/4] 训练 tokenizer")
        print("-" * 78)
        try:
            model_path = train_tokenizer(
                mix.conversations,
                args.out_dir,
                vocab_size=tk.get("vocab_size", 16000),
                model_type=tk.get("model_type", "bpe"),
                character_coverage=tk.get("character_coverage", 0.9995),
            )
        except ImportError:
            print("缺少 sentencepiece：pip install sentencepiece", file=sys.stderr)
            return 1

        print("\n[4/4] 打包 token 流")
        print("-" * 78)

        meta = pack_tokens(
            mix.conversations, model_path,
            os.path.join(args.out_dir, "tokens.bin"),
        )
        meta["seq_len"] = seq_len
        meta.update({
            "mix_target": mix.target,
            "mix_actual": mix.actual,
            "min_turns": q.get("min_turns", 2),
            "max_turns": q.get("max_turns", 30),
        })
        write_meta(os.path.join(args.out_dir, "meta.json"), meta)

        print(f"  词表大小   : {meta['vocab_size']:,}")
        print(f"  token 总数 : {meta['num_tokens']:,}")
        print(f"  token/字符 : {meta['tokens_per_char']:.2f}")
        print(f"  输出       : {os.path.join(args.out_dir, 'tokens.bin')}")
        print(f"  meta       : {os.path.join(args.out_dir, 'meta.json')}")
    else:
        print("\n[3/4] 跳过 tokenizer（加 --tokenize 启用）")
        print("[4/4] 跳过打包")

    print(f"\n完成，用时 {time.time() - t_start:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
