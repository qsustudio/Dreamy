"""统计各数据池的真实可用规模，并反推新配比需要重复几轮。

为什么需要这个脚本
------------------
训练时长是固定的（比如 10 小时），能训多少 token 也就固定了；
反过来，要凑够这个 token 数，各类数据够不够就成了硬约束。

「某类不够」有两种处理方式，必须先知道差多少才能决策：
    1. 换数据源（今晚去 HF 上补）
    2. 重复训练（同一份数据多过几遍）—— 4 遍以内基本无损，
       超过 4 遍收益迅速归零（Muennighoff et al. 2023）

所以这里直接算：**按目标配比，每一类需要几轮（epochs）**。
epochs <= 1   → 有富余，可以调高占比
epochs ~= 2-3 → 健康
epochs > 4    → 要么补数据，要么降占比

用法
----
    # 只看现有清洗分片有多大
    python scripts/report_pools.py

    # 按 pretrain_v2 配比、目标 2.4B token 反推每类需要几轮
    python scripts/report_pools.py --mix-key pretrain_v2 --total-tokens 2.4e9

    # 把还没清洗的原始类别也估进来（depth / persona 等）
    python scripts/report_pools.py --include-raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "data_pipeline"))

PROCESSED = os.path.join(ROOT, "data", "processed")
CLEANED = os.path.join(PROCESSED, "cleaned")
RAW = os.path.join(ROOT, "data", "raw")


def count_cleaned(cat_dir: str) -> tuple[int, int]:
    """统计清洗分片里的 (条数, 正文字符数)。

    只解析需要的字段，不构造 Conversation 对象 —— 2000 万条数据下
    这个差别是「几十秒」和「几分钟」。
    """
    docs = 0
    chars = 0
    for name in sorted(os.listdir(cat_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(cat_dir, name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docs += 1
                for t in rec.get("turns") or ():
                    chars += len(t.get("text") or "")
    return docs, chars


def count_raw(cat: str) -> tuple[int, int]:
    """未清洗类别：走清洗流水线的同一个适配器估算原始规模。"""
    from schema import load_raw_file

    raw_dir = os.path.join(RAW, cat)
    if not os.path.isdir(raw_dir):
        return 0, 0
    docs = 0
    chars = 0
    for name in sorted(os.listdir(raw_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        # document 类走文档解析；其余按对话解析（与 configs/mix.yaml 保持一致）
        is_doc = cat in {"general", "knowledge"}
        try:
            for conv in load_raw_file(path, meta={"category": cat}, document=is_doc):
                docs += 1
                chars += conv.total_chars
        except Exception as exc:  # 原始格式千奇百怪，估不准不该让整份报告失败
            print(f"    [warn] {name}: {exc}", file=sys.stderr)
    return docs, chars


def load_weights(mix_key: str) -> dict[str, float] | None:
    path = os.path.join(ROOT, "configs", "mix.yaml")
    if not os.path.isfile(path):
        return None
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sec = cfg.get(mix_key) or {}
    weights = sec.get("categories") or {}
    total = sum(weights.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in weights.items()}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="统计数据池规模并反推重复轮数")
    p.add_argument("--mix-key", default=None,
                   help="configs/mix.yaml 里的段落名，如 pretrain / pretrain_v2")
    p.add_argument("--total-tokens", type=float, default=None,
                   help="目标总 token 数，如 2.4e9")
    p.add_argument("--tokens-per-char", type=float, default=None,
                   help="token/字符，默认从 meta.json 读（约 0.617）")
    p.add_argument("--include-raw", action="store_true",
                   help="把没有清洗分片的类别按原始数据估算")
    p.add_argument("--out", default=os.path.join(PROCESSED, "pool_report.json"))
    args = p.parse_args(argv)

    tpc = args.tokens_per_char
    if tpc is None:
        meta_path = os.path.join(PROCESSED, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                tpc = float(json.load(f).get("tokens_per_char") or 0.617)
        else:
            tpc = 0.617

    cats = []
    if os.path.isdir(CLEANED):
        cats += [d for d in sorted(os.listdir(CLEANED))
                 if os.path.isdir(os.path.join(CLEANED, d))]
    if args.include_raw and os.path.isdir(RAW):
        cats += [d for d in sorted(os.listdir(RAW))
                 if d not in cats and os.path.isdir(os.path.join(RAW, d))]
    if not cats:
        print("没有找到 data/processed/cleaned/，先跑 build_dataset_streaming.py",
              file=sys.stderr)
        return 1

    print("=" * 82)
    print(f"数据池规模报告   token/字符 = {tpc:.4f}")
    print("=" * 82)

    rows = {}
    t0 = time.time()
    for cat in cats:
        cleaned_dir = os.path.join(CLEANED, cat)
        if os.path.isdir(cleaned_dir):
            docs, chars = count_cleaned(cleaned_dir)
            src = "cleaned"
        elif args.include_raw:
            docs, chars = count_raw(cat)
            src = "raw(raw)"
        else:
            continue
        tokens = chars * tpc
        rows[cat] = {"docs": docs, "chars": chars,
                     "tokens": tokens, "source": src}
        print(f"  {cat:<12} {src:<9} {docs:>10,} 条  "
              f"{chars/1e6:>9.1f}M 字符  ≈ {tokens/1e9:>6.3f}B token")

    print(f"\n  统计耗时 {time.time()-t0:.1f}s")

    weights = load_weights(args.mix_key) if args.mix_key else None
    if weights and args.total_tokens:
        total = args.total_tokens
        print("\n" + "-" * 82)
        print(f"按 {args.mix_key} 配比 / 目标 {total/1e9:.2f}B token 反推")
        print("-" * 82)
        print(f"  {'类别':<12}{'目标占比':>9}{'需要':>11}{'可用':>11}"
              f"{'需重复':>9}  判定")
        missing = 0.0
        for cat, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            need = total * w
            avail = rows.get(cat, {}).get("tokens", 0.0)
            if avail <= 0:
                print(f"  {cat:<12}{w:>9.1%}{need/1e9:>10.3f}B"
                      f"{'—':>11}{'—':>9}  ✗ 无数据")
                missing += need
                continue
            epochs = need / avail
            if epochs <= 1.05:
                verdict = "✓ 富余"
            elif epochs <= 2.5:
                verdict = "✓ 健康"
            elif epochs <= 4.0:
                verdict = "△ 偏高"
            else:
                verdict = "✗ 需补数据/降占比"
            print(f"  {cat:<12}{w:>9.1%}{need/1e9:>10.3f}B"
                  f"{avail/1e9:>10.3f}B{epochs:>9.2f}x  {verdict}")
        print(f"\n  合计需要 {total/1e9:.2f}B token；"
              f"无数据缺口 {missing/1e9:.3f}B token")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"tokens_per_char": tpc, "pools": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
