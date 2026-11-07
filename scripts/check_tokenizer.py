"""检查「新数据 + 现有 tokenizer」是否兼容。

从零训练的模型，词表是固定的：如果新数据里出现词表外的字符，
SentencePiece 会吐 `<unk>`（id=1），那部分内容等于直接丢掉，
而且会把模型教成「在句子里乱吐 unk」。

这个脚本就是上机前的安检：**任何一类的 unk 率 > 0.5% 就不要拿去训**。

真实案例（2026-11-08）：
    中文维基 dump 大量条目是繁体，而词表是在简体语料上训的 ->
    knowledge_v2 的 unk 率 12.2%，5 亿 token 全是噪音。
    做完繁转简后降到 0.26%，同时压缩率从 0.669 改善到 0.635 token/字符。

用法::

    python scripts/check_tokenizer.py
    python scripts/check_tokenizer.py --tok data/processed/tokenizer.model --limit 5000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认要检查的目录（<类别> -> 匹配模式）
DEFAULT_TARGETS = {
    "general":      "data/raw/general/*.jsonl",
    "chat":         "data/raw/chat/*.jsonl",
    "knowledge_v2": "data/raw/knowledge_v2/*.jsonl",
    "depth_v2":     "data/raw/depth_v2/*.jsonl",
    "chat_v2":      "data/raw/chat_v2/*.jsonl",
    "empathy_v2":   "data/raw/empathy_v2/*.jsonl",
    "safety_v2":    "data/raw/safety_v2/*.jsonl",
}

WARN_UNK = 0.005      # 0.5%：超过这个比例就该处理数据了


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="检查新数据与现有 tokenizer 的兼容性")
    p.add_argument("--tok", default=os.path.join(
        ROOT, "data", "processed", "tokenizer.model"))
    p.add_argument("--limit", type=int, default=4000, help="每类抽多少条")
    p.add_argument("--targets", default=None,
                   help="逗号分隔的 名称=glob 覆盖默认列表")
    args = p.parse_args(argv)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(args.tok)
    unk = sp.unk_id()
    print(f"tokenizer: {args.tok}  vocab={sp.get_piece_size()}  unk_id={unk}")

    if args.targets:
        targets = {}
        for item in args.targets.split(","):
            name, _, pat = item.partition("=")
            targets[name.strip()] = pat.strip()
    else:
        targets = DEFAULT_TARGETS

    print(f"\n{'类别':<16}{'样本':>7}{'字符':>13}{'token':>13}"
          f"{'tok/char':>10}{'unk数':>10}{'unk%':>8}  判定")
    bad = []
    for name, pat in targets.items():
        files = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not files:
            print(f"{name:<16}  (无文件: {pat})")
            continue
        n = chars = toks = n_unk = 0
        for path in files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if n >= args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(d, dict):
                        continue
                    text = d.get("text")
                    if not isinstance(text, str):
                        text = "\n".join(t.get("text", "")
                                         for t in d.get("turns") or [])
                    if not text.strip():
                        continue
                    ids = sp.encode(text, out_type=int)
                    n += 1
                    chars += len(text)
                    toks += len(ids)
                    n_unk += sum(1 for i in ids if i == unk)
            if n >= args.limit:
                break
        if not n:
            print(f"{name:<16}  (没有可用样本)")
            continue
        rate = n_unk / max(1, toks)
        flag = "OK" if rate <= WARN_UNK else "!! 需要处理"
        if rate > WARN_UNK:
            bad.append(name)
        print(f"{name:<16}{n:>7}{chars:>13,}{toks:>13,}"
              f"{toks/chars:>10.4f}{n_unk:>10,}{rate*100:>7.3f}%  {flag}")

    if bad:
        print(f"\n[x] 这些类别 unk 率过高: {', '.join(bad)}")
        print("    常见原因：繁体字 / 生僻字 / 其它语言。修法：")
        print("    - 繁转简（scripts/prepare_sources_v2.py 已内置，表在 scripts/data/t2s_table.txt）")
        print("    - 或在清洗阶段按 unk 率过滤掉这些条目")
        return 1
    print("\n全部通过，可以直接拿去训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
