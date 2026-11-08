"""把训练产物推到 Hugging Face Hub —— 别用 GitHub，会卡死。

为什么不用 GitHub
-----------------
GitHub 单文件硬上限 100MB。我们把 checkpoint 分卷成 48MB 的块能绕过去，
但代价是：要分卷、要写 MD5 清单、下载端要拼回来再校验 ——
而上次正是「下载回来的权重是坏的」。三层手工步骤，每一层都可能出错。

Hugging Face 直接支持大文件（LFS），一个命令就能拿到，还能续传。
公开仓库免费，模型文件也不需要 token。

用法
----
    # 0) 装依赖（有 xet / hf_transfer 会快很多）
    pip install -U "huggingface_hub[xet]"

    # 1) 登录（只需要一次，会存到 ~/.cache/huggingface/token）
    hf auth login

    # 2) 把 checkpoint 传到 HF（自动按文件大小选 xet / hf_transfer）
    python scripts/push_hub.py --repo qsustudio/Dreamy --ckpt runs_v2/checkpoints/sft_final.pt

    # 3) 只传某个目录（例如整个 runs_v2/checkpoints）
    python scripts/push_hub.py --repo qsustudio/Dreamy --dir runs_v2/checkpoints

    # 4) 下载端（任意机器）
    hf download qsustudio/Dreamy --local-dir Dreamy-weights

上传策略
--------
    1.5GB 的 .pt 走 xet（HF 的块级去重传输，断点续传，比 LFS 稳很多）
    先用 --dry-run 看一眼会传哪些文件、多大。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HF 仓库里放一份说明，免得别人（和未来的自己）不知道这些文件是什么
README_TEMPLATE = """---
license: apache-2.0
language:
- zh
tags:
- chinese
- chat
- from-scratch
library_name: pytorch
---

# Dreamy

一个从零训练的中文对话模型（不使用任何预训练权重）。

## 架构

| 项目 | 值 |
| --- | --- |
| 参数量 | {n_params} |
| 层数 / 维度 / 头 | {depth} / {dim} / {heads} |
| 上下文 | {seq_len} |
| 词表 | {vocab}（SentencePiece BPE） |
| 位置编码 | RoPE |
| 归一化 | RMSNorm（Pre-Norm） |
| FFN | SwiGLU |
| 注意力 | Flash / SDPA |

## 训练

| 阶段 | 数据量 | 结果 |
| --- | --- | --- |
| 预训练 | 2.35B token（11 类中文语料） | val loss 3.7979 / ppl 44.6 |
| SFT | 33,229 段对话，3 epoch | D/N = 18.5 |

## 用法

```bash
pip install torch x-transformers sentencepiece
python dreamy_chat.py --weights dreamy_weights.pt --data-dir .
```

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `dreamy_weights.pt` | 推理权重（bf16，仅 state_dict） |
| `tokenizer.model` | SentencePiece 词表 |
| `meta.json` | 词表大小 / 上下文长度 |
| `manifest.json` | 各文件的 MD5，用于校验下载是否完整 |
"""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def collect(args) -> list[tuple[str, str]]:
    """返回 [(本地路径, 仓库内路径)]。"""
    items: list[tuple[str, str]] = []
    if args.ckpt:
        p = os.path.abspath(args.ckpt)
        if not os.path.isfile(p):
            raise SystemExit(f"找不到 {p}")
        remote = args.remote_name or os.path.basename(p)
        items.append((p, f"checkpoints/{remote}"))
    if args.dir:
        d = os.path.abspath(args.dir)
        if not os.path.isdir(d):
            raise SystemExit(f"找不到 {d}")
        for name in sorted(os.listdir(d)):
            fp = os.path.join(d, name)
            if os.path.isfile(fp):
                items.append((fp, f"checkpoints/{name}"))
    if args.extra:
        for fp in args.extra.split(","):
            fp = os.path.abspath(fp.strip())
            if os.path.isfile(fp):
                items.append((fp, os.path.basename(fp)))
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="把 checkpoint / 权重推到 Hugging Face Hub")
    p.add_argument("--repo", required=True, help="仓库名，例如 qsustudio/Dreamy")
    p.add_argument("--ckpt", default=None, help="单个 checkpoint 文件")
    p.add_argument("--dir", default=None, help="整个目录")
    p.add_argument("--extra", default=None,
                   help="逗号分隔的额外文件（会放在仓库根目录）")
    p.add_argument("--remote-name", default=None,
                   help="checkpoint 在仓库里的文件名，默认用本地文件名")
    p.add_argument("--path-in-repo", default=None,
                   help="把 --ckpt 放到仓库的哪个子路径（默认 checkpoints/）")
    p.add_argument("--private", action="store_true", help="建私有仓库")
    p.add_argument("--message", default=None, help="commit message")
    p.add_argument("--dry-run", action="store_true", help="只列出要传的文件")
    p.add_argument("--num-workers", type=int, default=8,
                   help="并发上传数（默认 8；网络差可降到 4）")
    args = p.parse_args(argv)

    items = collect(args)
    if not items:
        raise SystemExit("没有要上传的文件，给 --ckpt / --dir / --extra")

    total = sum(os.path.getsize(fp) for fp, _ in items)
    print("=" * 70)
    print(f"目标仓库: {args.repo}  ({'私有' if args.private else '公开'})")
    print(f"文件数  : {len(items)}    合计 {human(total)}")
    print("-" * 70)
    for fp, rp in items:
        print(f"  {human(os.path.getsize(fp)):>9}  {rp}")
    print("=" * 70)

    if args.dry_run:
        print("(dry-run，没有实际上传)")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit('需要 huggingface_hub: pip install -U "huggingface_hub[xet]"')

    api = HfApi()
    try:
        user = api.whoami()
        print(f"登录身份: {user.get('name')}")
    except Exception as e:
        raise SystemExit(
            "没有登录 HF。先跑:\n    hf auth login\n"
            f"（原始报错: {type(e).__name__}: {str(e)[:120]}）") from None

    print(f"\n[1/3] 创建/确认仓库 {args.repo} ...")
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)
    print("  OK")

    print(f"[2/3] 上传 {len(items)} 个文件（并发 {args.num_workers}）...")
    commit_msg = args.message or f"upload {len(items)} file(s), {human(total)}"
    for fp, rp in items:
        if args.path_in_repo and rp.startswith("checkpoints/"):
            rp = args.path_in_repo.rstrip("/") + "/" + os.path.basename(fp)
        print(f"  -> {rp}  ({human(os.path.getsize(fp))})")
        api.upload_file(
            path_or_fileobj=fp,
            path_in_repo=rp,
            repo_id=args.repo,
            repo_type="model",
            commit_message=commit_msg,
        )
    print("  OK")

    print("[3/3] 写 README ...")
    meta = {}
    for cand in (os.path.join(ROOT, "data", "processed", "meta.json"),):
        if os.path.isfile(cand):
            import json
            with open(cand, encoding="utf-8") as f:
                meta = json.load(f)
            break
    readme = README_TEMPLATE.format(
        n_params="127.01M", depth=12, dim=832, heads=13, seq_len=1024,
        vocab=meta.get("vocab_size", 32768),
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="add model card",
    )
    print("  OK")

    print(f"\n完成: https://huggingface.co/{args.repo}")
    print(f"下载端: hf download {args.repo} --local-dir Dreamy-weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
