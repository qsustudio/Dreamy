"""下载 Dreamy 训练数据（HuggingFace）。

优先级：**已清洗过的公开数据集**，全部 gated=False，不需要申请权限。

用法::

    # 验证子集（约 3GB，够跑通管线 + 精确测 token 比）
    python scripts/download_data.py --preset small

    # 完整（约 8GB 原料，清洗后够 2.0B token 训练）
    python scripts/download_data.py --preset full

    # 国内网络走镜像
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_data.py --preset full

下载后目录::

    data/raw_src/
    ├── skypile/          通用中文网页（已质量过滤）
    ├── wikipedia/        中文维基
    ├── lccc/             中文闲聊对话
    └── instruct/         指令/对话（含心理、共情类）

下一步用 scripts/prepare_sources.py 把它们转成 data/raw/<类别>/。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "raw_src")

# HF 默认缓存到 ~/.cache/huggingface，在受限环境下会因权限失败。
# 统一指到项目内，也方便整体打包搬到训练机上。
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(ROOT, ".cache", "xet"))
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ---- 各数据集的下载规则 ----
# repo: HF 数据集 id
# small/full: allow_patterns 列表（glob）
SOURCES = {
    "skypile": {
        "repo": "opencsg/chinese-fineweb-edu",
        "desc": "通用中文网页（SkyPile 子集，已质量过滤）",
        "small": ["Skypile/00000.parquet"],
        "full": [f"Skypile/{i:05d}.parquet" for i in range(8)],
    },
    "wikipedia": {
        "repo": "wanng/wikipedia-zh-mnbvc",
        "desc": "中文维基百科",
        "small": ["data/0.jsonl"],
        "full": ["data/0.jsonl", "data/1.jsonl", "data/2.jsonl"],
    },
    "lccc": {
        "repo": "silver/lccc",
        "desc": "中文闲聊对话（LCCC）",
        "small": ["lccc_base_train.jsonl.gz"],
        "full": ["lccc_base_train.jsonl.gz", "lccc_large.jsonl.gz"],
    },
    "instruct": {
        "repo": "Mxode/Chinese-Instruct",
        "desc": "中文指令/对话（含心理共情类）",
        "small": ["coig-cqia/train.jsonl", "psycho-10k-dpsk-r1/train.jsonl",
                  "neo_sft_phase2/train.jsonl"],
        "full": ["coig-cqia/train.jsonl", "psycho-10k-dpsk-r1/train.jsonl",
                 "neo_sft_phase2/train.jsonl", "infinity-instruct/train.jsonl",
                 "firefly/train.jsonl"],
    },
}


def human(n):
    for u, s in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= s:
            return f"{n/s:.1f}{u}"
    return f"{n}B"


def dir_size(path):
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="下载 Dreamy 训练数据")
    p.add_argument("--preset", default="small", choices=["small", "full"])
    p.add_argument("--dest", default=DEST)
    p.add_argument("--only", default=None,
                   help="只下某几个源，逗号分隔: skypile,wikipedia,lccc,instruct")
    p.add_argument("--list", action="store_true", help="只列出计划，不下载")
    args = p.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub：\n  pip install huggingface_hub hf_transfer",
              file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print("=" * 74)
    print("Dreamy 数据下载")
    print("=" * 74)
    print(f"端点   : {endpoint}")
    print(f"预设   : {args.preset}")
    print(f"目标   : {args.dest}")

    names = list(SOURCES)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() in SOURCES]

    plan = []
    for name in names:
        s = SOURCES[name]
        plan.append((name, s["repo"], s[args.preset]))
        print(f"\n  {name:<10} {s['desc']}")
        print(f"    repo    : {s['repo']}")
        print(f"    文件数  : {len(s[args.preset])}")

    if args.list:
        for name, repo, pats in plan:
            for pth in pats:
                print(f"  {repo} :: {pth}")
        return 0

    t0 = time.time()
    failed = []
    for name, repo, pats in plan:
        target = os.path.join(args.dest, name)
        print(f"\n{'='*74}\n>>> {name}  ({repo})\n{'='*74}")
        before = dir_size(target)
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=pats,
                local_dir=target,
                max_workers=8,
            )
        except Exception as e:
            print(f"  [失败] {type(e).__name__}: {str(e)[:200]}")
            failed.append((name, str(e)[:120]))
            continue
        after = dir_size(target)
        print(f"  完成: {human(after)} (新增 {human(after-before)})")

    total = dir_size(args.dest)
    print(f"\n{'='*74}")
    print(f"总计下载: {human(total)}   用时 {(time.time()-t0)/60:.1f} 分钟")
    if failed:
        print("\n以下数据源失败，可以重跑（断点续传）:")
        for n, e in failed:
            print(f"  {n}: {e}")
        print("\n国内网络可尝试镜像：")
        print("  HF_ENDPOINT=https://hf-mirror.com python scripts/download_data.py "
              f"--preset {args.preset}")
        return 1
    print("全部完成。下一步：python scripts/prepare_sources.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
