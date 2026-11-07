"""下载 Dreamy v2 补充数据（HF，只下核实过 gated=False 的数据集）。

与 scripts/download_data.py 的区别:
    1. 只落 data/raw_src2/<name>/，不覆盖 v1 的 data/raw_src/；
    2. 每条都记录**核实过的精确字节数**，--list 就能估算今晚要下多久；
    3. 端点自动探测：huggingface.co 不通就切 hf-mirror.com（国内网络）；
    4. 断点续传 + 每个 repo 自动重试，失败不中断其它源；
    5. 下完写 data/raw_src2/_manifest.json，明天 prepare_sources.py 扩展时
       按 manifest 找目录和子集，不用再猜文件名。

用法::

    # 先看计划（含核实大小、按 0.2/1/5 MB/s 估算的耗时）
    python scripts/download_v2.py --list

    # 今晚核心清单（9 个源，约 4.98GB -> 约 1.0B clean token）
    python scripts/download_v2.py --preset core

    # 只下其中几个
    python scripts/download_v2.py --only zhihu_kol,sharegpt90k_zh

    # 带宽富裕再加（Belle / sharegpt_gpt4 / cvalues-zh 等）
    python scripts/download_v2.py --preset optional

    # 强制某个端点（默认 auto：先探测 huggingface.co，8s 不通自动切镜像）
    python scripts/download_v2.py --preset core --endpoint mirror

下载后目录::

    data/raw_src2/
    ├── zhihu_kol/            wangrui6/Zhihu-KOL                    -> depth / opinion
    ├── sharegpt90k_zh/       shareAI/ShareGPT-Chinese-English-90k  -> chat
    ├── counseling_cpsdd/     XuShihao6715/counseling-cpsdd          -> empathy
    ├── counseling_psycheval/ XuShihao6715/counseling-psycheval      -> empathy
    ├── counseling_soulchat_r1/ XuShihao6715/counseling-soulchat-r1  -> empathy
    ├── counseling_sage/      XuShihao6715/counseling-sage           -> empathy
    ├── wikipedia_zh/         wikimedia/wikipedia (20231101.zh)       -> knowledge
    ├── coig_cqia/            m-a-p/COIG-CQIA 子集                    -> knowledge/depth/safety
    └── cvalues_rlhf/         Skepsun/cvalues_rlhf                    -> safety

明天接着做：scripts/prepare_sources.py 加几种 adapter（见 docs/v2_data_sources.md §3）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "raw_src2")

# HF 缓存放到项目内，受限环境下 HOME 缓存会因权限失败；也方便整体搬去训练机。
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(ROOT, ".cache", "xet"))
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ---------------------------------------------------------------- 数据源
# 所有 repo 都在 2026-11-07 通过 hf-mirror.com/api 核实:
#   gated=False / 文件清单 / size_bytes（/tree API 精确求和）。
# category 是给 prepare_sources 用的目标类别提示，不是最终配比。
# tier: core = 今晚必下；optional = 带宽富裕再下。
SOURCES = {
    # ======================= core：真实多轮对话 =======================
    "sharegpt90k_zh": {
        "repo": "shareAI/ShareGPT-Chinese-English-90k",
        "desc": "ShareGPT 中英 90k 的中文切片（真实用户提问+多轮）",
        "category": "chat",
        "tier": "core",
        # 只下中文，不下 common_en_70k / computer_en_26k
        "patterns": [
            "sharegpt_jsonl/common_zh_70k.jsonl",
            "sharegpt_jsonl/unknow_zh_38k.jsonl",
            "sharegpt_jsonl/unknow_zh_38k_continue.jsonl",
            "sharegpt_jsonl/computer_zh_26k.jsonl",
        ],
        "size_bytes": 733_329_945,
        "adapter": "需新增 _from_human_assistant_pairs（conversation:[{human,assistant}]）",
    },
    "counseling_cpsdd": {
        "repo": "XuShihao6715/counseling-cpsdd",
        "desc": "CPsDD 真实中文心理咨询多轮（54,508 段，平均 17.1 轮）",
        "category": "empathy",
        "tier": "core",
        # 只下 dialogue 级文件（*.turns.jsonl 是逐轮展开、与 dialogue 重复）
        "patterns": ["train.jsonl", "dev.jsonl", "test.jsonl"],
        "size_bytes": 365_313_094,
        "adapter": "role 别名加 client->user / counselor->assistant",
    },
    "counseling_psycheval": {
        "repo": "XuShihao6715/counseling-psycheval",
        "desc": "心理剧/咨询对话评估集（dialogue 级）",
        "category": "empathy",
        "tier": "core",
        "patterns": ["all.jsonl"],
        "size_bytes": 161_805_076,
        "adapter": "同上（同一套 messages/counselor/client schema）",
    },
    "counseling_soulchat_r1": {
        "repo": "XuShihao6715/counseling-soulchat-r1",
        "desc": "SoulChat 系中文咨询对话（R1 改写版）",
        "category": "empathy",
        "tier": "core",
        "patterns": ["all.jsonl"],
        "size_bytes": 103_892_249,
        "adapter": "同上（跨仓可能有重复，靠近似去重）",
    },
    "counseling_sage": {
        "repo": "XuShihao6715/counseling-sage",
        "desc": "Sage 系中文咨询对话",
        "category": "empathy",
        "tier": "core",
        "patterns": ["all.jsonl"],
        "size_bytes": 107_979_440,
        "adapter": "同上",
    },
    # ======================= core：knowledge / depth =======================
    "zhihu_kol": {
        "repo": "wangrui6/Zhihu-KOL",
        "desc": "知乎高赞问答 100.6 万条长回答（depth/opinion/knowledge）",
        "category": "depth",
        "tier": "core",
        "patterns": ["data/*.parquet"],
        "size_bytes": 1_501_204_472,
        "adapter": "大写键 INSTRUCTION/RESPONSE，需 lower() 或加别名",
    },
    "wikipedia_zh": {
        "repo": "wikimedia/wikipedia",
        "desc": "中文维基百科 20231101（官方数据集，text/paragraphs）",
        "category": "knowledge",
        "tier": "core",
        "patterns": ["20231101.zh/*.parquet"],
        "size_bytes": 1_721_150_260,
        "adapter": "文档模式，列名 text 可直接复用 iter_skypile",
    },
    "coig_cqia": {
        "repo": "m-a-p/COIG-CQIA",
        "desc": "COIG-CQIA 人工写作子集（zhihu/wiki/douban/xhs/wikihow/100poison）",
        "category": "mixed",
        "tier": "core",
        # 不下 COIG-CQIA-full.jsonl（149MB，与子集重复）
        "patterns": [
            "zhihu/*", "wiki/*", "human_value/*", "douban/*", "xhs/*",
            "wikihow/*", "finance/*", "segmentfault/*", "exam/*",
            "logi_qa/*", "ruozhiba/*", "chinese_traditional/*", "coig_pc/*",
        ],
        "size_bytes": 149_490_936,
        "adapter": "多数 instruction/output 原生支持；逐子集明天 --limit 100 抽验",
    },
    # ======================= core：safety =======================
    "cvalues_rlhf": {
        "repo": "Skepsun/cvalues_rlhf",
        "desc": "CValues 中文安全偏好对（prompt/pos_resp/neg_resp）",
        "category": "safety",
        "tier": "core",
        "patterns": ["harmless_train.json", "helpful_train.json"],
        "size_bytes": 134_217_960,
        "adapter": "需新增 _from_pos_neg（user=prompt, assistant=pos_resp）",
    },
    # ======================= optional（带宽富裕再下）=======================
    "belle_multiturn": {
        "repo": "BelleGroup/multiturn_chat_0.8M",
        "desc": "BELLE 0.8M 多轮对话（ChatGPT 生成，GPL-3.0）",
        "category": "chat",
        "tier": "optional",
        "patterns": ["multiturn_chat_0.8M.json"],
        "size_bytes": 990_032_546,
        "adapter": "_from_pair 可认；建议按 Human:/Assistant: 切真轮",
    },
    "sharegpt_gpt4_zh": {
        "repo": "shibing624/sharegpt_gpt4",
        "desc": "GPT-4 中文多轮 ShareGPT（与 sharegpt90k 有重叠）",
        "category": "chat",
        "tier": "optional",
        "patterns": [
            "sharegpt_zh_38K_format.jsonl",
            "sharegpt_gpt4.jsonl",
            "sharegpt_V3_format.jsonl",
        ],
        "size_bytes": 602_315_943,
        "adapter": "conversations:[{from,value}] 则原生支持，明天抽验",
    },
    "cvalues_zh": {
        "repo": "david9dragon9/cvalues-chinese",
        "desc": "CValues 中文全量偏好数据（safety 补充）",
        "category": "safety",
        "tier": "optional",
        "patterns": ["chinese_train.json"],
        "size_bytes": 477_580_000,
        "adapter": "同 cvalues_rlhf（字段待抽验）",
    },
    # 下面两个体积大/尚未抽样验证 schema，默认不下载；确认后再 --only 单独下。
    "industry_lit_emotion": {
        "repo": "BAAI/IndustryCorpus2_literature_emotion",
        "desc": "BAAI 行业语料·文学/情感域（每文件约 0.9GB，按文件下）",
        "category": "depth",
        "tier": "optional",
        "patterns": ["chinese/high/rank_00063.parquet"],
        "size_bytes": 948_750_000,
        "adapter": "文档模式；列名待抽验，先 --only 下 1 个文件验证",
        "verify_first": True,
    },
    "psychology_books": {
        "repo": "Mxode/Chinese-Psychology-Books",
        "desc": "中文心理学书籍（22 个 zip，非商用 license，整库 10.6GB）",
        "category": "depth",
        "tier": "optional",
        "patterns": ["*.zip"],
        "size_bytes": 10_630_000_000,
        "adapter": "解压后按文档读；体积大，除非训练机带宽 >5MB/s 否则别下",
        "verify_first": True,
    },
}

# 已实测：本机 -> hf-mirror 约 0.2 MB/s；训练机/直连一般 1~10 MB/s。
SPEEDS_MBPS = (0.2, 1.0, 5.0)


def human(n: float) -> str:
    for u, s in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= s:
            return f"{n/s:.2f}{u}"
    return f"{n:.0f}B"


def dir_size(path: str) -> int:
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def probe(url: str, timeout: float = 8.0) -> bool:
    """探测端点是否可用（只请求 dataset API 的极小响应）。"""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "dreamy-download/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def resolve_endpoint(mode: str) -> str:
    """auto: 先试官方，8s 不通就切镜像。返回实际使用的 endpoint。"""
    if os.environ.get("HF_ENDPOINT"):
        ep = os.environ["HF_ENDPOINT"].rstrip("/")
        print(f"端点   : {ep}（来自环境变量 HF_ENDPOINT）")
        return ep

    hf = "https://huggingface.co"
    mirror = "https://hf-mirror.com"
    if mode == "hf":
        ep = hf
    elif mode == "mirror":
        ep = mirror
    else:
        print("端点   : 探测 huggingface.co ...", end="", flush=True)
        if probe(hf + "/api/datasets/silver/lccc"):
            print(" 可用")
            ep = hf
        else:
            print(" 不通，切换 hf-mirror.com")
            ep = mirror
    os.environ["HF_ENDPOINT"] = ep
    print(f"端点   : {ep}")
    return ep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="下载 Dreamy v2 补充数据")
    p.add_argument("--preset", default="core",
                   choices=["core", "optional", "all"])
    p.add_argument("--only", default=None,
                   help="只下某几个源，逗号分隔；名字见 --list")
    p.add_argument("--dest", default=DEST)
    p.add_argument("--endpoint", default="auto",
                   choices=["auto", "hf", "mirror"],
                   help="auto=先探测官方，不通自动用 hf-mirror")
    p.add_argument("--list", action="store_true", help="只打印计划，不下载")
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args(argv)

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in SOURCES]
        if unknown:
            print(f"未知数据源: {unknown}；可用: {list(SOURCES)}", file=sys.stderr)
            return 1
    elif args.preset == "all":
        names = list(SOURCES)
    else:
        names = [n for n, s in SOURCES.items() if s["tier"] == args.preset]
        # optional 预设把 core 一起带上，免得用户以为 --preset optional 只下零头
        if args.preset == "optional":
            names = [n for n, s in SOURCES.items() if s["tier"] in ("core", "optional")]

    plan = [(n, SOURCES[n]) for n in names]
    total_bytes = sum(s["size_bytes"] for _, s in plan)

    print("=" * 78)
    print("Dreamy v2 数据下载计划")
    print("=" * 78)
    print(f"目标   : {args.dest}")
    print(f"预设   : {args.preset}   数据源: {len(plan)} 个")
    print(f"核实大小: {human(total_bytes)}（来自 HF tree API 精确求和）")
    est = " / ".join(f"{total_bytes/1e6/s/3600:.1f}h" for s in SPEEDS_MBPS)
    print(f"预估耗时: {est}（按 {' / '.join(str(s) for s in SPEEDS_MBPS)} MB/s；"
          f"本机镜像实测 ~0.2MB/s）")
    print("-" * 78)
    for name, s in plan:
        flag = " [先验证]" if s.get("verify_first") else ""
        print(f"  {name:<24} {human(s['size_bytes']):>9}  {s['repo']}{flag}")
        print(f"  {'':<24} -> {s['category']:<10} {s['desc']}")
        print(f"  {'':<24}    adapter: {s['adapter']}")
    if args.list:
        return 0

    # 依赖：huggingface_hub 已在 v1 的 requirements 里
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n缺少 huggingface_hub：\n  pip install huggingface_hub hf_transfer",
              file=sys.stderr)
        return 1
    # hf_transfer 能显著提速（有就用，没有也能跑）
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("加速   : hf_transfer 已启用")
        except ImportError:
            pass

    resolve_endpoint(args.endpoint)
    os.makedirs(args.dest, exist_ok=True)

    # 粗略磁盘检查（不阻止，只警告）
    free = shutil.disk_usage(args.dest).free
    need = total_bytes * 1.05  # 解压/临时文件余量
    print(f"磁盘   : 剩余 {human(free)}，预计需要 {human(need)}")
    if free < need:
        print(f"  [警告] 磁盘可能不够（还差 {human(need - free)}），"
              f"建议先下部分或换 --dest", file=sys.stderr)

    t0 = time.time()
    ok, failed = [], []
    for i, (name, s) in enumerate(plan, 1):
        target = os.path.join(args.dest, name)
        print(f"\n{'='*78}\n>>> [{i}/{len(plan)}] {name}  ({s['repo']})\n{'='*78}")
        before = dir_size(target)
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(
                    repo_id=s["repo"],
                    repo_type="dataset",
                    allow_patterns=s["patterns"],
                    local_dir=target,
                    max_workers=8,
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [第 {attempt}/{args.retries} 次失败] "
                      f"{type(e).__name__}: {str(e)[:200]}")
                if attempt == args.retries:
                    failed.append((name, f"{type(e).__name__}: {str(e)[:150]}"))
                    break
                time.sleep(3)
        else:  # pragma: no cover
            failed.append((name, "retries exhausted"))

        if name in dict(failed):
            print("  [跳过] 该源最终失败，继续下一个（重跑可断点续传）")
            continue
        after = dir_size(target)
        ok.append((name, after - before))
        print(f"  完成: {human(after)}（本次新增 {human(after - before)}）")

    # ---- manifest：明天 prepare_sources 按它找目录 ----
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint": os.environ.get("HF_ENDPOINT"),
        "preset": args.preset,
        "total_bytes": dir_size(args.dest),
        "sources": {
            n: {
                "repo": SOURCES[n]["repo"],
                "category": SOURCES[n]["category"],
                "patterns": SOURCES[n]["patterns"],
                "adapter": SOURCES[n]["adapter"],
                "path": os.path.join(args.dest, n),
            }
            for n, _ in plan
            if n in dict(ok)
        },
        "failed": [n for n, _ in failed],
    }
    with open(os.path.join(args.dest, "_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*78}")
    print(f"总计: {human(dir_size(args.dest))}   用时 {(time.time()-t0)/60:.1f} 分钟")
    if failed:
        print("\n失败的数据源（重跑同一命令即断点续传）:")
        for n, e in failed:
            print(f"  {n}: {e}")
        print("\n国内网络建议:  python scripts/download_v2.py --preset "
              f"{args.preset} --endpoint mirror")
        return 1
    print("\n全部完成。manifest: data/raw_src2/_manifest.json")
    print("下一步: 给 scripts/prepare_sources.py 加 adapter（见 docs/v2_data_sources.md §3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
