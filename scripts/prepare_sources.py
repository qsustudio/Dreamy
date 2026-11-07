"""把下载的原始数据集转成 Dreamy 的目录结构。

输入: data/raw_src/{skypile,wikipedia,lccc,instruct}
输出:
    data/raw/<类别>/          预训练用
    data/raw_sft/<类别>/      SFT 用

数据源映射（都是已清洗过的公开数据集）::

    skypile    -> general   通用中文网页（SkyPile，已质量过滤）
    wikipedia  -> knowledge 中文维基（已去重/低质量过滤）
    lccc       -> chat      中文闲聊（原数据浅，靠下游清洗过滤）
    instruct   -> empathy / depth / persona / safety（关键词分流）
                 以及 data/raw_sft 下的 SFT 类别

用法::

    python scripts/prepare_sources.py --stage pretrain
    python scripts/prepare_sources.py --stage sft
    python scripts/prepare_sources.py --stage both
    python scripts/prepare_sources.py --dry-run        # 只看能产出多少
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw_src")
OUT_PRE = os.path.join(ROOT, "data", "raw")
OUT_SFT = os.path.join(ROOT, "data", "raw_sft")

# ---------------------------------------------------------------- 关键词分流
# 指令数据没有明确类别，按内容关键词分流到预训练/SFT 类别。
EMPATHY_KW = ["难过", "难受", "想哭", "失眠", "焦虑", "抑郁", "孤独", "失恋",
              "痛苦", "压力", "害怕", "委屈", "崩溃", "emo", "情绪", "心理",
              "陪伴", "倾诉", "安慰", "心累", "烦躁", "无助"]
DEEP_KW = ["为什么", "意义", "哲学", "本质", "思考", "观点", "看法", "如何看待",
           "人生", "价值观", "存在", "自由", "幸福", "人性", "社会现象"]
PERSONA_KW = ["你是谁", "你叫什么", "介绍一下你", "你是ai", "你是机器人",
              "你的性格", "你叫什么名字"]
SAFETY_KW = ["自杀", "自残", "伤害自己", "不想活", "结束生命", "活着没意思",
             "违法", "毒品", "诈骗", "黑客", "炸弹"]
FOLLOWUP_KW = ["然后呢", "接着说", "展开说说", "还有吗", "具体点", "举个例子"]
OPINION_KW = ["你觉得", "你认为", "你支持", "你反对", "你怎么看"]


def count_hits(text: str, kws) -> int:
    return sum(1 for k in kws if k in text)


def classify_instruct(user: str, asst: str) -> tuple[str, str]:
    """返回 (预训练类别 or '', SFT 类别)。"""
    blob = user + asst
    # 安全优先：一旦命中就走 boundary，不能被别的类抢走
    if count_hits(blob, SAFETY_KW):
        return "safety", "boundary"
    if count_hits(blob, PERSONA_KW):
        return "persona", "chat"
    e = count_hits(blob, EMPATHY_KW)
    d = count_hits(blob, DEEP_KW)
    f = count_hits(blob, FOLLOWUP_KW)
    o = count_hits(blob, OPINION_KW)
    if e >= 2 and e >= d:
        return "empathy", "empathy"
    if d >= 2 and d > e:
        return "depth", "depth"
    if o >= 1:
        return "", "opinion"
    if f >= 1:
        return "", "followup"
    return "", "chat"


# ---------------------------------------------------------------- 写入
class Sink:
    """按类别写 JSONL，自动分片。"""

    def __init__(self, root: str, shard_size: int = 200_000, dry: bool = False):
        self.root = root
        self.shard_size = shard_size
        self.dry = dry
        self.counts: dict[str, int] = {}
        self._buf: dict[str, list] = {}
        self._idx: dict[str, int] = {}

    def add(self, cat: str, rec: dict, n_chars: int = 0) -> None:
        self.counts[cat] = self.counts.get(cat, 0) + 1
        self.counts[cat + ":chars"] = self.counts.get(cat + ":chars", 0) + n_chars
        if self.dry:
            return
        buf = self._buf.setdefault(cat, [])
        buf.append(rec)
        if len(buf) >= self.shard_size:
            self.flush(cat)

    def flush(self, cat: str) -> None:
        buf = self._buf.get(cat) or []
        if not buf:
            return
        d = os.path.join(self.root, cat)
        os.makedirs(d, exist_ok=True)
        idx = self._idx.get(cat, 0)
        path = os.path.join(d, f"part-{idx:03d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._idx[cat] = idx + 1
        self._buf[cat] = []

    def flush_all(self) -> None:
        for cat in list(self._buf):
            self.flush(cat)

    def report(self) -> None:
        print(f"\n  产出类别（字符数 ≈ 后面 token 上限）:")
        cats = [c for c in self.counts if not c.endswith(":chars")]
        for c in sorted(cats):
            print(f"    {c:<12} {self.counts[c]:>10,} 条  "
                  f"{self.counts.get(c+':chars',0)/1e6:>8.1f}M 字符")


# ---------------------------------------------------------------- 各数据源
def iter_skypile(root: str, limit_rows: int | None):
    import pyarrow.parquet as pq

    files = []
    for dp, _, fs in os.walk(root):
        files += [os.path.join(dp, f) for f in fs if f.endswith(".parquet")]
    files.sort()
    n = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if not t:
                    continue
                yield t
                n += 1
                if limit_rows and n >= limit_rows:
                    return


def iter_wikipedia(root: str, limit_rows: int | None):
    files = []
    for dp, _, fs in os.walk(root):
        files += [os.path.join(dp, f) for f in fs if f.endswith(".jsonl")]
    files.sort()
    n = 0
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("是否重复文件"):
                    continue
                paras = d.get("段落") or []
                text = "\n".join(p.get("内容", "") for p in paras
                                 if not p.get("是否重复"))
                if text.strip():
                    yield text
                    n += 1
                    if limit_rows and n >= limit_rows:
                        return


_RE_CJK = re.compile(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])")


def despace_cjk(text: str) -> str:
    """LCCC 的语料是「词间带空格」的分词形式，需要还原成正常中文。

    只去掉两个汉字之间的空格，避免把英文单词粘在一起。
    """
    prev = None
    while prev != text:
        prev = text
        text = _RE_CJK.sub(r"\1\2", text)
    return text


def iter_lccc(root: str, limit_rows: int | None):
    """LCCC: 每行是一个 utterance 的 JSON 数组（词间带空格）。"""
    files = []
    for dp, _, fs in os.walk(root):
        files += [os.path.join(dp, f) for f in fs
                  if f.endswith((".jsonl.gz", ".jsonl"))]
    files.sort()
    n = 0
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    utts = d.get("dialog") or d.get("conversation") or []
                elif isinstance(d, list):
                    utts = d
                else:
                    continue
                turns = []
                for i, u in enumerate(utts):
                    if not isinstance(u, str):
                        continue
                    u = despace_cjk(u).strip()
                    if not u:
                        continue
                    turns.append({"role": "user" if i % 2 == 0 else "assistant",
                                  "text": u})
                if len(turns) >= 2:
                    yield {"turns": turns}
                    n += 1
                    if limit_rows and n >= limit_rows:
                        return


def iter_instruct(root: str, limit_rows: int | None):
    """指令数据：尽量识别 instruction/output、conversations 等结构。"""
    files = []
    for dp, _, fs in os.walk(root):
        files += [os.path.join(dp, f) for f in fs if f.endswith(".jsonl")]
    files.sort()
    n = 0
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = normalize_instruct(d)
                if rec:
                    yield rec
                    n += 1
                    if limit_rows and n >= limit_rows:
                        return


def normalize_instruct(d):
    if not isinstance(d, dict):
        return None
    # 多轮
    for k in ("conversations", "messages", "turns"):
        v = d.get(k)
        if isinstance(v, list):
            turns = []
            for it in v:
                if not isinstance(it, dict):
                    continue
                role = it.get("role") or it.get("from")
                text = it.get("content") or it.get("value") or it.get("text")
                if not role or not text:
                    continue
                role = str(role).lower()
                role = "user" if role in ("human", "user") else "assistant"
                turns.append({"role": role, "text": str(text)})
            if len(turns) >= 2:
                return {"turns": turns}
    # 单轮
    for ku, ka in (("instruction", "output"), ("question", "answer"),
                   ("prompt", "response"), ("query", "response"), ("input", "output")):
        u, a = d.get(ku), d.get(ka)
        if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
            return {"turns": [{"role": "user", "text": u},
                              {"role": "assistant", "text": a}]}
    return None


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="转换 HF 原始数据到 Dreamy 目录结构")
    p.add_argument("--src", default=SRC)
    p.add_argument("--stage", default="both",
                   choices=["pretrain", "sft", "both"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="每个源最多处理多少条（调试用）")
    args = p.parse_args(argv)

    if not os.path.isdir(args.src):
        print(f"找不到 {args.src}，先跑 scripts/download_data.py", file=sys.stderr)
        return 1

    do_pre = args.stage in ("pretrain", "both")
    do_sft = args.stage in ("sft", "both")
    dry = args.dry_run

    print("=" * 74)
    print("转换 HF 原始数据 -> Dreamy 目录")
    print("=" * 74)
    print(f"源目录: {args.src}")
    print(f"阶段  : {args.stage}{'  (dry-run)' if dry else ''}")

    pre_sink = Sink(OUT_PRE, dry=dry) if do_pre else None
    sft_sink = Sink(OUT_SFT, shard_size=20_000, dry=dry) if do_sft else None

    # ---- 文档类：skypile -> general, wikipedia -> knowledge ----
    if do_pre:
        for name, it, cat in [
            ("skypile", iter_skypile, "general"),
            ("wikipedia", iter_wikipedia, "knowledge"),
        ]:
            d = os.path.join(args.src, name)
            if not os.path.isdir(d):
                print(f"\n[跳过] {name}: 未下载")
                continue
            print(f"\n>>> {name} -> {cat}")
            n = 0
            for text in it(d, args.limit):
                pre_sink.add(cat, {"text": text}, len(text))
                n += 1
                if n % 100_000 == 0:
                    print(f"    {n:,} 条 ...")
            print(f"    完成 {n:,} 条")

        # ---- 对话：lccc -> chat ----
        d = os.path.join(args.src, "lccc")
        if os.path.isdir(d):
            print("\n>>> lccc -> chat")
            n = 0
            for rec in iter_lccc(d, args.limit):
                chars = sum(len(t["text"]) for t in rec["turns"])
                pre_sink.add("chat", rec, chars)
                n += 1
                if n % 100_000 == 0:
                    print(f"    {n:,} 条 ...")
            print(f"    完成 {n:,} 条")

    # ---- 指令：按关键词分流 ----
    d = os.path.join(args.src, "instruct")
    if os.path.isdir(d) and (do_pre or do_sft):
        print("\n>>> instruct -> 关键词分流")
        n = 0
        for rec in iter_instruct(d, args.limit):
            turns = rec["turns"]
            user = next((t["text"] for t in turns if t["role"] == "user"), "")
            asst = next((t["text"] for t in turns if t["role"] == "assistant"), "")
            pre_cat, sft_cat = classify_instruct(user, asst)
            chars = sum(len(t["text"]) for t in turns)
            if do_pre and pre_cat:
                pre_sink.add(pre_cat, rec, chars)
            if do_sft:
                sft_sink.add(sft_cat, rec, chars)
            n += 1
            if n % 100_000 == 0:
                print(f"    {n:,} 条 ...")
        print(f"    完成 {n:,} 条")

    if pre_sink:
        pre_sink.flush_all()
        print("\n=== 预训练数据 (%s) ===" % OUT_PRE)
        pre_sink.report()
    if sft_sink:
        sft_sink.flush_all()
        print("\n=== SFT 数据 (%s) ===" % OUT_SFT)
        sft_sink.report()

    print("\n下一步: python scripts/build_dataset.py --stage both --tokenize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
