"""把 v2 新增数据源（data/raw_src2/）转成 Dreamy 的类别目录。

和 prepare_sources.py 分开写的理由：
    老的四个源（skypile/wikipedia/lccc/instruct）已经跑通过，不能动 ——
    改坏了就要重洗 1.07B token。这里只做增量，输出到
    `data/raw/<类别>_v2/`（和旧分片物理隔离，--clean-only 只洗新目录）。

源 -> 类别（全部已在 docs/v2_data_sources.md 核实过）::

    sharegpt90k_zh        -> chat_v2       真实用户多轮（ShareGPT 中文切片）
    counseling_*  (4 个)  -> empathy_v2    真实中文心理咨询多轮
    zhihu_kol             -> depth_v2      知乎长回答
    wikipedia_zh          -> knowledge_v2  中文维基 20231101（文档类）
    coig_cqia             -> 按关键词分流到 depth_v2/knowledge_v2/safety_v2
    cvalues_rlhf          -> safety_v2     CValues 安全偏好对（取正例回复）

用法::

    python scripts/prepare_sources_v2.py --dry-run --limit 200   # 先抽验结构
    python scripts/prepare_sources_v2.py --stage pretrain
    python scripts/prepare_sources_v2.py --stage both
    python scripts/prepare_sources_v2.py --only sharegpt90k_zh,counseling_cpsdd
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_sources import (                       # noqa: E402
    OUT_PRE, OUT_SFT, Sink, classify_instruct, iter_skypile,
)

SRC2 = os.path.join(ROOT, "data", "raw_src2")
CAT_SUFFIX = "_v2"

# 角色别名：咨询类数据用 client/counselor，别的一律靠这张表兜住
ROLE_MAP = {
    "human": "user", "user": "user", "client": "user", "patient": "user",
    "客户": "user", "用户": "user", "来访者": "user", "求助者": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "ai": "assistant", "助手": "assistant", "counselor": "assistant",
    "咨询师": "assistant", "therapist": "assistant", "doctor": "assistant",
    "system": "system",
}

# 源 -> (预训练类别 or None=关键词分流, 是否文档, SFT 类别 or None)
SOURCES2 = {
    "sharegpt90k_zh":         ("chat",      False, "chat"),
    "counseling_cpsdd":       ("empathy",   False, "empathy"),
    "counseling_psycheval":   ("empathy",   False, "empathy"),
    "counseling_soulchat_r1": ("empathy",   False, "empathy"),
    "counseling_sage":        ("empathy",   False, "empathy"),
    "zhihu_kol":              ("depth",     False, "depth"),
    "wikipedia_zh":           ("knowledge", True,  None),
    "coig_cqia":              (None,        False, None),
    "cvalues_rlhf":           ("safety",    False, "boundary"),
}

# ShareGPT-zh 的 category 里这些偏「技术问答」，对陪伴模型价值低，且占比很高
DROP_CATEGORY_KW = ("program", "code", "编程", "代码")


# ---------------------------------------------------------------- 写入
class SinkTagged(Sink):
    """写 `<prefix>-part-NNN.jsonl`，重跑时只清自己的文件，绝不碰老数据。

    为什么要这样：
      1. 老分片叫 part-000.jsonl，直接复用 Sink 会从 0 开始编号 -> 覆盖掉
         data/raw_sft/empathy 之类的既有 SFT 数据。加前缀就物理隔离开了。
      2. 重跑一次脚本不该产生重复，所以写之前先删同前缀的旧文件。
    """

    def __init__(self, root: str, prefix: str = "v2", **kw):
        super().__init__(root, **kw)
        self.prefix = prefix

    def flush(self, cat: str) -> None:
        if cat not in self._idx:
            d = os.path.join(self.root, cat)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                if f.startswith(self.prefix + "-part-"):
                    os.remove(os.path.join(d, f))
            self._idx[cat] = 0
        buf = self._buf.get(cat) or []
        if not buf:
            return
        d = os.path.join(self.root, cat)
        path = os.path.join(d, f"{self.prefix}-part-{self._idx[cat]:03d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._idx[cat] += 1
        self._buf[cat] = []


# ---------------------------------------------------------------- 读原始文件
def iter_data_files(root: str):
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.startswith(".") or f.endswith(".metadata"):
                continue
            p = os.path.join(dp, f)
            if f.endswith((".parquet", ".jsonl", ".jsonl.gz")):
                files.append(p)
            elif f.endswith(".json"):
                # 只认真正是数据的大 json，数据集自带的 dataset_info.json 之类跳过
                try:
                    if os.path.getsize(p) > 1 << 20:
                        files.append(p)
                except OSError:
                    pass
    files.sort()
    return files


def iter_records(root: str, limit: int | None = None):
    """逐条产出原始记录（parquet / jsonl / json / jsonl.gz 都能读）。"""
    n = 0
    for path in iter_data_files(root):
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4096):
                for rec in batch.to_pylist():
                    yield rec
                    n += 1
                    if limit and n >= limit:
                        return
            continue

        if path.endswith(".json"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            for rec in (data if isinstance(data, list) else [data]):
                yield rec
                n += 1
                if limit and n >= limit:
                    return
            continue

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield rec
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- 结构归一
def turns_from_messages(msgs) -> list[dict]:
    turns = []
    for it in msgs or ():
        if not isinstance(it, dict):
            continue
        raw_role = str(it.get("role") or it.get("from")
                       or it.get("speaker") or "").strip().lower()
        role = ROLE_MAP.get(raw_role)
        text = (it.get("content") or it.get("text")
                or it.get("value") or it.get("message"))
        if role == "system":
            continue
        if role and isinstance(text, str) and text.strip():
            turns.append({"role": role, "text": text.strip()})
    return turns


def _pair(u, a) -> list[dict] | None:
    if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
        return [{"role": "user", "text": u.strip()},
                {"role": "assistant", "text": a.strip()}]
    return None


def normalize(source: str, d):
    """原始记录 -> {"turns": [...]} / {"text": ...}；认不出返回 None。"""
    if isinstance(d, str):
        return {"text": d} if d.strip() else None
    if not isinstance(d, dict):
        return None

    # ---- 文档类 ----
    if source == "wikipedia_zh":
        for k in ("text", "wikitext", "content", "article"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return {"text": v}
        return None

    # ---- 心理咨询：messages[{role: client|counselor, content}] ----
    if source.startswith("counseling_"):
        lang = str(d.get("lang") or "").strip().lower()
        if lang and not (lang.startswith("zh") or lang.startswith("cn")
                         or lang.startswith("中文")):
            return None          # 个别仓库混了英文对话，直接丢
        turns = turns_from_messages(
            d.get("messages") or d.get("conversation") or d.get("dialogue"))
        return {"turns": turns} if len(turns) >= 2 else None

    # ---- ShareGPT 中文：conversation:[{human, assistant}] ----
    if source == "sharegpt90k_zh":
        turns = []
        for it in d.get("conversation") or d.get("conversations") or ():
            if not isinstance(it, dict):
                continue
            turns += turns_from_messages([{"role": "user",
                                           "content": it.get("human")}])
            turns += turns_from_messages([{"role": "assistant",
                                           "content": it.get("assistant")}])
        if not turns:
            turns = turns_from_messages(d.get("messages"))
        return {"turns": turns} if len(turns) >= 2 else None

    # ---- 知乎：大写 INSTRUCTION / RESPONSE ----
    if source == "zhihu_kol":
        low = {str(k).lower(): v for k, v in d.items()}
        return_pair = _pair(
            low.get("instruction") or low.get("question") or low.get("title"),
            low.get("response") or low.get("output") or low.get("answer"))
        if return_pair:
            return {"turns": return_pair}
        return None

    # ---- CValues 安全偏好：取正例回复 ----
    if source == "cvalues_rlhf":
        low = {str(k).lower(): v for k, v in d.items()}
        pair = _pair(low.get("prompt") or low.get("question")
                     or low.get("instruction"),
                     low.get("pos_resp") or low.get("response")
                     or low.get("answer"))
        return {"turns": pair} if pair else None

    # ---- COIG-CQIA 及兜底：按通用 instruction/output 解析 ----
    low = {str(k).lower(): v for k, v in d.items()}
    return normalize_generic(low)


def normalize_generic(d: dict):
    for k in ("conversations", "messages", "turns"):
        v = d.get(k)
        if isinstance(v, list):
            turns = turns_from_messages(v)
            if len(turns) >= 2:
                return {"turns": turns}
    for ku, ka in (("instruction", "output"), ("question", "answer"),
                   ("prompt", "response"), ("query", "response"),
                   ("input", "output"), ("instruction", "response")):
        pair = _pair(d.get(ku), d.get(ka))
        if pair:
            return {"turns": pair}
    for k in ("text", "content", "document"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return {"text": v}
    return None


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="把 v2 新增数据源转成类别目录")
    p.add_argument("--src2", default=SRC2)
    p.add_argument("--stage", default="both",
                   choices=["pretrain", "sft", "both"])
    p.add_argument("--only", default=None, help="逗号分隔，只处理这些源")
    p.add_argument("--limit", type=int, default=None, help="每个源最多读多少条")
    p.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    p.add_argument("--suffix", default=CAT_SUFFIX, help="输出类别后缀")
    args = p.parse_args(argv)

    if not os.path.isdir(args.src2):
        print(f"找不到 {args.src2}，先跑 scripts/download_v2.py", file=sys.stderr)
        return 1

    only = (set(x.strip() for x in args.only.split(",") if x.strip())
            if args.only else None)
    do_pre = args.stage in ("pretrain", "both")
    do_sft = args.stage in ("sft", "both")
    dry = args.dry_run

    print("=" * 74)
    print("v2 新增数据源 -> 类别目录")
    print("=" * 74)
    print(f"源目录: {args.src2}")
    print(f"输出  : {OUT_PRE}/<类别>{args.suffix}/  "
          f"{OUT_SFT}/<类别>/")
    print(f"阶段  : {args.stage}{'  (dry-run)' if dry else ''}")

    pre_sink = SinkTagged(OUT_PRE, prefix="v2", dry=dry) if do_pre else None
    sft_sink = (SinkTagged(OUT_SFT, prefix="v2", shard_size=20_000, dry=dry)
                if do_sft else None)
    reports = {}

    for name, (pre_cat, is_doc, sft_cat) in SOURCES2.items():
        if only and name not in only:
            continue
        d = os.path.join(args.src2, name)
        if not os.path.isdir(d):
            print(f"\n[跳过] {name}: 未下载")
            continue

        print(f"\n>>> {name}  ->  {pre_cat or '关键词分流'}{args.suffix}")
        read = kept = dropped = 0
        chars = 0
        for rec in iter_records(d, args.limit):
            read += 1
            if name == "sharegpt90k_zh":
                cat = str(rec.get("category") or "").lower() if isinstance(rec, dict) else ""
                if any(k in cat for k in DROP_CATEGORY_KW):
                    dropped += 1
                    continue
            norm = normalize(name, rec)
            if not norm:
                continue

            if "text" in norm:
                text = norm["text"]
                n_chars = len(text)
                if n_chars < 50:
                    continue
                kept += 1
                chars += n_chars
                if pre_sink and pre_cat:
                    pre_sink.add(pre_cat + args.suffix, {"text": text}, n_chars)
                continue

            turns = norm["turns"]
            n_chars = sum(len(t["text"]) for t in turns)
            if n_chars < 20:
                continue
            kept += 1
            chars += n_chars

            if pre_sink and do_pre:
                if pre_cat:
                    pre_sink.add(pre_cat + args.suffix, {"turns": turns}, n_chars)
                else:
                    user = next((t["text"] for t in turns
                                 if t["role"] == "user"), "")
                    asst = next((t["text"] for t in turns
                                 if t["role"] == "assistant"), "")
                    pcat, _ = classify_instruct(user, asst)
                    if pcat:
                        pre_sink.add(pcat + args.suffix, {"turns": turns}, n_chars)
            if sft_sink and do_sft and sft_cat:
                sft_sink.add(sft_cat, {"turns": turns}, n_chars)

            if read % 200_000 == 0:
                print(f"    {read:,} 读入 / {kept:,} 可用 ...")

        reports[name] = (read, kept, dropped, chars)
        print(f"    读入 {read:,}，可用 {kept:,}"
              f"{f'，按 category 丢弃 {dropped:,}' if dropped else ''}"
              f"，{chars/1e6:.1f}M 字符")

    if pre_sink:
        pre_sink.flush_all()
        print(f"\n=== 预训练新增类别 ({OUT_PRE}) ===")
        pre_sink.report()
    if sft_sink:
        sft_sink.flush_all()
        print(f"\n=== SFT 新增 ({OUT_SFT}) ===")
        sft_sink.report()

    print("\n下一步:")
    print(f"  python scripts/report_pools.py --include-raw")
    print(f"  # 然后按新的可用量调 configs/mix.yaml 权重，再跑 "
          f"build_dataset_streaming.py --clean-only <含 *_v2 的类别>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
