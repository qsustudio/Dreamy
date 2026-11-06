"""数据清洗：规范化、过滤低质、去重、脱敏。

设计原则：**规则要能解释、统计要能看见**。
因为配比出问题导致训练崩掉时，你需要在训练前就看到数据长什么样，
而不是训完了才发现一半是复读数据。

调用方式见 build_dataset.py；也可以单独用 clean_conversations()。
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from dataclasses import dataclass, field

from schema import Conversation, Turn

# ---------------------------------------------------------------- 正则
_RE_MULTI_SPACE = re.compile(r"[ \t\u3000]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")
# 控制字符（保留换行）
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# 隐私
_RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")

# 判断「像乱码/无意义」
_RE_ONLY_SYMBOL = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass
class CleanConfig:
    min_turns: int = 2
    max_turns: int = 30
    max_chars: int = 3000
    min_chars_per_turn: int = 1
    max_repeat_ngram_ratio: float = 0.15
    repeat_ngram_n: int = 5
    max_punct_ratio: float = 0.5
    min_unique_char_ratio: float = 0.05
    exact_dedup: bool = True
    near_dedup: bool = True
    near_dedup_similarity: float = 0.55
    redact_pii: bool = True
    # 宽松模式：安全类数据不能按长度规则砍（短但关键）
    lenient: bool = False
    # 文档模式：通用语料是整段文本，没有轮次概念。
    # 不开这个开关会把通用语料全部当成「轮数不足」丢掉。
    document: bool = False
    min_chars_document: int = 50
    max_chars_document: int = 4000


@dataclass
class CleanStats:
    """清洗统计，用来生成健康度报告。"""
    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    kept_chars: list[int] = field(default_factory=list)
    kept_turns: list[int] = field(default_factory=list)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def drop_total(self) -> int:
        return sum(self.dropped.values())


# ---------------------------------------------------------------- 规范化
def normalize_text(s: str) -> str:
    """统一空白、去掉控制符和零宽字符。"""
    s = unicodedata.normalize("NFKC", s)
    s = _RE_ZERO_WIDTH.sub("", s)
    s = _RE_CONTROL.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_MULTI_SPACE.sub(" ", s)
    s = _RE_MULTI_NL.sub("\n\n", s)
    return s.strip()


def redact_pii(s: str) -> str:
    """脱敏。宁可多打码，也不要训练数据里带真实手机号。"""
    s = _RE_EMAIL.sub("<EMAIL>", s)
    s = _RE_PHONE.sub("<PHONE>", s)
    s = _RE_IDCARD.sub("<ID>", s)
    s = _RE_URL.sub("<URL>", s)
    return s


def normalize_turns(conv: Conversation, cfg: CleanConfig) -> Conversation:
    """规范化每个 turn，去掉空 turn，合并相邻同角色 turn。"""
    out: list[Turn] = []
    for t in conv.turns:
        text = normalize_text(t.text)
        if cfg.redact_pii:
            text = redact_pii(text)
        if not text:
            continue
        if out and out[-1].role == t.role:
            # 相邻同角色（脏数据常见）→ 合并，避免角色交替被破坏
            out[-1] = Turn(t.role, normalize_text(out[-1].text + "\n" + text))
        else:
            out.append(Turn(t.role, text))
    return Conversation(turns=out, meta=conv.meta)


# ---------------------------------------------------------------- 质量指标
def repeat_ngram_ratio(s: str, n: int = 5) -> float:
    """重复 n-gram 占比。复读/灌水数据会很高。"""
    if len(s) < n + 1:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def punct_ratio(s: str) -> float:
    if not s:
        return 1.0
    punct = sum(1 for ch in s if unicodedata.category(ch).startswith("P"))
    return punct / len(s)


def unique_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(set(s)) / len(s)


def is_garbage(text: str) -> bool:
    """整条就是符号/乱码。"""
    return bool(_RE_ONLY_SYMBOL.match(text))


# ---------------------------------------------------------------- 过滤
def check_document(conv: Conversation, cfg: CleanConfig) -> str | None:
    """通用语料（整段文本）的校验。只看文本质量，不看轮次。"""
    text = conv.text
    if not text.strip():
        return "empty"
    if len(text) < cfg.min_chars_document:
        return "document_too_short"
    if is_garbage(text):
        return "garbage"
    if repeat_ngram_ratio(text, cfg.repeat_ngram_n) > cfg.max_repeat_ngram_ratio:
        return "repetitive"
    if punct_ratio(text) > cfg.max_punct_ratio:
        return "too_much_punctuation"
    if unique_char_ratio(text) < cfg.min_unique_char_ratio:
        return "low_diversity"
    return None


def check_conversation(conv: Conversation, cfg: CleanConfig) -> str | None:
    """返回丢弃原因；None 表示保留。"""
    if cfg.document or conv.is_document:
        return check_document(conv, cfg)

    turns = conv.turns

    # 必须是 user/assistant 交替，且以 user 开头
    if not turns:
        return "empty"
    if turns[0].role != "user":
        return "not_user_first"

    if not cfg.lenient:
        if conv.num_turns < cfg.min_turns:
            return "too_few_turns"
        if conv.num_turns > cfg.max_turns:
            return "too_many_turns"
        if conv.total_chars > cfg.max_chars:
            return "too_long"

    for t in turns:
        if len(t.text) < cfg.min_chars_per_turn:
            return "turn_too_short"
        if is_garbage(t.text):
            return "garbage"
        if not cfg.lenient:
            if repeat_ngram_ratio(t.text, cfg.repeat_ngram_n) > cfg.max_repeat_ngram_ratio:
                return "repetitive"
            if punct_ratio(t.text) > cfg.max_punct_ratio:
                return "too_much_punctuation"
            if unique_char_ratio(t.text) < cfg.min_unique_char_ratio:
                return "low_diversity"

    # 整条对话只有一个 assistant 回复且极短 —— 信息量太低
    if not cfg.lenient:
        asst = [t for t in turns if t.role == "assistant"]
        if asst and sum(len(t.text) for t in asst) <= 2:
            return "assistant_too_short"

    return None


def truncate(conv: Conversation, max_chars: int, doc_max: int | None = None) -> Conversation:
    """超长内容截断。文档按字符截，对话按轮截。"""
    if conv.is_document:
        limit = doc_max or max_chars
        if conv.total_chars <= limit:
            return conv
        return Conversation([Turn(conv.turns[0].role, conv.turns[0].text[:limit])],
                            conv.meta)

    if conv.total_chars <= max_chars:
        return conv
    kept: list[Turn] = []
    total = 0
    for t in conv.turns:
        if total + len(t.text) > max_chars:
            break
        kept.append(t)
        total += len(t.text)
    # 保证以 assistant 结尾（截断后不该停在用户说完的地方）
    while kept and kept[-1].role != "assistant":
        kept.pop()
    return Conversation(turns=kept, meta=conv.meta)


# ---------------------------------------------------------------- 去重
# 用 MinHash + LSH 而不是 SimHash：
# 对话文本很短（几十到几百字），SimHash 在这种长度上信号很弱，
# 实测召回接近 0。MinHash 直接估计 3-gram 集合的 Jaccard 相似度，
# 对短文本准得多，而且 LSH 分桶保证不用 O(n^2) 比较。
NUM_HASH = 32
BANDS = 16               # 16 band x 2 row，对应 LSH 阈值约 0.25
ROWS = NUM_HASH // BANDS
SHINGLE_N = 3

# 用 Mersenne 素数 2^31-1：保证 a*h 不溢出 int64
_MERSENNE = (1 << 31) - 1
_HASH_PARAMS: tuple | None = None


def _get_hash_params():
    """生成 (a, b) 参数对。固定种子保证跨进程可复现。"""
    global _HASH_PARAMS
    if _HASH_PARAMS is None:
        import random as _random
        import numpy as _np

        rnd = _random.Random(0xC0FFEE)
        a = _np.array([rnd.randrange(1, _MERSENNE) for _ in range(NUM_HASH)],
                      dtype=_np.int64)
        b = _np.array([rnd.randrange(0, _MERSENNE) for _ in range(NUM_HASH)],
                      dtype=_np.int64)
        _HASH_PARAMS = (a, b)
    return _HASH_PARAMS


def _shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _minhash_signature(text: str):
    """返回长度 NUM_HASH 的 MinHash 签名。"""
    import numpy as np

    grams = _shingles(text)
    if not grams:
        return np.zeros(NUM_HASH, dtype=np.int64)

    hs = np.fromiter(
        (zlib.crc32(g.encode("utf-8")) % _MERSENNE for g in grams),
        dtype=np.int64, count=len(grams),
    )
    a, b = _get_hash_params()
    # (NUM_HASH, n_grams) -> 各行取最小
    vals = (a[:, None] * hs[None, :] + b[:, None]) % _MERSENNE
    return vals.min(axis=1)


def jaccard_from_signatures(sig1, sig2) -> float:
    """由签名估计原始集合的 Jaccard 相似度。"""
    return float((sig1 == sig2).mean())


class Deduper:
    """精确去重 + MinHash/LSH 近似去重。

    similarity_threshold 是判定为重复的 Jaccard 相似度阈值，
    实验上 0.55 对中文对话短文本比较合适：意思一样、措辞小改的能抓到，
    真正不同的内容不会被误杀。
    """

    def __init__(self, similarity_threshold: float = 0.55):
        self.threshold = similarity_threshold
        self._exact: set[int] = set()
        self._buckets: dict[tuple[int, bytes], list] = {}
        self.dups_exact = 0
        self.dups_near = 0
        self.last_was_exact = False

    def _band_keys(self, sig):
        import numpy as np

        keys = []
        for b in range(BANDS):
            chunk = sig[b * ROWS:(b + 1) * ROWS]
            keys.append((b, chunk.tobytes()))
        return keys

    def is_duplicate(self, conv: Conversation) -> bool:
        text = conv.text
        h = zlib.crc32(text.encode("utf-8"))
        if h in self._exact:
            self.dups_exact += 1
            self.last_was_exact = True
            return True

        sig = _minhash_signature(text)
        keys = self._band_keys(sig)

        for k in keys:
            for other in self._buckets.get(k, ()):
                if jaccard_from_signatures(sig, other) >= self.threshold:
                    self.dups_near += 1
                    self.last_was_exact = False
                    return True

        self._exact.add(h)
        for k in keys:
            self._buckets.setdefault(k, []).append(sig)
        return False


# ---------------------------------------------------------------- 主入口
def clean_conversations(
    convs,
    cfg: CleanConfig,
    deduper: Deduper | None = None,
    stats: CleanStats | None = None,
    progress_every: int = 0,
):
    """清洗迭代器。返回 (保留的 Conversation 迭代器, 统计)。"""
    stats = stats or CleanStats()
    keep_dedup = cfg.exact_dedup or cfg.near_dedup
    if keep_dedup and deduper is None:
        deduper = Deduper(cfg.near_dedup_similarity)

    for conv in convs:
        stats.seen += 1

        conv = normalize_turns(conv, cfg)
        reason = check_conversation(conv, cfg)
        if reason:
            stats.drop(reason)
            continue

        conv = truncate(conv, cfg.max_chars, cfg.max_chars_document)
        reason = check_conversation(conv, cfg) if not cfg.lenient else None
        if reason:
            stats.drop(reason)
            continue

        if keep_dedup:
            if deduper.is_duplicate(conv):
                reason = "duplicate_exact" if deduper.last_was_exact else "duplicate_near"
                stats.drop(reason)
                continue

        stats.kept += 1
        stats.kept_chars.append(conv.total_chars)
        stats.kept_turns.append(conv.num_turns)

        if progress_every and stats.seen % progress_every == 0:
            print(f"  ... 已处理 {stats.seen:,} 条，保留 {stats.kept:,} 条")

        yield conv
