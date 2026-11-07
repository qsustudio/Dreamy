"""按配比混合各类数据。

**关键：按字符/ token 预算分配，不是按条数。**

为什么必须这样：
    不同类别的单条长度差得离谱 —— LCCC 闲聊平均 39 字/条，
    通用网页文章平均 1380 字/条，差 35 倍。
    如果按「条数」分配占比，同样 22% 的条数预算，
    闲聊只能贡献 22% 的 token，而实际它占满了大量条数，
    会把真实的 token 配比彻底拉偏（实测 chat 从 22% 变成 78%）。

    训练关心的是 token 量，所以这里统一按「字符数」做预算
    （中文字符 ≈ token 的稳定代理）。

某一类数据不够时：
    不硬凑，接受实际占比低于目标，并在报告里明确告警。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from schema import Conversation


@dataclass
class MixResult:
    conversations: list[Conversation]
    target: dict[str, float]          # 目标占比（已归一化）
    available: dict[str, int]         # 各类可用条数
    taken: dict[str, int]             # 各类实际取用条数
    available_chars: dict[str, int]   # 各类可用字符数
    taken_chars: dict[str, int]       # 各类实际取用字符数
    strict: bool = False
    unit: str = "chars"

    @property
    def total(self) -> int:
        return len(self.conversations)

    @property
    def total_chars(self) -> int:
        return sum(self.taken_chars.values())

    @property
    def actual(self) -> dict[str, float]:
        """实际 token 占比 —— 按字符算，这才是训练真正关心的。"""
        tot = self.total_chars
        if not tot:
            return {k: 0.0 for k in self.target}
        return {k: v / tot for k, v in self.taken_chars.items()}

    def deviations(self) -> dict[str, float]:
        return {k: self.actual.get(k, 0.0) - self.target[k] for k in self.target}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("配比权重之和必须大于 0")
    return {k: v / total for k, v in weights.items()}


def _take_by_char_budget(
    pool: list[Conversation],
    target_chars: float,
    avail_chars: int,
    rng: random.Random,
) -> list[Conversation]:
    """从池子里取约 target_chars 个字符的数据。

    先用随机顺序打散，再累加到预算为止，避免总是取同一条前缀。
    """
    if target_chars <= 0 or not pool:
        return []
    if target_chars >= avail_chars:
        return list(pool)

    idx = list(range(len(pool)))
    rng.shuffle(idx)
    picked: list[Conversation] = []
    acc = 0.0
    for i in idx:
        c = pool[i]
        picked.append(c)
        acc += c.total_chars
        if acc >= target_chars:
            break
    return picked


def mix_datasets(
    pools: dict[str, list[Conversation]],
    weights: dict[str, float],
    total: int | None = None,
    strict: bool = False,
    seed: int = 42,
    unit: str = "chars",
) -> MixResult:
    """按配比从各类池子里采样。

    Args:
        pools: 类别 -> 清洗后的数据列表
        weights: 类别 -> 目标占比（不必和为 1）
        total: 目标总规模。unit='chars' 时是目标总字符数；None = 尽量用满
        strict: True 时严格按占比，总量受最稀缺类别限制
        unit: 'chars'（默认，推荐）或 'items'（按条数，仅调试用）
    """
    target = normalize_weights(weights)
    available = {k: len(pools.get(k, [])) for k in target}
    available_chars = {
        k: sum(c.total_chars for c in pools.get(k, [])) for k in target
    }

    if unit == "items":
        # 兼容旧行为：按条数（已被证明会让配比严重失真，只留作调试）
        avail_metric = available
    else:
        avail_metric = available_chars

    # ---- 决定总量 ----
    if total is None:
        if strict:
            limits = [
                avail_metric[k] / target[k]
                for k in target
                if target[k] > 0 and avail_metric.get(k, 0) > 0
            ]
            total = int(min(limits)) if limits else 0
        else:
            total = sum(avail_metric.values())

    # ---- 各类预算 ----
    taken: dict[str, int] = {}
    taken_chars: dict[str, int] = {}
    picked_by_cat: dict[str, list[Conversation]] = {}
    rng = random.Random(seed)

    for k, w in target.items():
        pool = pools.get(k, [])
        if not pool:
            taken[k] = 0
            taken_chars[k] = 0
            picked_by_cat[k] = []
            continue

        budget = total * w
        if unit == "items":
            need = int(round(min(budget, available[k])))
            picked = (list(pool) if need >= len(pool)
                      else rng.sample(pool, need))
        else:
            picked = _take_by_char_budget(
                pool, min(budget, available_chars[k]),
                available_chars[k], rng,
            )

        for c in picked:
            c.meta["category"] = k
        picked_by_cat[k] = picked
        taken[k] = len(picked)
        taken_chars[k] = sum(c.total_chars for c in picked)

    out: list[Conversation] = []
    for k in target:
        out.extend(picked_by_cat.get(k, []))
    rng.shuffle(out)

    return MixResult(
        conversations=out,
        target=target,
        available=available,
        taken=taken,
        available_chars=available_chars,
        taken_chars=taken_chars,
        strict=strict,
        unit=unit,
    )
