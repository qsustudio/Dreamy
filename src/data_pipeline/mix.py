"""按配比混合各类数据。

核心难点不是「按比例抽」，而是**某一类数据不够时怎么办**。
默认策略：
  - 不做严格配比（否则一个很小的 safety 类会把你 95% 的数据砍掉）
  - 各类尽量用满，然后把「实际占比 vs 目标占比」的偏差明确报出来
  - 偏差超过阈值就告警，因为配比失衡正是训练崩掉的常见原因
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from schema import Conversation


@dataclass
class MixResult:
    conversations: list[Conversation]
    target: dict[str, float]          # 目标占比（已归一化）
    available: dict[str, int]         # 各类清洗后可用条数
    taken: dict[str, int]             # 各类实际取用条数
    strict: bool = False

    @property
    def total(self) -> int:
        return len(self.conversations)

    @property
    def actual(self) -> dict[str, float]:
        if not self.total:
            return {k: 0.0 for k in self.target}
        return {k: v / self.total for k, v in self.taken.items()}

    def deviations(self) -> dict[str, float]:
        return {k: self.actual.get(k, 0.0) - self.target[k] for k in self.target}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("配比权重之和必须大于 0")
    return {k: v / total for k, v in weights.items()}


def mix_datasets(
    pools: dict[str, list[Conversation]],
    weights: dict[str, float],
    total: int | None = None,
    strict: bool = False,
    seed: int = 42,
) -> MixResult:
    """按配比从各类池子里采样。

    Args:
        pools: 类别 -> 清洗后的对话列表
        weights: 类别 -> 目标占比（不必和为 1）
        total: 目标总条数；None = 尽量用满
        strict: True 时严格按占比，总条数受最稀缺类别限制
    """
    target = normalize_weights(weights)
    available = {k: len(pools.get(k, [])) for k in target}

    # ---- 决定总条数 ----
    if total is None:
        if strict:
            limits = [
                available[k] / target[k]
                for k in target
                if target[k] > 0 and available.get(k, 0) > 0
            ]
            total = int(min(limits)) if limits else 0
        else:
            total = sum(available.values())

    # ---- 各类取多少 ----
    taken: dict[str, int] = {}
    for k, w in target.items():
        if strict:
            need = int(round(total * w))
        else:
            # 非严格：按比例算，但不超过实际可用
            need = int(round(total * w))
            need = min(need, available.get(k, 0))
        taken[k] = min(need, available.get(k, 0))

    # ---- 采样 ----
    rng = random.Random(seed)
    out: list[Conversation] = []
    for k, n in taken.items():
        pool = pools.get(k, [])
        if n <= 0 or not pool:
            continue
        if n >= len(pool):
            picked = list(pool)
        else:
            picked = rng.sample(pool, n)
        for c in picked:
            c.meta["category"] = k
        out.extend(picked)

    rng.shuffle(out)

    return MixResult(
        conversations=out,
        target=target,
        available=available,
        taken=taken,
        strict=strict,
    )
