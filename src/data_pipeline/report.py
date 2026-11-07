"""数据健康度报告。

训练前把数据的真实样貌打出来 —— 配比失衡、复读、过短、
类别缺失这些问题必须在**开训之前**发现，而不是训了 6 小时才发现。
"""
from __future__ import annotations

from dataclasses import dataclass

from clean import CleanStats
from mix import MixResult

# T4 的 fp16 峰值算力约 65 TFLOPS；小模型实际利用率通常 15%~25%
T4_PEAK_FLOPS_FP16 = 65e12
DEFAULT_MFU = 0.20


def _bar(value: float, width: int = 24, max_value: float = 1.0) -> str:
    if max_value <= 0:
        return ""
    n = int(round(width * min(1.0, value / max_value)))
    return "#" * n


def estimate_training_time(
    total_tokens: int,
    n_params: int,
    seq_len: int,
    mfu: float = DEFAULT_MFU,
) -> tuple[float, float]:
    """粗估 T4 上的训练时间（秒）。返回 (tokens/s, 秒)。

    训练 FLOPs ≈ 6 * N * D（前向2 + 反向4）。T4 有效算力 ≈ 峰值 * MFU。
    """
    flops_per_token = 6 * n_params
    effective_flops = T4_PEAK_FLOPS_FP16 * mfu
    tokens_per_sec = max(1.0, effective_flops / flops_per_token)
    return tokens_per_sec, total_tokens / tokens_per_sec


def format_report(
    mix: MixResult,
    stats_by_cat: dict[str, CleanStats],
    total_tokens: int,
    n_params: int,
    seq_len: int,
) -> str:
    lines: list[str] = []
    add = lines.append
    W = 78

    add("=" * W)
    add("Dreamy 数据健康度报告".center(W - 16))
    add("=" * W)

    # ---------------- 1. 配比 ----------------
    add("")
    add("[1] 数据配比（目标 vs 实际，按 token/字符统计）")
    add("-" * W)
    add(f"{'类别':<12}{'目标':>8}{'实际':>8}{'偏差':>9}"
        f"{'可用字符':>12}{'取用字符':>12}")
    for k in mix.target:
        tgt = mix.target[k]
        act = mix.actual.get(k, 0.0)
        dev = act - tgt
        flag = "  <-- 偏离大" if abs(dev) > 0.05 else ""
        ac = mix.available_chars.get(k, 0) if hasattr(mix, "available_chars") else 0
        tc = mix.taken_chars.get(k, 0) if hasattr(mix, "taken_chars") else 0
        add(
            f"{k:<12}{tgt:>7.1%}{act:>8.1%}{dev:>+9.1%}"
            f"{ac/1e6:>10.1f}M{tc/1e6:>11.1f}M{flag}"
        )
    add("-" * W)
    tot_chars = mix.total_chars if hasattr(mix, "total_chars") else 0
    add(f"{'合计':<12}{'':>8}{'':>8}{'':>9}"
        f"{sum(mix.available_chars.values())/1e6:>10.1f}M"
        f"{tot_chars/1e6:>11.1f}M")

    # ---------------- 2. 清洗情况 ----------------
    add("")
    add("[2] 清洗情况（各类丢弃原因）")
    add("-" * W)
    for cat, st in stats_by_cat.items():
        if st.seen == 0 and st.kept == 0:
            add(f"  {cat}: 未重新清洗（复用 cleaned 数据）")
            continue
        keep_rate = st.kept / st.seen if st.seen else 0
        add(f"  {cat}: 读入 {st.seen:,} -> 保留 {st.kept:,} "
            f"({keep_rate:.0%})，丢弃 {st.drop_total:,}")
        if st.dropped:
            reasons = sorted(st.dropped.items(), key=lambda x: -x[1])
            parts = [f"{r}={n:,}" for r, n in reasons if n > 0]
            add(f"      丢弃原因: {', '.join(parts)}")

    # ---------------- 3. 长度与轮数 ----------------
    add("")
    add("[3] 长度与轮数分布（保留数据）")
    add("-" * W)
    all_chars: list[int] = []
    all_turns: list[int] = []
    for st in stats_by_cat.values():
        all_chars.extend(st.kept_chars)
        all_turns.extend(st.kept_turns)

    if all_chars:
        sc = sorted(all_chars)
        add(f"  字数  中位数 {sc[len(sc)//2]:>6,}  "
            f"P90 {sc[int(len(sc)*0.9)]:>6,}  "
            f"最大 {sc[-1]:>6,}")
        st_t = sorted(all_turns)
        add(f"  轮数  中位数 {st_t[len(st_t)//2]:>6}  "
            f"P90 {st_t[int(len(st_t)*0.9)]:>6}  "
            f"最大 {st_t[-1]:>6}")
    else:
        add("  （无数据）")

    # ---------------- 4. 规模与时间 ----------------
    add("")
    add("[4] 规模与 T4 训练时间估算")
    add("-" * W)
    add(f"  对话总数   : {mix.total:,}")
    add(f"  预估 tokens: {total_tokens:,}  (seq_len={seq_len})")
    add(f"  模型参数   : {n_params/1e6:.1f}M")
    tps, secs = estimate_training_time(total_tokens, n_params, seq_len)
    add(f"  预估吞吐   : ~{tps:,.0f} tokens/s（MFU={DEFAULT_MFU:.0%} 假设）")
    add(f"  预估时长   : ~{secs/3600:.1f} 小时（跑 1 个 epoch）")
    add(f"  建议 epochs: 2~3 -> ~{secs*2/3600:.1f} ~ {secs*3/3600:.1f} 小时")

    # ---------------- 5. 告警 ----------------
    add("")
    add("[5] 告警")
    add("-" * W)
    warnings: list[str] = []

    for k, dev in mix.deviations().items():
        if abs(dev) > 0.05:
            warnings.append(
                f"{k} 实际占比 {mix.actual.get(k,0):.1%} 偏离目标 "
                f"{mix.target[k]:.1%} 达 {dev:+.1%}"
            )
    for k, n in mix.available.items():
        if n == 0:
            warnings.append(f"{k} 类没有数据（目标 {mix.target[k]:.1%}）")

    # 通用语料占比过低是从零训练最容易崩的原因
    # （SFT 阶段没有 general 类别，不适用这条检查）
    if "general" in mix.target and mix.actual.get("general", 0.0) < 0.35:
        warnings.append(
            f"general（通用语料）实际仅 {mix.actual.get('general',0):.1%}。"
            f"从零训练时这一项低于 35% 极易导致语言能力不稳定/训练发散。"
        )

    # 复读率
    for cat, st in stats_by_cat.items():
        if st.seen and st.dropped.get("repetitive", 0) / st.seen > 0.20:
            warnings.append(
                f"{cat} 类 {st.dropped['repetitive']/st.seen:.0%} 因复读被丢，"
                f"原始数据质量堪忧，建议换源或加强清洗"
            )

    # 重复率过高：说明数据源本身高度重复，不是清洗的锅
    for cat, st in stats_by_cat.items():
        if not st.seen:
            continue
        dup = st.dropped.get("duplicate_exact", 0) + st.dropped.get("duplicate_near", 0)
        if dup / st.seen > 0.50:
            warnings.append(
                f"{cat} 类 {dup/st.seen:.0%} 被判为重复而丢弃，"
                f"说明该数据源本身高度重复。考虑换源，"
                f"或把 near_dedup_similarity 调高到 0.65 再看"
            )

    if mix.total < 1000:
        warnings.append(
            f"总数据仅 {mix.total:,} 条，从零训练可能不够（建议至少几万条对话 + 大量通用语料）"
        )

    if warnings:
        for w in warnings:
            add(f"  [!] {w}")
    else:
        add("  无明显问题")

    add("")
    add("=" * W)
    return "\n".join(lines)
