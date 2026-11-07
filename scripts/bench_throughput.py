"""吞吐基准：给 5090 的 10 小时训练算 max-steps。

为什么需要这个脚本：
  * 110M -> 145M、micro batch 换大之后，tok/s 必须重新实测，不能拍脑袋；
  * 单卡 10 小时训练，--max-steps 填多了会被中途掐断，填少了浪费钱；
  * eval 曾因 batch 32 OOM，micro batch 到底能开多大也要实测出来。

用法::

    # 单次基准：60 步（前 10 步 warmup 不计），输出 tok/s / MFU / 10 小时计划
    python scripts/bench_throughput.py --dim 896 --depth 12 --heads 14 \\
        --head-dim 64 --batch-size 16 --grad-accum 2

    # 扫 micro batch：找出 32G 能开多大，OOM 的会标出来（不会中断）
    python scripts/bench_throughput.py --dim 896 --depth 12 --heads 14 \\
        --head-dim 64 --sweep 8,16,24,32

输出三块：
  1. 实测吞吐：tok/s、每步耗时、达到的 TFLOPS + MFU（对比 --peak-tflops）
  2. 训练计划：扣掉 --reserve-hours 后能跑多少 token / 多少步
  3. 可直接复制的训练命令（--max-steps 已算好）

约定：tok/s 会随模型尺寸/上下文长度/micro batch/驱动变化，换配置就重跑。
本脚本只读 data_dir，不写任何文件（不产生 checkpoint，不碰 runs/）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# 和 src/train.py 保持一致：必须在 import torch 之前设置显存分配策略，
# 否则基准测到的分配器行为和正式训练不一样。
# 新旧两个变量名都设（旧版 torch 读 PYTORCH_CUDA_ALLOC_CONF，新版读
# PYTORCH_ALLOC_CONF），值相同不冲突；用户自己设过就不覆盖。
_ALLOC_CONF = "expandable_segments:True"
if ("PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        and "PYTORCH_ALLOC_CONF" not in os.environ):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _ALLOC_CONF
    os.environ["PYTORCH_ALLOC_CONF"] = _ALLOC_CONF

# 让 src/ 下的模块可被 import（和 scripts/smoke_test.py 一致）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch                                                       # noqa: E402

from config import Config, ModelConfig                             # noqa: E402
from data import TokenStream, resolve_data_file                    # noqa: E402
from model import (                                               # noqa: E402
    build_model, count_params, describe_device, pick_device,
)
# 复用 train.py 里的 CUDA 开关 / 计时器 / OOM 判断，避免两处逻辑漂移
from train import StepTimer, enable_cuda_perf, is_cuda_oom          # noqa: E402


# ---------------------------------------------------------------- 参数
def parse_sweep_list(text: str) -> list[int]:
    try:
        values = [int(x) for x in text.replace(" ", "").split(",") if x]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--sweep 需要形如 8,16,24,32 的列表: {e}") from e
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("--sweep 里的 micro batch 必须是正整数")
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Dreamy 吞吐基准（给 5090 10 小时训练规划 max-steps）")
    p.add_argument("--data-dir", default=None,
                   help="默认取 config.Paths.data_dir；找不到数据就用随机 token")
    p.add_argument("--seq-len", type=int, default=1024, help="训练上下文长度")
    # ---- 模型尺寸（默认 = config.py 的 110M）----
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    # ---- batch ----
    p.add_argument("--batch-size", type=int, default=16,
                   help="micro batch（默认 16，5090 32G 起步值）")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="梯度累积（只影响 token/step，不影响 tok/s）")
    p.add_argument("--sweep", type=parse_sweep_list, default=None,
                   help="micro batch 扫描，如 8,16,24,32；OOM 的会标出来")
    p.add_argument("--sweep-steps", type=int, default=20,
                   help="扫描时每组跑多少步（默认 20）")
    p.add_argument("--sweep-warmup", type=int, default=5,
                   help="扫描时每组前多少步不计（默认 5）")
    # ---- 基准 ----
    p.add_argument("--steps", type=int, default=60,
                   help="基准总步数（默认 60，含 warmup）")
    p.add_argument("--warmup", type=int, default=10,
                   help="前多少步只热身不计时（默认 10）")
    p.add_argument("--peak-tflops", type=float, default=209.5,
                   help="GPU bf16 峰值 TFLOPS（默认 209.5 = RTX 5090）")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4_000_000,
                   help="最多读多少 token 做基准（默认 400 万，约 8MB）")
    # ---- 训练计划 ----
    p.add_argument("--hours", type=float, default=10.0,
                   help="总共可训练小时数（默认 10）")
    p.add_argument("--reserve-hours", type=float, default=1.0,
                   help="预留小时（启动/存盘/评估/意外，默认 1）")
    p.add_argument("--safety", type=float, default=0.95,
                   help="在可用时长上再乘的安全系数（默认 0.95）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------- 数据 / 模型
def read_vocab(data_dir: str, default: int) -> int:
    """优先用 data_dir/meta.json 的 vocab_size，和训练保持一致。"""
    path = os.path.join(data_dir, "meta.json")
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("vocab_size") or default)
    except (OSError, ValueError, TypeError):
        return default


def load_bench_stream(data_dir: str, seq_len: int, max_tokens: int,
                      vocab: int, device):
    """读一段真实 token 做基准。

    只读前 max_tokens 个（400 万 uint16 = 8MB），不把 2GB 全量载入内存：
    随机采样 batch 只需要一部分样本，开发机内存也小。
    找不到真实数据时退回随机 token —— 只测速度，loss 无意义。
    """
    if data_dir and os.path.isdir(data_dir):
        path = resolve_data_file(data_dir, ["tokens.bin", "tokens.npy"])
        if path:
            if path.endswith(".npy"):
                # mmap 读大 .npy，只拷前 max_tokens 个出来（避免只读数组告警）
                arr = np.load(path, mmap_mode="r").reshape(-1)
                tokens = np.array(arr[:max_tokens])
            else:
                tokens = np.fromfile(path, dtype=np.uint16, count=max_tokens)
            if len(tokens) > seq_len + 1:
                print(f"数据: {path}（只读前 {len(tokens):,} 个 token）")
                return TokenStream(tokens, seq_len, device=device)
    print("[!] 没找到 tokens.bin / tokens.npy，用随机 token 跑基准"
          "（吞吐不受影响，loss 无意义）")
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, vocab, size=max_tokens, dtype=np.uint16)
    return TokenStream(tokens, seq_len, device=device)


def build_bench_model(args, device):
    mc = ModelConfig()
    for name in ("dim", "depth", "heads", "head_dim"):
        value = getattr(args, name)
        if value is not None:
            setattr(mc, name, value)
    mc.max_seq_len = args.seq_len
    mc.num_tokens = read_vocab(args.data_dir, mc.num_tokens)
    return mc, build_model(mc).to(device)


def make_amp(device, no_amp):
    """和 src/train.py 相同的精度策略：bf16 优先，不行才 fp16+Scaler。"""
    use_amp = device.type == "cuda" and not no_amp
    bf16_ok = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = use_amp and not bf16_ok
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    label = (("bf16" if bf16_ok else "fp16 + GradScaler")
             if use_amp else "off")
    return use_amp, amp_dtype, scaler, label


def make_optimizer(model, device):
    """CUDA 上优先 fused AdamW（和训练一致），失败回退普通实现。"""
    kwargs = {"lr": 3e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)}
    if device.type == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
            return opt, True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


# ---------------------------------------------------------------- 基准
def train_step(model, stream, batch_size, grad_accum, optimizer, scaler,
               use_amp, amp_dtype, device, gen) -> None:
    """一步完整训练：forward + backward + clip + optimizer step。"""
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        x, y = stream.get_batch(batch_size, generator=gen)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ) / grad_accum
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


def run_bench(model, stream, device, batch_size, grad_accum, steps, warmup,
              optimizer, scaler, use_amp, amp_dtype, seed) -> dict:
    """跑 steps 步，前 warmup 步只热身，用 CUDA Event 统计剩余步的耗时。"""
    warmup = max(0, min(warmup, steps - 1))
    measured = max(1, steps - warmup)
    gen = torch.Generator().manual_seed(seed)

    for _ in range(warmup):
        train_step(model, stream, batch_size, grad_accum, optimizer, scaler,
                   use_amp, amp_dtype, device, gen)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timer = StepTimer(device)
    for _ in range(measured):
        train_step(model, stream, batch_size, grad_accum, optimizer, scaler,
                   use_amp, amp_dtype, device, gen)
        timer.tick()
    elapsed = max(timer.elapsed(), 1e-9)     # elapsed 内部会 synchronize

    tokens_per_step = batch_size * grad_accum * stream.seq_len
    result = {
        "batch_size": batch_size,
        "steps": measured,
        "seconds": elapsed,
        "step_ms": elapsed / measured * 1000.0,
        "tokens_per_step": tokens_per_step,
        "tok_s": measured * tokens_per_step / elapsed,
    }
    if device.type == "cuda":
        result["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    return result


def flops_per_token(n_params: int, mc: ModelConfig) -> float:
    """训练 FLOPs/token 近似：6N + 注意力项。

    6N        : 线性层/MLP 的前向+反向。N 含词表 embedding，而本模型
                权重共享（tie_embedding），输出头 matmul 的 FLOPs 正好
                对应这部分参数，所以 6N 是准确的。
    12·L·S·d  : 注意力 QK^T / AV；每层每 token 前向约 4·S·d，训练按 3 倍算。
                （因果 mask 实际只算约一半，这里按全量算，偏保守）
    """
    return 6 * n_params + 12 * mc.depth * mc.max_seq_len * mc.dim


def fmt_gb(result: dict) -> str:
    if "peak_alloc_gb" not in result:
        return "    -   "
    return f"{result['peak_alloc_gb']:.2f}GB"


# ---------------------------------------------------------------- 报告
def print_perf(result: dict, n_params: int, mc: ModelConfig, args) -> None:
    fpt = flops_per_token(n_params, mc)
    tflops = result["tok_s"] * fpt / 1e12
    mfu = tflops / args.peak_tflops if args.peak_tflops > 0 else 0.0
    print("-" * 70)
    print(f"实测吞吐 : {result['tok_s']:,.0f} tok/s")
    print(f"每步耗时 : {result['step_ms']:.1f} ms/step"
          f"（{result['steps']} 步实测，warmup 已排除）")
    print(f"算力利用 : {tflops:.1f} TFLOPS / 峰值 {args.peak_tflops:.1f} TFLOPS"
          f"  =>  MFU {mfu * 100:.1f}%")
    if "peak_alloc_gb" in result:
        print(f"显存峰值 : {result['peak_alloc_gb']:.2f} GB allocated / "
              f"{result['peak_reserved_gb']:.2f} GB reserved")
    print("-" * 70)


def compute_plan(result: dict, args) -> dict:
    usable_s = max(0.0, args.hours - args.reserve_hours) * 3600.0
    tokens_cap = result["tok_s"] * usable_s * args.safety
    step_tokens = max(1, result["tokens_per_step"])
    max_steps = int(tokens_cap // step_tokens)
    # 往下取整到 100 步：宁早几十步收工，也不要超时被掐断
    suggested = max_steps // 100 * 100
    wall_h = (suggested * step_tokens / result["tok_s"] / 3600.0
              if result["tok_s"] > 0 else 0.0)
    return {
        "usable_h": usable_s / 3600.0,
        "tokens": int(tokens_cap),
        "max_steps": max_steps,
        "suggested_steps": suggested,
        "suggested_tokens": suggested * step_tokens,
        "wall_h": wall_h,
    }


def print_plan(plan: dict, result: dict, args) -> None:
    print()
    print("=" * 70)
    print(f"{args.hours:.1f} 小时训练计划"
          f"（--reserve-hours {args.reserve_hours} / 安全系数 {args.safety}）")
    print("=" * 70)
    print(f"  可用时长       : {plan['usable_h']:.2f} h")
    print(f"  实测吞吐       : {result['tok_s']:,.0f} tok/s")
    print(f"  每步 token     : {result['tokens_per_step']:,}")
    print(f"  能跑 token     : {plan['tokens']:,}")
    print(f"  能跑步数       : {plan['max_steps']:,}")
    print(f"  建议 --max-steps : {plan['suggested_steps']:,}"
          f"（{plan['suggested_tokens']:,} token，约 {plan['wall_h']:.2f} h）")
    print("=" * 70)


def training_command(args, mc: ModelConfig, result: dict, steps: int) -> str:
    """生成可直接复制的训练命令（Linux 续行符；Windows 换成反引号）。"""
    return "\n".join([
        "python -u src/train.py --stage pretrain \\",
        f"    --dim {mc.dim} --depth {mc.depth} --heads {mc.heads} "
        f"--head-dim {mc.head_dim} --seq-len {mc.max_seq_len} \\",
        f"    --batch-size {result['batch_size']} "
        f"--grad-accum {args.grad_accum} \\",
        f"    --max-steps {steps} \\",
        "    --eval-every 500 --eval-batch-size 8 --eval-batches 64 \\",
        "    --ckpt-every 1000",
    ])


def run_sweep(model, stream, device, args, optimizer, scaler, use_amp,
              amp_dtype) -> list[dict]:
    """扫 micro batch，遇到 OOM 记下来继续扫下一个，不中断。"""
    print()
    print("=" * 70)
    print(f"micro batch 扫描（每组 {args.sweep_steps} 步，"
          f"前 {args.sweep_warmup} 步不计）")
    print("=" * 70)
    print(f"  {'batch':>5} | {'token/step':>12} | {'tok/s':>12} | "
          f"{'ms/step':>8} | {'显存':>7} | {'10h 步数':>9} | 状态")
    print("  " + "-" * 84)
    results = []
    for bs in args.sweep:
        try:
            res = run_bench(
                model, stream, device, batch_size=bs,
                grad_accum=args.grad_accum,
                steps=args.sweep_steps, warmup=args.sweep_warmup,
                optimizer=optimizer, scaler=scaler, use_amp=use_amp,
                amp_dtype=amp_dtype, seed=args.seed,
            )
        except RuntimeError as e:
            if not is_cuda_oom(e):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            print(f"  {bs:>5} | {'-':>12} | {'-':>12} | {'-':>8} | "
                  f"{'-':>7} | {'-':>9} | OOM")
            continue
        results.append(res)
        steps_10h = int(
            (res["tok_s"] * max(0.0, args.hours - args.reserve_hours) * 3600
             * args.safety) // max(1, res["tokens_per_step"]))
        print(f"  {bs:>5} | {res['tokens_per_step']:>12,} | "
              f"{res['tok_s']:>12,.0f} | {res['step_ms']:>8.0f} | "
              f"{fmt_gb(res):>7} | {steps_10h:>9,} | OK")
    print("  " + "-" * 84)
    if results:
        best = max(results, key=lambda r: r["tok_s"])
        print(f"  => 吞吐最高: micro batch {best['batch_size']}"
              f"（{best['tok_s']:,.0f} tok/s）")
        print(f"  => 32G 显存能用多大: 上面没标 OOM 的最大 batch = "
              f"{max(r['batch_size'] for r in results)}")
    return results


# ---------------------------------------------------------------- 主入口
def main(argv=None) -> int:
    args = parse_args(argv)
    if args.data_dir is None:
        args.data_dir = Config().paths.data_dir
    if args.steps < 2:
        raise SystemExit("--steps 至少 2（1 步 warmup + 1 步计时）")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise SystemExit("--batch-size / --grad-accum 必须是正整数")

    device = pick_device()
    print("=" * 70)
    print("Dreamy 吞吐基准")
    print(describe_device())
    print(f"data_dir = {args.data_dir}")
    print("=" * 70)
    if device.type == "cuda":
        enable_cuda_perf()

    torch.manual_seed(args.seed)
    mc, model = build_bench_model(args, device)
    n_params = count_params(model)
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    print(f"模型   : {n_params / 1e6:.2f}M 参数 | dim={mc.dim} depth={mc.depth} "
          f"heads={mc.heads} head_dim={mc.head_dim} "
          f"seq_len={mc.max_seq_len} vocab={mc.num_tokens}")
    print(f"训练   : micro batch {args.batch_size} x grad_accum "
          f"{args.grad_accum} = {tokens_per_step:,} token/step | "
          f"{args.steps} 步（前 {args.warmup} 步不计）")
    use_amp, amp_dtype, scaler, amp_label = make_amp(device, args.no_amp)
    print(f"混合精度: {amp_label}")
    stream = load_bench_stream(args.data_dir, args.seq_len, args.max_tokens,
                               mc.num_tokens, device)
    print(f"样本数 : {len(stream):,}（seq_len={args.seq_len}）")
    optimizer, fused = make_optimizer(model, device)
    if device.type == "cuda":
        print(f"优化器 : AdamW(fused={fused})")
    print()

    sweep_rows: list[dict] = []
    if args.sweep:
        sweep_rows = run_sweep(model, stream, device, args, optimizer, scaler,
                               use_amp, amp_dtype)

    # ---- 主基准 ----
    print()
    print(f">>> 主基准：micro batch {args.batch_size}，"
          f"{args.steps} 步（前 {args.warmup} 步不计）")
    try:
        result = run_bench(
            model, stream, device, batch_size=args.batch_size,
            grad_accum=args.grad_accum, steps=args.steps,
            warmup=args.warmup, optimizer=optimizer, scaler=scaler,
            use_amp=use_amp, amp_dtype=amp_dtype, seed=args.seed,
        )
    except RuntimeError as e:
        if not is_cuda_oom(e):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"\n[OOM] micro batch {args.batch_size} 在 seq_len "
              f"{args.seq_len} 下显存不够。")
        if not sweep_rows:
            print("       用 --sweep 8,16,24,32 扫一下，或 --batch-size 减半再试。")
            return 1
        result = max(sweep_rows, key=lambda r: r["tok_s"])
        print(f"       改用扫描里吞吐最高的 micro batch "
              f"{result['batch_size']} 出计划表。")

    print_perf(result, n_params, mc, args)
    plan = compute_plan(result, args)
    print_plan(plan, result, args)

    print()
    print("明天可直接复制（Linux；Windows PowerShell 把行尾的 \\ 换成反引号）：")
    print("-" * 70)
    print(training_command(args, mc, result, plan["suggested_steps"]))
    print("-" * 70)
    if sweep_rows:
        best = max(sweep_rows, key=lambda r: r["tok_s"])
        if best["batch_size"] != result["batch_size"]:
            print(f"提示：扫描里 micro batch {best['batch_size']} 吞吐最高"
                  f"（{best['tok_s']:,.0f} tok/s），把上面命令里的 "
                  f"--batch-size 换成它更划算（换完建议重跑主基准确认）。")
    print("注意：--max-steps 已扣掉预留时间和 5% 安全系数；")
    print("      如果换了 seq_len / 架构 / batch，先重跑本脚本再改 --max-steps。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
