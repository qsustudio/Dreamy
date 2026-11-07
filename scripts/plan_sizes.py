"""按「固定训练时长」反推模型尺寸与 token 预算。

为什么需要这个
--------------
训练时间固定（比如 5090 上 10 小时），能花掉的算力就固定了：

    C = 有效峰值算力 × 时间 = 6 · N · D · f

其中 N 是参数量、D 是训练 token 数、f 是注意力带来的额外开销系数。
这条约束下 N 和 D 只能一个涨一个跌 —— 所以「多大模型配多少数据」
不能凭感觉，要算。

f 是什么
--------
业界常用的 6ND 只算了线性层。但序列长度 S 变大后，注意力的
QK^T 和 AV 也是实打实的算力：

    f = 1 + 2·S·dim·depth / N

本项目 S=1024、dim=768、depth=12、N=110M 时 f ≈ 1.17，
也就是真实算力比 6ND 多 17%。规划时不算这一项，会高估能训的 token 数。

Chinchilla 参考
---------------
算力最优是 D/N ≈ 20。但这只在「数据管够」时成立。
数据不够就必须承认现实：D 被数据卡死，N 只能按 D 反推（见 --data-tokens）。

用法
----
    # 5090、10 小时、留 1 小时余量，各 MFU 假设下能训多少 token
    python scripts/plan_sizes.py

    # 我手上只有 2.4B token，该选多大模型？
    python scripts/plan_sizes.py --data-tokens 2.4e9

    # 实测 MFU 是 31%，重新算
    python scripts/plan_sizes.py --mfu 0.31
"""
from __future__ import annotations

import argparse
import sys

# RTX 5090: 21760 CUDA cores @ 2.41GHz -> FP32 104.8 TFLOPS
# Tensor Core BF16 dense（FP32 累加）= 2x FP32 = 209.5 TFLOPS
DEFAULT_PEAK_TFLOPS = 209.5


def count_params(dim: int, depth: int, heads: int, head_dim: int,
                 vocab: int, ffn_mult: float) -> int:
    """按 x-transformers 的 TransformerWrapper(Decoder) 结构数参数。

    每层:
        Attention  to_qkv: dim -> 3*heads*head_dim, to_out: heads*head_dim -> dim
                   （默认无 bias）
        QK-Norm    q 和 k 各一个 head_dim 宽的 RMSNorm
        RMSNorm    pre-norm × 2
        SwiGLU     dim -> 2*ffn_dim -> ffn_dim -> dim，ffn_dim = int(dim*ffn_mult)
                   （ff_glu=True 时第一个线性层输出翻倍，GLU 再减半）
    词表:
        权重共享（tie_embedding），只算一份 vocab*dim
    """
    inner = heads * head_dim
    attn = dim * inner * 3 + inner * dim
    qk_norm = 2 * inner
    norms = 2 * dim
    ffn_dim = int(dim * ffn_mult)
    ffn = dim * ffn_dim * 2 + ffn_dim * dim
    per_layer = attn + qk_norm + norms + ffn
    return per_layer * depth + vocab * dim + dim


def attn_overhead(dim: int, depth: int, seq_len: int, n_params: int) -> float:
    """f = 1 + 2·S·dim·depth / N（注意力额外算力占比）。"""
    return 1.0 + 2.0 * seq_len * dim * depth / n_params


def plan(dim: int, depth: int, head_dim: int, vocab: int, ffn_mult: float,
         seq_len: int, mfu: float, peak: float, flops_budget: float):
    heads = dim // head_dim
    if heads * head_dim != dim:
        return None
    n = count_params(dim, depth, heads, head_dim, vocab, ffn_mult)
    f = attn_overhead(dim, depth, seq_len, n)
    tok = flops_budget / (6.0 * n * f)
    return {
        "dim": dim, "depth": depth, "heads": heads, "head_dim": head_dim,
        "params": n, "f": f, "tokens": tok, "dn": tok / n,
        "tok_s": tok / 1.0,  # 由调用方按时间换算
        "flops_per_tok": 6.0 * n * f,
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="按固定训练时长反推模型尺寸")
    p.add_argument("--hours", type=float, default=10.0, help="训练总时长（小时）")
    p.add_argument("--reserve-hours", type=float, default=1.0,
                   help="留给评测/存盘/重启的余量（小时）")
    p.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS,
                   help="BF16 dense 峰值 TFLOPS（5090 = 209.5）")
    p.add_argument("--mfu", type=float, default=None,
                   help="固定 MFU；不给则扫 25%%/30%%/35%%/40%%")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--ffn-mult", type=float, default=8 / 3)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dims", type=str, default="768,832,896,960,1024,1152")
    p.add_argument("--depths", type=str, default="10,12,14,16")
    p.add_argument("--data-tokens", type=float, default=None,
                   help="手上实际有多少 token；给了就反推该用多大模型")
    args = p.parse_args(argv)

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    hours = max(0.1, args.hours - args.reserve_hours)
    seconds = hours * 3600.0
    print("=" * 88)
    print(f"训练时长 {args.hours:.1f}h，扣掉余量 {args.reserve_hours:.1f}h "
          f"→ 实跑 {hours:.1f}h；峰值 {args.peak_tflops:.1f} TFLOPS(BF16 dense)")
    print(f"seq_len={args.seq_len}  vocab={args.vocab}  ffn_mult={args.ffn_mult:.3f}")
    print("=" * 88)

    mfus = [args.mfu] if args.mfu else [0.25, 0.30, 0.35, 0.40]

    if args.data_tokens:
        print(f"\n[反推] 手上 {args.data_tokens/1e9:.2f}B token，"
              f"各 MFU 下能塞下的最大模型：")
        print(f"  {'MFU':>6} {'总算力':>10} {'可选最大N':>11} {'对应dim/depth':>18}"
              f" {'D/N':>7}")
        for mfu in mfus:
            flops = mfu * args.peak_tflops * 1e12 * seconds
            best = None
            for dim in dims:
                for depth in depths:
                    r = plan(dim, depth, args.head_dim, args.vocab,
                             args.ffn_mult, args.seq_len, mfu,
                             args.peak_tflops, flops)
                    if not r:
                        continue
                    if r["tokens"] >= args.data_tokens:
                        if best is None or r["params"] > best["params"]:
                            best = r
            if best:
                print(f"  {mfu:>6.0%} {flops/1e18:>9.3f}E "
                      f"{best['params']/1e6:>10.1f}M "
                      f"{str(best['dim'])+'/'+str(best['depth']):>18} "
                      f"{args.data_tokens/best['params']:>7.1f}")
            else:
                print(f"  {mfu:>6.0%} {flops/1e18:>9.3f}E   （表里没有够小的配置）")
        print("\n  说明：D/N < 20 表示相对参数而言数据偏少（欠训练）；"
              "但数据封顶时，把时间用在更大的模型上仍然更划算，"
              "只是收益递减。")

    print(f"\n[正推] 各配置在 {hours:.1f}h 内能训完的 token 数（D/N 越接近 20 越充分）：")
    for mfu in mfus:
        flops = mfu * args.peak_tflops * 1e12 * seconds
        tflops = mfu * args.peak_tflops
        print(f"\n  ── MFU {mfu:.0%}（{tflops:.1f} TFLOPS）" + "─" * 60)
        print(f"  {'dim':>5}{'depth':>6}{'heads':>6}{'params':>10}"
              f"{'f':>6}{'token':>10}{'D/N':>7}{'tok/s':>9}{'耗时':>8}")
        for dim in dims:
            for depth in depths:
                r = plan(dim, depth, args.head_dim, args.vocab,
                         args.ffn_mult, args.seq_len, mfu,
                         args.peak_tflops, flops)
                if not r:
                    continue
                toks_per_s = tflops * 1e12 / r["flops_per_tok"]
                dn = r["dn"]
                flag = ""
                if 16 <= dn <= 24:
                    flag = " ←"
                print(f"  {r['dim']:>5}{r['depth']:>6}{r['heads']:>6}"
                      f"{r['params']/1e6:>9.1f}M{r['f']:>6.2f}"
                      f"{r['tokens']/1e9:>9.2f}B{dn:>7.1f}"
                      f"{toks_per_s/1000:>8.0f}k{args.hours:>7.1f}h{flag}")
    print("\n  f  = 注意力额外开销系数（真实算力 = 6ND·f）")
    print("  ←  = D/N 落在 16~24，Chinchilla 附近，比较健康的区间")
    print("\n  下一步：上机后先跑 scripts/bench_throughput.py 拿到实测 tok/s，")
    print("  用 --mfu 把这里换成实测值重算，再决定 --max-steps。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
