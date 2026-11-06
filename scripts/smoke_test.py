"""冒烟测试：验证环境是否正确，不需要数据和 GPU。

    python scripts/smoke_test.py

检查项：
  1. 依赖版本（torch / x-transformers）
  2. 设备与 T4(sm_75) kernel 支持
  3. 模型前向 + 反向
  4. 一次完整的 optimizer step
  5. 数据流切片逻辑（用合成数据，不读 Drive）
"""
from __future__ import annotations

import importlib.metadata as md
import os
import sys

import numpy as np
import torch

# 让 src/ 下的模块可被 import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import Config, ModelConfig          # noqa: E402
from data import TokenStream                    # noqa: E402
from model import build_model, describe_device  # noqa: E402

ok = True


def check(name: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=" * 60)
    print("Dreamy 冒烟测试")
    print("=" * 60)

    # 1. 依赖
    print("\n-- 依赖 --")
    try:
        ver = md.version("x-transformers")
        check("x-transformers 已安装", True, ver)
    except md.PackageNotFoundError:
        check("x-transformers 已安装", False, "pip install -r requirements.txt")
        return 1

    # 2. 设备
    print("\n-- 设备 --")
    print(describe_device())
    if torch.cuda.is_available():
        arch = torch.cuda.get_device_capability(0)
        arch_s = f"sm_{arch[0]}{arch[1]}"
        check(
            f"当前 GPU kernel 可用 ({arch_s})",
            arch_s in torch.cuda.get_arch_list(),
            torch.cuda.get_device_name(0),
        )
    else:
        print("  (无 GPU，跳过 GPU kernel 检查 —— 本机开发属正常)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 模型前向 + 反向
    print("\n-- 模型 --")
    cfg = ModelConfig(num_tokens=256, max_seq_len=64, dim=64, depth=2, heads=4)
    model = build_model(cfg).to(device)
    x = torch.randint(0, cfg.num_tokens, (2, 32), device=device)
    logits = model(x)
    check("前向形状正确", tuple(logits.shape) == (2, 32, cfg.num_tokens),
          str(tuple(logits.shape)))

    loss = logits.float().sum()
    loss.backward()
    n_params = sum(1 for _ in model.parameters())
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    check("反向传播覆盖全部参数", n_grads == n_params, f"{n_grads}/{n_params}")

    # 4. optimizer step
    print("\n-- 优化器 --")
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    y = torch.randint(0, cfg.num_tokens, (2, 32), device=device)
    out = model(x)
    l = torch.nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
    opt.zero_grad()
    l.backward()
    opt.step()
    changed = sum(
        1 for b, p in zip(before, model.parameters()) if not torch.equal(b, p.detach())
    )
    check("参数已更新", changed == n_params, f"loss={l.item():.4f}, {changed}/{n_params}")

    # 5. 数据切片（合成数据）
    print("\n-- 数据流 --")
    toks = np.random.randint(0, 256, size=1000).astype(np.uint16)
    stream = TokenStream(toks, seq_len=64)
    check("样本数计算正确", len(stream) == (1000 - 1) // 64, f"{len(stream)} 条")
    bx, by = stream.get_batch(4, generator=torch.Generator().manual_seed(0))
    check("batch 形状正确", tuple(bx.shape) == (4, 64) and tuple(by.shape) == (4, 64),
          f"x{tuple(bx.shape)} y{tuple(by.shape)}")
    check("标签是输入左移一位",
          torch.equal(bx[:, 1:], by[:, :-1]),
          "y[:, :-1] == x[:, 1:]")

    # 6. 路径配置
    print("\n-- 配置 --")
    c = Config()
    print(c.paths.describe())

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
