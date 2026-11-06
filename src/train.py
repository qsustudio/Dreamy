"""训练入口。

本地/Colab 通用：
    python src/train.py
    python src/train.py --max-steps 100 --batch-size 8    # 快速验证

checkpoint 和日志写回 DREAMY_OUT_DIR（Colab 上通常指向 Drive，
这样 Colab 断线重连后可以接着跑）。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 允许 `python src/train.py` 直接运行时正确 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import Config
from data import load_dataset
from model import build_model, describe_device, pick_device


def get_lr(step: int, cfg: Config) -> float:
    """线性 warmup + 余弦衰减。"""
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * step / max(1, t.warmup_steps)
    progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    return t.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(path: str, model, optimizer, step: int, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": cfg.model.__dict__,
            "train_config": cfg.train.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None) -> int:
    """恢复训练。返回已完成的 step。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"已恢复 checkpoint: step {ckpt['step']}  <- {path}")
    return int(ckpt["step"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 训练")
    p.add_argument("--data-dir", default=None, help="覆盖 DREAMY_DATA_DIR")
    p.add_argument("--out-dir", default=None, help="覆盖 DREAMY_OUT_DIR")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", default=None,
                   help="checkpoint 路径；传 'auto' 则用 ckpt_dir/last.pt")
    p.add_argument("--no-amp", action="store_true",
                   help="关闭混合精度（T4 上默认开）")
    args = p.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg.paths.data_dir = args.data_dir
    if args.out_dir:
        cfg.paths.out_dir = args.out_dir
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr

    # ---- 固定随机种子 ----
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = pick_device()
    print("=" * 60)
    print(describe_device())
    print(cfg.paths.describe())
    print("=" * 60)

    if device.type == "cuda":
        arch = torch.cuda.get_device_capability(0)
        if f"sm_{arch[0]}{arch[1]}" not in torch.cuda.get_arch_list():
            raise SystemExit(
                f"当前 torch 构建不含 {arch} 的 kernel，无法在该 GPU 上运行。\n"
                f"T4 需要 sm_75 —— 请改装 cu126 版：\n"
                f"  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126"
            )

    cfg.paths.ensure()

    # ---- 数据 ----
    print(f"\n从 {cfg.paths.data_dir} 加载数据 ...")
    stream = load_dataset(cfg.paths.data_dir, cfg.model.max_seq_len, device=device)
    print(f"共 {len(stream):,} 条样本（seq_len={cfg.model.max_seq_len}）")
    if len(stream) == 0:
        raise SystemExit("数据量不足一条样本，请检查数据文件。")

    # ---- 模型 / 优化器 ----
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    start_step = 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(cfg.paths.ckpt_dir, "last.pt")
    if resume_path:
        if os.path.isfile(resume_path):
            start_step = load_checkpoint(resume_path, model, optimizer)
        else:
            print(f"未找到 checkpoint: {resume_path}，从头开始训练")

    # T4 支持 fp16；用 torch.amp 的新 API 避免旧接口弃用警告
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    print(f"混合精度: {'fp16' if use_amp else 'off'}\n")

    # ---- 训练循环 ----
    model.train()
    gen = torch.Generator().manual_seed(cfg.seed)
    t0 = time.time()
    running = 0.0
    log_every = cfg.train.log_every

    for step in range(start_step, cfg.train.max_steps):
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, cfg)

        x, y = stream.get_batch(cfg.train.batch_size, generator=gen)

        with autocast:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()

        if (step + 1) % log_every == 0:
            avg = running / log_every
            elapsed = time.time() - t0
            print(
                f"step {step + 1:>6}/{cfg.train.max_steps} | "
                f"loss {avg:.4f} | ppl {math.exp(avg):.1f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.0f}s"
            )
            running = 0.0
            t0 = time.time()

        if (step + 1) % cfg.train.ckpt_every == 0 or (step + 1) == cfg.train.max_steps:
            path = os.path.join(cfg.paths.ckpt_dir, "last.pt")
            save_checkpoint(path, model, optimizer, step + 1, cfg)
            print(f"  -> 已保存 {path}")

    print("\n训练完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
