"""训练入口：预训练 + SFT 两个阶段。

预训练（从零）:
    python src/train.py --stage pretrain
    python src/train.py --stage pretrain --max-steps 200      # 冒烟

SFT（从预训练 checkpoint 出发，只对 assistant 回复算 loss）:
    python src/train.py --stage sft --init-from <pretrain ckpt>

checkpoint 和日志写回 DREAMY_OUT_DIR。Colab 断线后加 --resume auto 续训。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# 允许 `python src/train.py` 直接运行时正确 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import Config
from data import (
    load_dataset, load_sft_dataset, load_val_dataset,
)
from model import build_model, count_params, describe_device, pick_device


# ---------------------------------------------------------------- 调度
def lr_at(step: int, total: int, warmup: int, base_lr: float,
          min_ratio: float) -> float:
    """线性 warmup + 余弦衰减到 base_lr * min_ratio。

    spec §5.1：Warmup 2%，最低 LR 3e-5（= 3e-4 * 0.1）。
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(path, model, optimizer, scaler, step, cfg, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_config": cfg.model.__dict__,
        "train_config": cfg.train.__dict__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scaler=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f"已恢复 checkpoint: step {ckpt.get('step', 0)}  <- {path}")
    return int(ckpt.get("step", 0))


def load_weights_only(path, model) -> None:
    """只加载模型权重（SFT 从预训练出发时用，不继承优化器状态）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"已加载预训练权重: {path} (step {ckpt.get('step', '?')})")


# ---------------------------------------------------------------- 公共
def setup(args, cfg):
    device = pick_device()
    print("=" * 66)
    print(describe_device())
    print(cfg.paths.describe())
    print("=" * 66)

    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch not in torch.cuda.get_arch_list():
            raise SystemExit(
                f"当前 torch 构建不含 {arch} 的 kernel，无法在该 GPU 上运行。\n"
                f"T4 需要 sm_75 —— 请改装 cu126 版：\n"
                f"  pip install torch==2.14.1 --index-url "
                f"https://download.pytorch.org/whl/cu126"
            )

    cfg.paths.ensure()

    # 用 meta.json 里的词表大小对齐模型
    meta_path = os.path.join(cfg.paths.data_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if "vocab_size" in meta and cfg.model.num_tokens != meta["vocab_size"]:
            print(f"按 meta.json 对齐词表: num_tokens {cfg.model.num_tokens} "
                  f"-> {meta['vocab_size']}")
            cfg.model.num_tokens = int(meta["vocab_size"])
    return device


@torch.no_grad()
def eval_loss(model, batches, device, use_amp: bool, max_batches: int = 20,
              amp_dtype=None):
    """评估集平均 loss。"""
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(batches):
        if i >= max_batches:
            break
        with torch.autocast(device_type="cuda",
                            dtype=amp_dtype or torch.float16,
                            enabled=use_amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=-100,
            )
        total += float(loss)
        n += 1
    model.train()
    return total / n if n else None


def check_stream_usable(stream, name: str, min_len: int = 1):
    """验证集太小时跳过，而不是在训练中途崩掉。"""
    if stream is None:
        return None
    n = len(stream)
    if n < min_len:
        print(f"[!] {name} 太小（{n} 条样本，不足一个序列），跳过评估。"
              f"数据量上来后会自动启用。")
        return None
    return stream


# ---------------------------------------------------------------- 主入口
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 训练")
    p.add_argument("--stage", default="pretrain", choices=["pretrain", "sft"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="SFT 用")
    p.add_argument("--batch-size", type=int, default=None, help="micro batch")
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", default=None,
                   help="同阶段续训：checkpoint 路径，或 'auto'")
    p.add_argument("--init-from", default=None,
                   help="SFT 从预训练 checkpoint 初始化（只加载权重）")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--eval-every", type=int, default=None,
                   help="每多少步评估验证集（0 = 不评估）")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="每多少步保存 checkpoint（0 = 只在结束保存）")
    args = p.parse_args(argv)

    cfg = Config()
    if args.data_dir:
        cfg.paths.data_dir = args.data_dir
    if args.out_dir:
        cfg.paths.out_dir = args.out_dir

    is_sft = args.stage == "sft"
    sc = cfg.sft if is_sft else cfg.train
    if args.batch_size:
        sc.batch_size = args.batch_size
    if args.grad_accum:
        sc.grad_accum = args.grad_accum
    if args.lr:
        sc.lr = args.lr

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = setup(args, cfg)

    # ---- 数据 ----
    print(f"\n[{args.stage}] 从 {cfg.paths.data_dir} 加载数据 ...")
    val_stream = None
    if is_sft:
        stream = load_sft_dataset(cfg.paths.data_dir, device=device)
        try:
            val_stream = load_sft_dataset(cfg.paths.data_dir, device=device,
                                          split="val")
        except FileNotFoundError:
            val_stream = None
        print(f"SFT 样本 {len(stream):,} 条，seq_len={stream.seq_len}")
        val_stream = check_stream_usable(val_stream, "SFT 验证集")
    else:
        stream = load_dataset(cfg.paths.data_dir, cfg.model.max_seq_len,
                              device=device)
        val_stream = load_val_dataset(cfg.paths.data_dir,
                                      cfg.model.max_seq_len, device=device)
        print(f"共 {len(stream):,} 条样本（seq_len={cfg.model.max_seq_len}）")
        val_stream = check_stream_usable(val_stream, "预训练验证集")
        if val_stream:
            print(f"验证集 {len(val_stream):,} 条样本")
    if len(stream) == 0:
        raise SystemExit(
            f"训练数据不足一个序列（seq_len={cfg.model.max_seq_len}）。\n"
            f"请检查 {cfg.paths.data_dir}/tokens.bin 或降低上下文长度。"
        )

    # ---- 模型 ----
    model = build_model(cfg.model).to(device)
    n_params = count_params(model)
    print(f"模型参数量: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sc.lr,
        weight_decay=sc.weight_decay,
        betas=(cfg.train.beta1, cfg.train.beta2),
    )
    # ---- 精度策略 ----
    # 4090 (sm_89) 原生支持 bf16：不需要 GradScaler，数值更稳。
    # T4 (sm_75) 不支持 bf16，自动回退到 fp16 + GradScaler。
    use_amp = (device.type == "cuda") and not args.no_amp
    bf16_ok = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = use_amp and not bf16_ok
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype,
                              enabled=use_amp)

    eff_tokens = sc.batch_size * sc.grad_accum * cfg.model.max_seq_len
    print(f"有效 batch: {sc.batch_size} x {cfg.model.max_seq_len} x "
          f"{sc.grad_accum} = {eff_tokens:,} token/step")
    print(f"混合精度: "
          f"{('bf16' if bf16_ok else 'fp16 + GradScaler') if use_amp else 'off'}\n")

    gen = torch.Generator().manual_seed(cfg.seed)

    # ---- 起始权重 ----
    start_step = 0
    if is_sft and args.init_from:
        load_weights_only(args.init_from, model)
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(cfg.paths.ckpt_dir, f"{args.stage}_last.pt")
    if resume_path:
        if os.path.isfile(resume_path):
            start_step = load_checkpoint(resume_path, model, optimizer, scaler)
        else:
            print(f"未找到 checkpoint: {resume_path}，从头开始")

    # ---- 步数 & 调度 ----
    if is_sft:
        steps_per_epoch = math.ceil(len(stream) / sc.batch_size)
        total_steps = steps_per_epoch * (args.epochs or sc.epochs)
        if args.max_steps:
            total_steps = min(total_steps, start_step + args.max_steps)
        warmup = int(total_steps * sc.warmup_ratio)
        min_ratio = sc.min_lr_ratio
    else:
        total_steps = args.max_steps or cfg.train.max_steps
        warmup = int(total_steps * cfg.train.warmup_ratio)
        min_ratio = cfg.train.min_lr_ratio

    print(f"计划步数: {total_steps:,}（warmup {warmup}）")
    best_val = None
    t0 = time.time()
    running, running_n = 0.0, 0

    # ================= 训练循环 =================
    def train_pretrain():
        nonlocal best_val, running, running_n
        for step in range(start_step, total_steps):
            lr = lr_at(step, total_steps, warmup, sc.lr, min_ratio)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(sc.grad_accum):
                x, y = stream.get_batch(sc.batch_size, generator=gen)
                with autocast:
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                    loss = loss / sc.grad_accum
                scaler.scale(loss).backward()
                step_loss += float(loss)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += step_loss
            running_n += 1
            yield step, step_loss, lr

    def train_sft():
        nonlocal best_val, running, running_n
        step = start_step
        for epoch in range(sc.epochs):
            if step >= total_steps:
                break
            for x, y in stream.iter_epoch(sc.batch_size, generator=gen):
                if step >= total_steps:
                    break
                lr = lr_at(step, total_steps, warmup, sc.lr, min_ratio)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with autocast:
                    logits = model(x)
                    # 只有 assistant 位置参与 loss（其余标签是 -100）
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-100,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                running += float(loss)
                running_n += 1
                yield step, float(loss), lr
                step += 1

    iterator = train_sft() if is_sft else train_pretrain()
    ckpt_every = (args.ckpt_every if args.ckpt_every is not None
                  else cfg.train.ckpt_every)
    eval_every = (args.eval_every if args.eval_every is not None
                  else cfg.train.eval_every)
    log_every = cfg.train.log_every
    best_path = os.path.join(cfg.paths.ckpt_dir, f"{args.stage}_best.pt")
    last_step = start_step
    logged = False

    for step, loss_v, lr in iterator:
        done = step + 1
        last_step = done
        logged = True

        if done % log_every == 0 and running_n:
            avg = running / running_n
            print(f"step {done:>6}/{total_steps} | loss {avg:.4f} | "
                  f"ppl {math.exp(min(20, avg)):.1f} | lr {lr:.2e} | "
                  f"{time.time()-t0:.0f}s")
            running, running_n = 0.0, 0
            t0 = time.time()

        # ---- 验证集评估 ----
        if eval_every and done % eval_every == 0 and val_stream is not None:
            if is_sft:
                vb = val_stream.iter_epoch(32, generator=gen, shuffle=False)
            else:
                vb = (val_stream.get_batch(32, generator=gen) for _ in range(20))
            v = eval_loss(model, vb, device, use_amp, amp_dtype=amp_dtype)
            if v is not None:
                improved = best_val is None or v < best_val
                print(f"  [eval] step {done} | val loss {v:.4f}"
                      f"{'  (best)' if improved else ''}")
                if improved:
                    best_val = v
                    save_checkpoint(best_path, model, optimizer, scaler, done,
                                    cfg, {"val_loss": v, "stage": args.stage})

        # ---- 定期保存 ----
        if ckpt_every and done % ckpt_every == 0:
            save_checkpoint(
                os.path.join(cfg.paths.ckpt_dir, f"{args.stage}_last.pt"),
                model, optimizer, scaler, done, cfg,
                {"val_loss": best_val, "stage": args.stage},
            )
            print(f"  -> 已保存 {args.stage}_last.pt (step {done})")

    # ---- 收尾保存 ----
    final_step = last_step if logged else start_step
    save_checkpoint(
        os.path.join(cfg.paths.ckpt_dir, f"{args.stage}_last.pt"),
        model, optimizer, scaler, final_step, cfg,
        {"val_loss": best_val, "stage": args.stage},
    )
    print(f"\n最终 checkpoint: {args.stage}_last.pt (step {final_step})")
    if best_val is not None:
        print(f"最佳验证 loss: {best_val:.4f} -> {best_path}")
    print("训练完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
