"""训练入口：预训练 + SFT 两个阶段。

预训练（从零）:
    python src/train.py --stage pretrain
    python src/train.py --stage pretrain --max-steps 200      # 冒烟

SFT（从预训练 checkpoint 出发，只对 assistant 回复算 loss）:
    python src/train.py --stage sft --init-from <pretrain ckpt>

checkpoint 和日志写回 DREAMY_OUT_DIR。Colab 断线后加 --resume auto 续训。

模型尺寸覆盖（默认 = src/config.py 里的 110M，所有覆盖项都有默认值）:
    python src/train.py --stage pretrain --dim 896 --depth 12 \
        --heads 14 --head-dim 64 --seq-len 1024
生效后的架构会写进 <out_dir>/model_config.json，并随 checkpoint 一起保存；
之后加 --resume auto 续训会自动按 checkpoint 里的架构重建，不用再传尺寸参数
（传了但是和 checkpoint 不一致会直接报错，避免权重错位加载）。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import fields

# ---- 必须在 import torch 之前设置 ----
# expandable_segments:True 让 CUDA 分配器按需扩展显存段，缓解 10 小时长跑里
# 显存碎片累积导致的 OOM。只影响显存分配策略，CPU 上无副作用。
# 两个变量名都设成同一个值：旧版 torch 读 PYTORCH_CUDA_ALLOC_CONF，
# 新版读 PYTORCH_ALLOC_CONF，值相同不会有冲突；用户自己设过就不覆盖。
_ALLOC_CONF = "expandable_segments:True"
if ("PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        and "PYTORCH_ALLOC_CONF" not in os.environ):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _ALLOC_CONF
    os.environ["PYTORCH_ALLOC_CONF"] = _ALLOC_CONF

# 允许 `python src/train.py` 直接运行时正确 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import Config, ModelConfig
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


# ---------------------------------------------------------------- 模型架构
def model_config_to_dict(mc: ModelConfig) -> dict:
    """ModelConfig -> 可 JSON 序列化的 dict（只含 dataclass 字段）。"""
    return dict(mc.__dict__)


def model_config_from_dict(d: dict | None) -> ModelConfig | None:
    """dict -> ModelConfig。

    忽略未知字段、缺失字段用默认值，这样旧版本 checkpoint 里
    没有新开关（use_flash / qk_norm 等）也能正确重建。
    """
    if not d:
        return None
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in d.items() if k in known})


def describe_model_config(mc: ModelConfig) -> str:
    return (f"dim={mc.dim} depth={mc.depth} heads={mc.heads} "
            f"head_dim={mc.head_dim} seq_len={mc.max_seq_len} "
            f"vocab={mc.num_tokens}")


# 命令行参数名 -> ModelConfig 字段名（--head-dim 的 dest 就是 head_dim）
MODEL_OVERRIDE_MAP = {
    "dim": "dim",
    "depth": "depth",
    "heads": "heads",
    "head_dim": "head_dim",
    "seq_len": "max_seq_len",
}


def apply_model_overrides(mc: ModelConfig, args) -> None:
    """把命令行上的尺寸覆盖应用到模型配置（None = 保持原值）。"""
    for arg_name, field_name in MODEL_OVERRIDE_MAP.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(mc, field_name, value)


def save_model_config(path: str, mc: ModelConfig, extra: dict | None = None) -> None:
    """把生效的模型架构写到 <out_dir>/model_config.json。

    和 checkpoint 里的 model_config 字段一致，额外信息放在 "_meta" 下。
    改尺寸重训时先看这个文件就知道上次用的到底是什么架构。
    """
    payload = model_config_to_dict(mc)
    if extra:
        payload["_meta"] = extra
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model_config_file(path: str) -> ModelConfig | None:
    """读取 model_config.json（读不到/损坏返回 None，不阻塞训练）。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return model_config_from_dict(json.load(f))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] 读取 {path} 失败（{type(e).__name__}），忽略")
        return None


def report_attention_flash(model) -> None:
    """确认 Attend 层真的开了融合注意力。

    v1.0 的坑：attn_flash 没开时会物化 [B, H, S, S] 注意力矩阵，
    慢 2-3 倍还爆显存。这里只做检查+提示，不改配置。
    """
    flags = [bool(getattr(m, "flash", False))
             for m in model.modules() if m.__class__.__name__ == "Attend"]
    if not flags:
        return
    n_on = sum(flags)
    note = "" if n_on == len(flags) else "  [!] 未全部开启，会明显变慢"
    print(f"融合注意力(flash SDPA): {n_on}/{len(flags)} 层开启{note}")


# ---------------------------------------------------------------- 吞吐计时
class StepTimer:
    """按窗口统计每步耗时，用来算 step_time / tok/s。

    CUDA 上用 torch.cuda.Event：每步只 record 不 sync，打日志时同步一次，
    读到的是 GPU 真实执行时间（不受 CPU 侧排队影响）。
    CPU 上退回 perf_counter，统计口径一致。
    """

    def __init__(self, device):
        self._cuda = device.type == "cuda"
        self._n = 0
        self._t0 = time.perf_counter()
        if self._cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()

    def __len__(self) -> int:
        return self._n

    def tick(self) -> None:
        """每完成一个 optimizer step 调一次。"""
        self._n += 1

    def elapsed(self) -> float:
        """本窗口耗时（秒）。CUDA 上会 synchronize，只在打日志时调用。"""
        if self._cuda:
            self._end.record()
            self._end.synchronize()
            return self._start.elapsed_time(self._end) / 1000.0
        return time.perf_counter() - self._t0

    def reset(self) -> None:
        self._n = 0
        self._t0 = time.perf_counter()
        if self._cuda:
            self._start.record()


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


def load_checkpoint_payload(path: str) -> dict:
    """把 checkpoint 读进内存（取架构 + 权重 + 优化器状态只用读一次盘）。

    续训时要在 build_model 之前知道 architecture，所以提前读出来复用。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                       # noqa: BLE001
        raise SystemExit(
            f"读取 checkpoint 失败: {path}\n"
            f"{type(e).__name__}: {str(e)[:300]}\n"
            f"文件可能损坏或不是本项目的 checkpoint。"
        ) from e


def restore_checkpoint(ckpt: dict, model, optimizer=None, scaler=None) -> int:
    """把已读入内存的 checkpoint 恢复到 model / optimizer / scaler。"""
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f"已恢复 checkpoint: step {ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0))


def load_weights_only(ckpt: dict, model) -> None:
    """只加载模型权重（SFT 从预训练出发时用，不继承优化器状态）。"""
    model.load_state_dict(ckpt["model"])
    print(f"已加载预训练权重 (step {ckpt.get('step', '?')})")


# ---------------------------------------------------------------- 公共
def enable_cuda_perf() -> None:
    """打开 CUDA 侧性能相关开关（全部带 hasattr 守卫，兼容不同 torch 版本）。"""
    backends = torch.backends.cuda
    # 强制 SDPA 优先走 flash 实现（attn_flash 已在模型里打开，这里再兜一层）
    if hasattr(backends, "enable_flash_sdp"):
        backends.enable_flash_sdp(True)
    # 不满足 flash 条件时退到 mem-efficient 实现，避免掉回会物化注意力矩阵的 math 实现
    if hasattr(backends, "enable_mem_efficient_sdp"):
        backends.enable_mem_efficient_sdp(True)
    # bf16 矩阵乘允许降精度累加：5090 Tensor Core 上更快，收敛影响可忽略
    if hasattr(backends.matmul, "allow_bf16_reduced_precision_reduction"):
        backends.matmul.allow_bf16_reduced_precision_reduction = True
    if hasattr(backends.matmul, "allow_fp16_reduced_precision_reduction"):
        backends.matmul.allow_fp16_reduced_precision_reduction = True


def setup(args, cfg):
    device = pick_device()
    print("=" * 66)
    print(describe_device())
    if device.type == "cuda":
        enable_cuda_perf()
        alloc = (os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                 or os.environ.get("PYTORCH_ALLOC_CONF") or "默认")
        print(f"CUDA 调优: flash_sdp / mem_efficient / reduced_precision_reduction；"
              f"alloc_conf={alloc}")
    print(cfg.paths.describe())
    print("=" * 66)

    if device.type == "cuda":
        from model import cuda_runnable
        ok, err = cuda_runnable(0)
        if not ok:
            raise SystemExit(
                f"GPU 无法运行当前 torch 构建：{err}\n"
                f"请安装与该 GPU 匹配的 CUDA 版本 torch，例如 4090/4090 系：\n"
                f"  pip install torch==2.14.1 "
                f"--index-url https://download.pytorch.org/whl/cu126"
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


def is_cuda_oom(err: BaseException) -> bool:
    """判断是不是显存不足（不同 torch 版本异常类型/文案不完全一致）。"""
    if isinstance(err, torch.cuda.OutOfMemoryError):     # CPU 版 torch 也有这个类
        return True
    return "out of memory" in str(err).lower()


@torch.no_grad()
def eval_loss(model, batches, device, use_amp: bool, max_batches: int = 64,
              amp_dtype=None):
    """评估集平均 loss（分批评估，显存友好）。

    v1.0 的坑：这里固定 batch 32 + 20 批，4090 上 [B, S, V] logits 直接 OOM，
    只能靠 --eval-every 0 把评估整个关掉。现在 batch / 批数都由
    --eval-batch-size / --eval-batches 控制（默认 8 / 64），
    外层的 OOM 保护保证评估失败也绝不中断训练。
    """
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    total, n = 0.0, 0
    try:
        for i, (x, y) in enumerate(batches):
            if i >= max_batches:
                break
            with torch.autocast(device_type=device_type,
                                dtype=amp_dtype or torch.float16,
                                enabled=use_amp):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    ignore_index=-100,
                )
            del logits
            total += float(loss)
            n += 1
    finally:
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
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="评估 micro batch（默认 8；eval 曾因 batch 32 OOM，别调大）")
    p.add_argument("--eval-batches", type=int, default=64,
                   help="每次评估最多跑多少批（默认 64）")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="每多少步保存 checkpoint（0 = 只在结束保存）")
    # ---- 模型尺寸覆盖（默认全部沿用 src/config.py，即 110M v1.1）----
    p.add_argument("--dim", type=int, default=None,
                   help="hidden size（默认 768；145M 档用 896）")
    p.add_argument("--depth", type=int, default=None, help="层数（默认 12）")
    p.add_argument("--heads", type=int, default=None,
                   help="注意力头数（默认 12；145M 档用 14）")
    p.add_argument("--head-dim", type=int, default=None,
                   help="每个头的维度（默认 64）")
    p.add_argument("--seq-len", type=int, default=None,
                   help="训练上下文长度（默认 1024；CPU 冒烟可用 64）")
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

    # ---- 模型架构：默认值 <- checkpoint / model_config.json <- 命令行覆盖 ----
    # 先解析 resume 路径并把 checkpoint 读进内存：既用来重建架构，
    # 后面恢复权重/优化器时直接复用，避免 1GB+ 的 ckpt 读两遍。
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(cfg.paths.ckpt_dir, f"{args.stage}_last.pt")
    resume_ckpt = None
    if resume_path and os.path.isfile(resume_path):
        resume_ckpt = load_checkpoint_payload(resume_path)
    init_ckpt = None
    if is_sft and args.init_from and os.path.isfile(args.init_from):
        init_ckpt = load_checkpoint_payload(args.init_from)

    config_json_path = os.path.join(cfg.paths.out_dir, "model_config.json")
    arch_source = None
    for ckpt, label in ((resume_ckpt, "续训 checkpoint"),
                        (init_ckpt, "init-from checkpoint")):
        restored = (model_config_from_dict(ckpt.get("model_config"))
                    if ckpt else None)
        if restored is not None:
            cfg.model = restored
            arch_source = label
            break
    if arch_source is None:
        file_cfg = load_model_config_file(config_json_path)
        if file_cfg is not None:
            cfg.model = file_cfg
            arch_source = config_json_path
    if arch_source:
        print(f"按 {arch_source} 重建模型架构: {describe_model_config(cfg.model)}")
    apply_model_overrides(cfg.model, args)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = setup(args, cfg)

    # checkpoint 里的架构必须和最终配置完全一致，否则权重会错位加载
    for ckpt, label in ((resume_ckpt, "续训 checkpoint"),
                        (init_ckpt, "init-from checkpoint")):
        want = (model_config_from_dict(ckpt.get("model_config"))
                if ckpt else None)
        if want is None or want == cfg.model:
            continue
        diffs = [f"    {f.name}: checkpoint={getattr(want, f.name)} "
                 f"当前={getattr(cfg.model, f.name)}"
                 for f in fields(ModelConfig)
                 if getattr(want, f.name) != getattr(cfg.model, f.name)]
        raise SystemExit(
            f"当前模型架构与{label}不一致，无法加载权重：\n"
            + "\n".join(diffs)
            + "\n续训会自动按 checkpoint 里的架构重建，不要手动改尺寸；"
              "要换模型请换一个 out-dir 从头训练。"
        )

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
    print(f"\n模型架构: {describe_model_config(cfg.model)}")
    model = build_model(cfg.model).to(device)
    n_params = count_params(model)
    print(f"模型参数量: {n_params/1e6:.2f}M")
    report_attention_flash(model)
    # 生效配置落盘：崩溃重启后即使忘了传尺寸参数，也能按这个文件重建
    save_model_config(
        config_json_path, cfg.model,
        {"n_params": n_params, "stage": args.stage,
         "torch": torch.__version__,
         "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    print(f"模型配置已写入: {config_json_path}")

    # ---- 优化器 ----
    adamw_kwargs = {
        "lr": sc.lr,
        "weight_decay": sc.weight_decay,
        "betas": (cfg.train.beta1, cfg.train.beta2),
    }
    optimizer = None
    if device.type == "cuda":
        # fused AdamW 把整个 step 融合成少量 kernel，省掉大量小 launch 开销。
        # 老版本 torch / 非 CUDA 平台没有这个参数，必须守卫 + 回退。
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True,
                                          **adamw_kwargs)
            print("优化器: AdamW(fused=True)")
        except (TypeError, RuntimeError) as e:                   # noqa: PERF203
            print(f"[!] fused AdamW 不可用（{type(e).__name__}: "
                  f"{str(e)[:120]}），回退普通实现")
            optimizer = None
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        if device.type != "cuda":
            print("优化器: AdamW（CPU）")
    # ---- 精度策略 ----
    # 4090 (sm_89) / 5090 (sm_120) 原生支持 bf16：不需要 GradScaler，数值更稳。
    # T4 (sm_75) 不支持 bf16，自动回退到 fp16 + GradScaler。
    use_amp = (device.type == "cuda") and not args.no_amp
    bf16_ok = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = use_amp and not bf16_ok
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype,
                              enabled=use_amp)

    # 每个 optimizer step 实际喂进去的 token 数（算 tok/s 用）
    if is_sft:
        sft_seq_len = int(getattr(stream, "seq_len", cfg.model.max_seq_len))
        tokens_per_step = sc.batch_size * sft_seq_len
        print(f"每步 token: {sc.batch_size} x {sft_seq_len} = "
              f"{tokens_per_step:,}（SFT 不做梯度累积）")
    else:
        tokens_per_step = sc.batch_size * sc.grad_accum * cfg.model.max_seq_len
        print(f"有效 batch: {sc.batch_size} x {cfg.model.max_seq_len} x "
              f"{sc.grad_accum} = {tokens_per_step:,} token/step")
    print(f"混合精度: "
          f"{('bf16' if bf16_ok else 'fp16 + GradScaler') if use_amp else 'off'}\n")

    gen = torch.Generator().manual_seed(cfg.seed)

    # ---- 起始权重（ckpt 前面已读进内存，这里直接复用，不再读一遍盘）----
    start_step = 0
    if init_ckpt is not None:
        load_weights_only(init_ckpt, model)
        print(f"  <- {args.init_from}")
    if resume_ckpt is not None:
        start_step = restore_checkpoint(resume_ckpt, model, optimizer, scaler)
        print(f"  <- {resume_path}")
    elif resume_path is not None:
        if args.resume == "auto":
            print(f"未找到 checkpoint: {resume_path}，从头开始")
        else:
            # 显式指定的 resume 路径不存在 = 参数写错，不能悄悄从头训 10 小时
            raise SystemExit(f"找不到 --resume 指定的 checkpoint: {resume_path}")
    # 权重/优化器状态已进显存，释放内存里的整份 ckpt（几百 MB ~ 1.5GB）
    del resume_ckpt, init_ckpt
    gc.collect()

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

    plan_tokens = total_steps * tokens_per_step
    rest_tokens = max(0, total_steps - start_step) * tokens_per_step
    print(f"计划步数: {total_steps:,}（warmup {warmup}）")
    print(f"计划 token: {plan_tokens:,}（D/N ≈ {plan_tokens / n_params:.1f}）"
          + (f"；本次续训还需 {rest_tokens:,}" if start_step else ""))
    timer = StepTimer(device)
    run_t0 = time.time()
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
                step_loss += float(loss.detach())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            timer.tick()

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
                timer.tick()

                running += float(loss.detach())
                running_n += 1
                yield step, float(loss.detach()), lr
                step += 1

    iterator = train_sft() if is_sft else train_pretrain()
    ckpt_every = (args.ckpt_every if args.ckpt_every is not None
                  else cfg.train.ckpt_every)
    eval_every = (args.eval_every if args.eval_every is not None
                  else cfg.train.eval_every)
    eval_batch_size = max(1, args.eval_batch_size)
    eval_batches = max(1, args.eval_batches)
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
            window_s = timer.elapsed()      # CUDA 上会 synchronize 一次
            window_steps = max(1, len(timer))
            step_ms = window_s * 1000.0 / window_steps
            tok_s = (window_steps * tokens_per_step / window_s
                     if window_s > 0 else 0.0)
            print(f"step {done:>6}/{total_steps} | loss {avg:.4f} | "
                  f"ppl {math.exp(min(20, avg)):.1f} | lr {lr:.2e} | "
                  f"{step_ms:.0f}ms/step | {tok_s:,.0f} tok/s | "
                  f"{time.time()-t0:.0f}s")
            running, running_n = 0.0, 0
            timer.reset()
            t0 = time.time()

        # ---- 验证集评估（分批，且 OOM 只跳过本次 eval，绝不中断训练）----
        if eval_every and done % eval_every == 0 and val_stream is not None:
            try:
                if is_sft:
                    vb = val_stream.iter_epoch(eval_batch_size, generator=gen,
                                               shuffle=False)
                else:
                    vb = (val_stream.get_batch(eval_batch_size, generator=gen)
                          for _ in range(eval_batches))
                v = eval_loss(model, vb, device, use_amp,
                              max_batches=eval_batches, amp_dtype=amp_dtype)
            except RuntimeError as e:
                if not is_cuda_oom(e):
                    raise
                v = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  [eval] 显存不足，跳过本次评估（eval batch "
                      f"{eval_batch_size}）。可加 --eval-batch-size "
                      f"{max(1, eval_batch_size // 2)} 再试")
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
    ran_steps = max(0, final_step - start_step)
    elapsed_s = time.time() - run_t0
    if ran_steps and elapsed_s > 0:
        print(f"本轮吞吐: {ran_steps * tokens_per_step / elapsed_s:,.0f} tok/s"
              f"（{ran_steps:,} 步 / {elapsed_s/3600:.2f} 小时 / "
              f"{ran_steps * tokens_per_step:,} token）")
    print("训练完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
