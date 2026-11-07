"""模型构建：基于 x-transformers。

单独拆出来，方便冒烟测试不依赖数据、也方便以后换配置。
"""
from __future__ import annotations

import torch
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

from config import ModelConfig


def align_heads(cfg: ModelConfig) -> ModelConfig:
    """把 heads 和 dim / head_dim 对齐。

    `--dim 832 --depth 12` 这种写法很容易漏掉 `--heads`：heads 还是 12、
    head_dim 64，12×64=768 ≠ 832。x-transformers 遇到 inner_dim != dim
    不会报错，而是默默多插一个投影层 —— 参数量、速度、显存全都和预期不一样。
    这里直接按 dim/head_dim 纠正，并明确打出来。
    """
    if cfg.heads * cfg.head_dim == cfg.dim:
        return cfg
    if cfg.dim % cfg.head_dim == 0:
        new_heads = cfg.dim // cfg.head_dim
        print(f"[!] heads({cfg.heads})×head_dim({cfg.head_dim})="
              f"{cfg.heads * cfg.head_dim} 与 dim({cfg.dim}) 不匹配，"
              f"自动改成 heads={new_heads}（即 dim/head_dim）")
        cfg.heads = new_heads
        return cfg
    raise ValueError(
        f"dim={cfg.dim} 不能被 head_dim={cfg.head_dim} 整除："
        f"请显式给出匹配的 --heads / --head-dim"
        f"（当前 heads={cfg.heads}, head_dim={cfg.head_dim}）")


def build_model(cfg: ModelConfig) -> TransformerWrapper:
    """构建 Dreamy。

    架构：Decoder-only / RoPE / RMSNorm / Pre-Norm / SwiGLU / 权重共享

    必须显式打开的开关（默认值会让参数量或速度不对）：
        ff_glu=True         -> SwiGLU。不开只有 99.2M（开了 110.18M）
        tie_embedding=True  -> 输出头和 embedding 共享，省 25.2M
        use_abs_pos_emb=False + rotary_pos_emb=True  用 RoPE

    v1.1 新增（关键）：
        attn_flash=True     -> 走 F.scaled_dot_product_attention 融合实现。
                               不开的话注意力矩阵会被完整物化，
                               显存暴涨且 MFU 只有 15%。
        attn_qk_norm=True   -> QK-Norm，训练更稳，可用更高学习率
        zero_init_branch_output=True -> 残差分支零初始化
    """
    cfg = align_heads(cfg)
    return TransformerWrapper(
        num_tokens=cfg.num_tokens,
        max_seq_len=cfg.max_seq_len,
        use_abs_pos_emb=False,       # RoPE 取代绝对位置编码
        tie_embedding=True,          # 权重共享
        emb_dropout=cfg.dropout,
        attn_layers=Decoder(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            attn_dim_head=cfg.head_dim,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            pre_norm=True,
            attn_dropout=cfg.dropout,
            ff_dropout=cfg.dropout,
            ff_mult=cfg.ffn_mult,
            ff_glu=cfg.ffn_glu,
            # ---- v1.1 ----
            attn_flash=getattr(cfg, "use_flash", True),
            attn_qk_norm=getattr(cfg, "qk_norm", True),
            zero_init_branch_output=getattr(cfg, "zero_init_branch", True),
        ),
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def build_trainable(cfg: ModelConfig):
    """包一层 AutoregressiveWrapper，提供生成能力。"""
    model = build_model(cfg)
    return model, AutoregressiveWrapper(model)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_sm(entry: str) -> tuple[int, int] | None:
    """把 'sm_86' / 'sm_90a' / 'compute_75' 解析成 (major, minor)。"""
    for prefix in ("sm_", "compute_"):
        if entry.startswith(prefix):
            rest = entry[len(prefix):].rstrip("abcdef")
            try:
                return int(rest[0]), int(rest[1:])
            except (ValueError, IndexError):
                return None
    return None


def arch_compatible(cap: tuple[int, int], arch_list: list[str]) -> bool:
    """当前 GPU 能否跑这个 torch 构建。

    CUDA 的二进制兼容规则：为 X.y 编译的 kernel 可以跑在 X.z (z >= y) 上。
    所以 sm_89 的 4090 直接跑 sm_86 的 kernel 就行，不需要专门的 sm_89 构建。
    （之前这里用 `arch in arch_list` 精确匹配，把 4090 误判成不支持 —— 是个 bug。）

    如果构建里带 PTX（'compute_XY'），还能通过 JIT 编译支持更高版本。
    """
    major, minor = cap
    for entry in arch_list:
        parsed = _parse_sm(entry)
        if parsed is None:
            continue
        m, n = parsed
        if m == major and n <= minor:      # 二进制兼容
            return True
        if entry.startswith("compute_") and (m, n) >= (major, minor):
            return True                    # PTX 可 JIT
    return False


def cuda_runnable(device_index: int = 0) -> tuple[bool, str]:
    """最终判据：真的在 GPU 上跑一次 matmul。

    架构列表只是参考，能不能跑要实测。这是唯一不会误判的检查。
    """
    try:
        x = torch.randn(256, 256, device=f"cuda:{device_index}")
        y = torch.mm(x, x)
        torch.cuda.synchronize(device_index)
        if not torch.isfinite(y).all():
            return False, "matmul 结果异常"
        return True, ""
    except Exception as e:                                  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def describe_device() -> str:
    """打印设备信息，并确认当前 GPU 真的能跑 kernel。"""
    lines = [f"torch {torch.__version__}", f"cuda build: {torch.version.cuda}"]
    if not torch.cuda.is_available():
        lines.append("cuda: NOT available (running on CPU)")
        return "\n".join(lines)

    arch_list = torch.cuda.get_arch_list()
    lines.append(f"built archs: {', '.join(arch_list) if arch_list else '-'}")

    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = f"sm_{cap[0]}{cap[1]}"
        name = torch.cuda.get_device_name(i)
        compatible = arch_compatible(cap, arch_list)
        ok, err = cuda_runnable(i)
        if ok:
            note = "kernel OK" if compatible else "kernel OK（经兼容/PTX 运行）"
        else:
            note = f"kernel FAILED: {err}"
        lines.append(f"gpu {i}: {name} | {arch} | {note}")
        if not ok:
            lines.append(
                "  GPU 无法运行当前 torch。若是 4090(sm_89)/T4(sm_75)，"
                "请装对应 CUDA 版本的 torch，例如：\n"
                "    pip install torch==2.14.1 "
                "--index-url https://download.pytorch.org/whl/cu126"
            )
    return "\n".join(lines)
