"""模型构建：基于 x-transformers。

单独拆出来，方便冒烟测试不依赖数据、也方便以后换配置。
"""
from __future__ import annotations

import torch
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

from config import ModelConfig


def build_model(cfg: ModelConfig) -> TransformerWrapper:
    """构建 Dreamy-36M。

    对应 spec v1.1 的架构表：
        Decoder-only / RoPE / RMSNorm / Pre-Norm / SwiGLU / Weight Tying

    几个必须显式打开的开关（默认值会让参数量对不上）：
        ff_glu=True       -> SwiGLU，不开只有 29.4M（开了 35.69M）
        tie_embedding=True-> 输出头和 embedding 共享，省 8.39M
        use_abs_pos_emb=False + rotary_pos_emb=True  用 RoPE 不用绝对位置编码
    """
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
