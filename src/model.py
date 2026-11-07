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


def describe_device() -> str:
    """打印设备信息，重点确认 T4(sm_75) 是否被当前 torch 构建支持。"""
    lines = [f"torch {torch.__version__}", f"cuda build: {torch.version.cuda}"]
    if not torch.cuda.is_available():
        lines.append("cuda: NOT available (running on CPU)")
        return "\n".join(lines)

    arch_list = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = f"sm_{cap[0]}{cap[1]}"
        supported = arch in arch_list
        lines.append(
            f"gpu {i}: {torch.cuda.get_device_name(i)} | {arch} | "
            f"kernel {'OK' if supported else 'MISSING'}"
        )
    if "sm_75" not in arch_list and any(
        torch.cuda.get_device_capability(i) == (7, 5)
        for i in range(torch.cuda.device_count())
    ):
        lines.append(
            "WARNING: this torch build has no sm_75 kernel; "
            "T4 will not work. Use the cu126 build (torch 2.14.x)."
        )
    return "\n".join(lines)
