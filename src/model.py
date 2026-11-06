"""模型构建：基于 x-transformers。

单独拆出来，方便冒烟测试不依赖数据、也方便以后换配置。
"""
from __future__ import annotations

import torch
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

from config import ModelConfig


def build_model(cfg: ModelConfig) -> TransformerWrapper:
    """构建一个自回归 Transformer。"""
    return TransformerWrapper(
        num_tokens=cfg.num_tokens,
        max_seq_len=cfg.max_seq_len,
        attn_layers=Decoder(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
        ),
    )


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
