"""Dreamy 配置：路径与超参。

所有路径都可以用环境变量覆盖，这样同一套代码在本地（Windows）和
Colab（Linux + Drive）上都能跑，不需要改代码。

环境变量
    DREAMY_DATA_DIR   训练数据目录（默认 Colab Drive 路径）
    DREAMY_OUT_DIR    checkpoint / 日志输出目录
    DREAMY_SEED       随机种子
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath

# Colab 上挂载后的默认位置；本地跑时用环境变量指到本地目录。
# 用 POSIX 风格拼接，保证打印出来在 Linux/Windows 上都是同一个样子
# （Windows 也接受正斜杠）。
DEFAULT_DATA_DIR = "/content/drive/MyDrive/dreamy/data"
DEFAULT_OUT_DIR = "/content/drive/MyDrive/dreamy/runs"


def _join(base: str, *parts: str) -> str:
    """基于 POSIX 语义的路径拼接，跨平台一致。"""
    return str(PurePosixPath(base).joinpath(*parts))


@dataclass
class Paths:
    data_dir: str = field(default_factory=lambda: os.environ.get(
        "DREAMY_DATA_DIR", DEFAULT_DATA_DIR))
    out_dir: str = field(default_factory=lambda: os.environ.get(
        "DREAMY_OUT_DIR", DEFAULT_OUT_DIR))

    @property
    def ckpt_dir(self) -> str:
        return _join(self.out_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        return _join(self.out_dir, "logs")

    def ensure(self) -> "Paths":
        """创建输出目录（数据目录必须已存在，不自动创建以免掩盖路径错误）。"""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        return self

    def describe(self) -> str:
        return (
            f"data_dir = {self.data_dir}\n"
            f"out_dir  = {self.out_dir}\n"
            f"ckpt_dir = {self.ckpt_dir}"
        )


@dataclass
class ModelConfig:
    num_tokens: int = 256       # 词表大小
    max_seq_len: int = 256      # 最大序列长度
    dim: int = 256              # 隐藏维度
    depth: int = 6              # 层数
    heads: int = 8              # 注意力头数


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 1000
    warmup_steps: int = 100
    log_every: int = 20
    ckpt_every: int = 200
    seed: int = 42


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def seed(self) -> int:
        return int(os.environ.get("DREAMY_SEED", self.train.seed))
