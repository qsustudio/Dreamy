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
# 默认用「项目内相对路径」，不再假设 Colab 的 /content/drive。
# 换机器（Colab / 本地 / 实验室 4090）都不用改代码。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "runs")


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
    """Dreamy-110M（RTX 4090 版本）。

    实测参数量 110M。8h 4090 预算下正好落在 Chinchilla 最优附近
    （N≈118M, D≈2.35B @ MFU 35%）。
    """
    num_tokens: int = 32768     # 词表：中文压缩率比 16k 好约 15%
    max_seq_len: int = 1024     # 可覆盖 8-12 轮对话
    dim: int = 768              # Hidden Size
    depth: int = 12             # Layers
    heads: int = 12             # Attention Heads
    head_dim: int = 64          # Head Dimension
    ffn_mult: float = 8 / 3     # SwiGLU 标准比例，hidden = 2048
    ffn_glu: bool = True        # SwiGLU
    dropout: float = 0.05

    # 为什么从 36M 提到 110M：
    #   4090 算力是 T4 的 2.54x、带宽 3.15x，8h 能跑 2.15B token。
    #   Chinchilla 最优 N = sqrt(C/120) ≈ 118M，110M 正好贴合。
    #   原来的 36M/400M 方案只有 D/N=10.6，属于严重欠训练。


@dataclass
class TrainConfig:
    # 有效 batch = micro_batch * seq_len * grad_accum
    # 24GB 显存下 110M 模型固定开销只有 1.3GB，micro batch 可以开到 16
    # 16 * 1024 * 1 = 16,384 token/step
    batch_size: int = 16        # micro batch
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 1000
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1   # 最低 LR = lr * 0.1
    log_every: int = 20
    ckpt_every: int = 2000
    eval_every: int = 500
    seed: int = 42

    # 4090 原生支持 bf16，不需要 GradScaler，比 fp16 更稳
    amp_dtype: str = "bf16"

    @property
    def effective_batch_tokens(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass
class SFTConfig:
    """SFT 阶段超参，对应 spec §5.2。"""
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)

    @property
    def seed(self) -> int:
        return int(os.environ.get("DREAMY_SEED", self.train.seed))
