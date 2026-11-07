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
    """Dreamy-110M v1.1（5090 32G / 4090 24G 通用）。

    v1.1 改动 —— 参数量不变（110.18M），但训练效率和稳定性大幅提升：

    1. use_flash=True（最大收益）
       v1.0 默认 flash=False，x-transformers 走手写注意力，
       会把 [batch, heads, seq, seq] 的注意力矩阵完整物化：
         - 显存暴涨（4090 上 micro batch 8 就 OOM）
         - MFU 只有 15%（实测 32.7k tok/s）
       打开后走 F.scaled_dot_product_attention(is_causal=True)，
       即 PyTorch 原生融合注意力，不需要额外装 flash-attn 包。

    2. qk_norm=True
       attention 的 q/k 做 RMSNorm，logits 不发散，
       可以用更高学习率，收敛更快更稳。

    3. zero_init_branch=True
       残差分支输出初始化为 0，起步更像恒等映射。

    4. dropout 0.05 -> 0
       v1.0 只训了 0.62B token（D/N=5.7），严重欠训练，
       轮不到过拟合，dropout 只会拖慢收敛。

    尺寸说明（D/N = 训练 token / 参数量，Chinchilla 最优约 20）：
      110M 需约 2.2B token 才训练充分。v1.0 只喂了 0.62B，
      这是质量受限的主因，不是架构问题 —— 不加数据就加大参数只会更欠训练。

    要换更大尺寸（32G 显存 + 数据足够时）：
      dim=896,  heads=14 -> 145M（需约 2.9B token）
      dim=1024, heads=16 -> 185M（需约 3.7B token）
    """
    num_tokens: int = 32768     # 词表：中文压缩率比 16k 好约 15%
    max_seq_len: int = 1024     # 可覆盖 8-12 轮对话
    dim: int = 768              # Hidden Size
    depth: int = 12             # Layers
    heads: int = 12             # Attention Heads
    head_dim: int = 64          # Head Dimension
    ffn_mult: float = 8 / 3     # SwiGLU 标准比例，hidden = 2048
    ffn_glu: bool = True        # SwiGLU
    dropout: float = 0.0        # v1.1 关掉（欠训练阶段）

    # ---- v1.1 优化开关 ----
    use_flash: bool = True          # 融合注意力
    qk_norm: bool = True            # QK-Norm
    zero_init_branch: bool = True   # 残差分支零初始化


@dataclass
class TrainConfig:
    # 有效 batch = micro_batch * seq_len * grad_accum
    # 开了融合注意力后显存大幅下降（不再物化注意力矩阵）：
    #   4090 24G : batch 8  起步，可试 12-16
    #   5090 32G : batch 16 起步，可试 24-32
    # 8 * 1024 * 2 = 16,384 token/step（与 v1.0 有效 batch 一致）
    batch_size: int = 8         # micro batch
    grad_accum: int = 2
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
