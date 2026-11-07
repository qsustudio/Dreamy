"""数据加载。

数据放在 Google Drive（本地开发时放本地目录），不进 git 仓库。
路径由环境变量 DREAMY_DATA_DIR 指定。

目前提供两种最常用的输入格式，按需选用：
  * load_token_bin(path, dtype="uint16")
        tokens.bin —— 一整块扁平的 token id，按序列长度切成样本
        适合 enwik8 / 字符级等「语料本身就是 token 流」的场景
  * load_npy(path)
        任意 .npy 数组，取最后一维当作 token

如果你的数据是别的格式（jsonl / parquet / 图文对等），在 load_dataset
里加一个分支即可。
"""
from __future__ import annotations

import os

import numpy as np
import torch


class TokenStream:
    """把一维 token 流按固定长度切成训练样本。"""

    def __init__(self, tokens: np.ndarray, seq_len: int, device="cpu"):
        if tokens.ndim != 1:
            tokens = tokens.reshape(-1)
        self.tokens = torch.from_numpy(np.asarray(tokens)).long()
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        # 每条样本是 seq_len + 1 个 token（输入 + 下一个 token 作为标签）
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        """随机取一个 batch，返回 (x, y)，y 是 x 左移一位。"""
        if len(self) == 0:
            raise ValueError(
                f"数据太短：只有 {len(self.tokens)} 个 token，"
                f"至少需要 {self.seq_len + 1} 个"
            )
        starts = torch.randint(
            0, len(self) - 1, (batch_size,), generator=generator
        ) * self.seq_len
        span = self.seq_len + 1
        x = torch.stack([self.tokens[s:s + span - 1] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + span] for s in starts])
        return x.to(self.device), y.to(self.device)


def load_token_bin(path: str, dtype: str = "uint16") -> np.ndarray:
    """读取扁平的 token id 文件（如 enwik8 的 .npy/.bin）。"""
    if path.endswith(".npy"):
        return np.load(path)
    return np.fromfile(path, dtype=np.dtype(dtype))


def resolve_data_file(data_dir: str, candidates: list[str]) -> str | None:
    """在数据目录里按候选文件名查找，找到返回路径。"""
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    return None


def load_dataset(data_dir: str, seq_len: int, device="cpu") -> TokenStream:
    """从数据目录加载训练数据。

    按以下顺序查找：
        1. tokens.bin / tokens.npy
        2. 目录下唯一的 .npy 文件
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请在 Drive 上准备好数据，或设置环境变量 DREAMY_DATA_DIR 指向正确位置。"
        )

    path = resolve_data_file(data_dir, ["tokens.bin", "tokens.npy"])

    if path is None:
        npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        if len(npy_files) == 1:
            path = os.path.join(data_dir, npy_files[0])
        elif not npy_files:
            raise FileNotFoundError(
                f"{data_dir} 里没找到 tokens.bin / tokens.npy / 任何 .npy 文件。\n"
                f"目录内容: {sorted(os.listdir(data_dir))[:20]}\n"
                f"如果你的数据格式不同，请修改 src/data.py 的 load_dataset()。"
            )
        else:
            raise FileNotFoundError(
                f"{data_dir} 里有多个 .npy 文件，无法确定用哪个: {npy_files[:10]}\n"
                f"请显式指定，或改名为 tokens.npy。"
            )

    tokens = load_token_bin(path)
    return TokenStream(tokens, seq_len, device=device)


# ---------------------------------------------------------------- SFT
IGNORE_INDEX = -100


class SFTStream:
    """SFT 数据：(tokens[N, L], labels[N, L])，其中 labels 只有 assistant
    位置是真实 token，其余为 -100。这样 loss 只算助手的回复（spec §4.2）。

    数据由 scripts/build_dataset.py --stage sft 产出。
    """

    def __init__(self, tokens: "np.ndarray", labels: "np.ndarray", device="cpu"):
        import torch

        if tokens.shape != labels.shape:
            raise ValueError(
                f"tokens {tokens.shape} 与 labels {labels.shape} 形状不一致"
            )
        self.tokens = torch.from_numpy(tokens).long()   # (N, L)
        self.labels = torch.from_numpy(labels).long()   # (N, L)，-100 = 不计算
        self.device = device
        self.seq_len = int(tokens.shape[1])

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def get_batch(self, batch_size: int, generator=None):
        """随机取一个 batch，返回 (x, y)。

        x = tokens[:, :-1]，y = labels[:, 1:]（左移一位做下一 token 预测）。
        """
        import torch

        n = len(self)
        if n == 0:
            raise ValueError("SFT 数据为空")
        idx = torch.randint(0, n, (min(batch_size, n),), generator=generator)
        x = self.tokens[idx, :-1]
        y = self.labels[idx, 1:]
        return x.to(self.device), y.to(self.device)

    def iter_epoch(self, batch_size: int, generator=None, shuffle: bool = True):
        """按 epoch 顺序遍历（SFT 需要走完所有样本）。"""
        import torch

        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield (self.tokens[idx, :-1].to(self.device),
                   self.labels[idx, 1:].to(self.device))


def load_sft_dataset(data_dir: str, device="cpu", split: str = "train") -> SFTStream:
    """加载 SFT 数据。split='train' 或 'val'。"""
    prefix = "sft" if split == "train" else "sft_val"
    tpath = os.path.join(data_dir, f"{prefix}_tokens.npy")
    lpath = os.path.join(data_dir, f"{prefix}_labels.npy")
    if not (os.path.isfile(tpath) and os.path.isfile(lpath)):
        raise FileNotFoundError(
            f"找不到 SFT 数据: {tpath}\n"
            f"请先运行:\n"
            f"  python scripts/build_dataset.py --stage sft --tokenize"
        )
    return SFTStream(np.load(tpath), np.load(lpath), device=device)


def load_val_dataset(data_dir: str, seq_len: int, device="cpu") -> TokenStream | None:
    """加载预训练验证集（按整条对话切分，不参与训练）。"""
    p = os.path.join(data_dir, "val_tokens.bin")
    if not os.path.isfile(p):
        return None
    return TokenStream(load_token_bin(p), seq_len, device=device)
