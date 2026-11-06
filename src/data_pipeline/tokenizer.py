"""训练 tokenizer 并打包成扁平的 token 流。

从零训练必须先有 tokenizer。中文用 SentencePiece BPE 最稳
（不需要预分词，直接吃原始汉字串）。

输出：
    tokenizer.model / tokenizer.vocab   SentencePiece 模型
    tokens.bin                          uint16 扁平 token 流，供 src/data.py 读取
    meta.json                           词表大小等，训练脚本据此设置 num_tokens

线性化格式（对话拼成一条流）::

    <user> 你好呀 <assistant> 你好，今天怎么样 <eos> <user> ... 
"""
from __future__ import annotations

import json
import os
import tempfile

from schema import Conversation

USER_TOKEN = "<user>"
ASST_TOKEN = "<assistant>"


def _linearize(conv: Conversation) -> str:
    """把一条数据拼成一行文本（仅用于喂 tokenizer 训练）。

    对话加角色标记；文档（通用语料）就是原文。
    """
    if conv.is_document:
        return conv.text
    parts = []
    for t in conv.turns:
        tag = USER_TOKEN if t.role == "user" else ASST_TOKEN
        parts.append(f"{tag} {t.text}")
    return " ".join(parts)


def _train_once(corpus_path: str, model_prefix: str, vocab_size: int,
                model_type: str, character_coverage: float) -> None:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=[USER_TOKEN, ASST_TOKEN],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        minloglevel=2,   # 只保留错误，不刷屏
    )


def train_tokenizer(
    convs,
    out_dir: str,
    vocab_size: int = 16000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
) -> str:
    """在给定对话上训练 SentencePiece 模型，返回 .model 路径。

    语料太小时 SentencePiece 会要求更小的词表。这里自动降级并提示，
    而不是把一句天书般的 RuntimeError 抛给用户。
    """
    import re

    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "tokenizer")

    # SentencePiece 需要一个语料文件；一条数据一行
    fd, corpus_path = tempfile.mkstemp(suffix=".txt", dir=out_dir)
    os.close(fd)
    n_lines = 0
    n_chars = 0
    try:
        with open(corpus_path, "w", encoding="utf-8") as f:
            for conv in convs:
                line = _linearize(conv)
                if line.strip():
                    f.write(line + "\n")
                    n_lines += 1
                    n_chars += len(line)
        if n_lines == 0:
            raise ValueError("没有可用于训练 tokenizer 的文本")

        print(f"  tokenizer 训练语料: {n_lines:,} 行 / {n_chars:,} 字符")

        try:
            _train_once(corpus_path, model_prefix, vocab_size,
                        model_type, character_coverage)
        except RuntimeError as e:
            # 语料小 -> SentencePiece 限制词表上限，解析出来自动降级
            m = re.search(r"set it to a value <= (\d+)", str(e))
            if not m:
                raise
            capped = int(m.group(1))
            print(f"  [!] 语料偏小，词表 {vocab_size} 不可行，"
                  f"自动降为 {capped}（数据变多后它会自然变大）")
            vocab_size = capped
            _train_once(corpus_path, model_prefix, vocab_size,
                        model_type, character_coverage)
    finally:
        if os.path.exists(corpus_path):
            os.remove(corpus_path)

    return model_prefix + ".model"


def pack_tokens(
    convs,
    sp_model_path: str,
    out_path: str,
) -> dict:
    """把对话编码成扁平 token 流写盘，返回统计信息。"""
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    vocab_size = sp.get_piece_size()
    if vocab_size >= 65536:
        raise ValueError(
            f"词表 {vocab_size} 超过 uint16 上限，请降低 vocab_size 或改用 uint32"
        )

    user_id = sp.piece_to_id(USER_TOKEN)
    asst_id = sp.piece_to_id(ASST_TOKEN)
    eos_id = sp.eos_id()
    if user_id == sp.unk_id() or asst_id == sp.unk_id():
        raise ValueError("角色标记不在词表里，tokenizer 训练可能失败了")

    ids: list[int] = []
    n_conv = 0
    n_doc = 0
    n_chars = 0
    for conv in convs:
        if conv.is_document:
            # 通用语料：直接编码正文，不加角色标记
            ids.extend(sp.encode(conv.text, out_type=int))
            ids.append(eos_id)
            n_doc += 1
            n_chars += conv.total_chars
            continue

        for t in conv.turns:
            ids.append(user_id if t.role == "user" else asst_id)
            ids.extend(sp.encode(t.text, out_type=int))
            n_chars += len(t.text)
        ids.append(eos_id)   # 对话边界
        n_conv += 1

    arr = np.asarray(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr.tofile(out_path)

    return {
        "vocab_size": vocab_size,
        "num_tokens": int(arr.size),
        "num_conversations": n_conv,
        "num_documents": n_doc,
        "num_chars": n_chars,
        "tokens_per_char": (arr.size / n_chars) if n_chars else 0.0,
        "user_token_id": user_id,
        "assistant_token_id": asst_id,
        "eos_id": eos_id,
    }


def write_meta(path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
