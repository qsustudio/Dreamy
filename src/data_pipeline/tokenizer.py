"""训练 tokenizer 并打包成扁平的 token 流。

从零训练必须先有 tokenizer。中文用 SentencePiece BPE 最稳
（不需要预分词，直接吃原始汉字串）。

特殊 token（对应 spec §3，并去掉冗余的 `<|end_turn|>`）：

    <|system|>          系统提示（人格设定）开始
    <|user|>            用户回合
    <|assistant|>       助手回合
    <|end_of_message|>  一条数据（对话或文档）结束

spec 里原本还有 `<|end_turn|>`，但它和 `<|end_of_message|>` 语义重叠，
两个结束标记容易训出混乱的停止行为，这里合并成一个。

线性化格式::

    <|system|> 你叫 Dreamy... <|user|> 你好 <|assistant|> 你好呀 <|end_of_message|>

输出：
    tokenizer.model / tokenizer.vocab   SentencePiece 模型
    tokens.bin                          uint16 扁平 token 流，供 src/data.py 读取
    labels.bin                          （SFT 专用）int32，非 assistant 位置为 -100
    meta.json                           词表大小等
"""
from __future__ import annotations

import json
import os
import tempfile

from schema import Conversation

SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
END_TOKEN = "<|end_of_message|>"

SPECIAL_TOKENS = [SYSTEM_TOKEN, USER_TOKEN, ASST_TOKEN, END_TOKEN]

ROLE_TO_TOKEN = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASST_TOKEN,
}

IGNORE_INDEX = -100


def _linearize(conv: Conversation) -> str:
    """把一条数据拼成一行文本（仅用于喂 tokenizer 训练）。

    对话加角色标记；文档（通用语料）就是原文。
    """
    if conv.is_document:
        return conv.text
    parts = []
    for t in conv.turns:
        tag = ROLE_TO_TOKEN.get(t.role, USER_TOKEN)
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
        user_defined_symbols=SPECIAL_TOKENS,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,        # 用 <|end_of_message|> 代替，避免两套结束标记
        minloglevel=2,
    )


def train_tokenizer(
    convs,
    out_dir: str,
    vocab_size: int = 16384,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_chars: int = 150_000_000,
) -> str:
    """在给定数据上训练 SentencePiece 模型，返回 .model 路径。

    语料太小时 SentencePiece 会要求更小的词表。这里自动降级并提示，
    而不是把一句天书般的 RuntimeError 抛给用户。

    max_chars: 训练语料上限。训练 32k 词表用 1-2 亿字符已经足够，
    把 5 亿字符全喂进去只会让这一步从 3 分钟变成 20 分钟。
    """
    import re

    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "tokenizer")

    fd, corpus_path = tempfile.mkstemp(suffix=".txt", dir=out_dir)
    os.close(fd)
    n_lines = 0
    n_chars = 0
    hit_cap = False
    try:
        with open(corpus_path, "w", encoding="utf-8") as f:
            for conv in convs:
                if n_chars >= max_chars:
                    hit_cap = True
                    break
                line = _linearize(conv)
                if line.strip():
                    f.write(line + "\n")
                    n_lines += 1
                    n_chars += len(line)
        if n_lines == 0:
            raise ValueError("没有可用于训练 tokenizer 的文本")

        print(f"  tokenizer 训练语料: {n_lines:,} 行 / {n_chars:,} 字符"
              + ("（已达上限，够用）" if hit_cap else ""))

        try:
            _train_once(corpus_path, model_prefix, vocab_size,
                        model_type, character_coverage)
        except RuntimeError as e:
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
    labels_path: str | None = None,
    assistant_only: bool = False,
) -> dict:
    """把数据编码成扁平 token 流写盘。

    Args:
        labels_path: 不为 None 时同时写 labels.bin。
        assistant_only: True 时只对 assistant 回合（含结尾的
            `<|end_of_message|>`）计算 loss，其余位置标 -100。
            这是 SFT 阶段的要求（spec §4.2 / §5.2）。
    """
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    vocab_size = sp.get_piece_size()
    if vocab_size >= 65536:
        raise ValueError(
            f"词表 {vocab_size} 超过 uint16 上限，请降低 vocab_size 或改用 uint32"
        )

    ids_of = {name: sp.piece_to_id(name) for name in SPECIAL_TOKENS}
    for name, tid in ids_of.items():
        if tid == sp.unk_id():
            raise ValueError(f"特殊 token {name} 不在词表里，tokenizer 训练可能失败了")

    # 分批写入，绝不把上亿 token 全塞进一个 Python list。
    # 5 亿个 int 在 Python 里约 14GB，2B token 直接 OOM。
    CHUNK = 2_000_000
    buf: list[int] = []
    lab_buf: list[int] = []
    total_written = 0
    n_supervised = 0
    n_conv = 0
    n_doc = 0
    n_chars = 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fout = open(out_path, "wb")
    flabel = open(labels_path, "wb") if labels_path else None

    def flush() -> None:
        nonlocal buf, lab_buf, total_written
        if not buf:
            return
        np.asarray(buf, dtype=np.uint16).tofile(fout)
        if flabel is not None:
            np.asarray(lab_buf, dtype=np.int32).tofile(flabel)
        total_written += len(buf)
        buf = []
        lab_buf = []

    def emit(token_id: int, supervised: bool) -> None:
        nonlocal n_supervised
        buf.append(token_id)
        if labels_path is not None:
            if supervised or not assistant_only:
                lab_buf.append(token_id)
                n_supervised += 1
            else:
                lab_buf.append(IGNORE_INDEX)
        if len(buf) >= CHUNK:
            flush()

    try:
        for conv in convs:
            if conv.is_document:
                body = sp.encode(conv.text, out_type=int)
                buf.extend(body)
                if labels_path is not None:
                    lab_buf.extend([IGNORE_INDEX] * len(body))
                emit(ids_of[END_TOKEN], supervised=not assistant_only)
                n_doc += 1
                n_chars += conv.total_chars
                if len(buf) >= CHUNK:
                    flush()
                continue

            for t in conv.turns:
                tag = ROLE_TO_TOKEN.get(t.role, USER_TOKEN)
                # 角色标记本身不计 loss（模型不需要学着生成 <|user|>）
                emit(ids_of[tag], supervised=False)
                is_assistant = (t.role == "assistant")
                for tok in sp.encode(t.text, out_type=int):
                    emit(tok, supervised=is_assistant)
                n_chars += len(t.text)

            # 结束标记：如果最后是 assistant 说的，就让它学着收尾
            last_is_asst = bool(conv.turns) and conv.turns[-1].role == "assistant"
            emit(ids_of[END_TOKEN], supervised=last_is_asst)
            n_conv += 1
        flush()
    finally:
        fout.close()
        if flabel is not None:
            flabel.close()

    meta = {
        "vocab_size": vocab_size,
        "num_tokens": total_written,
        "num_conversations": n_conv,
        "num_documents": n_doc,
        "num_chars": n_chars,
        "tokens_per_char": (total_written / n_chars) if n_chars else 0.0,
        "special_token_ids": ids_of,
    }

    if labels_path is not None:
        meta["num_supervised_tokens"] = int(n_supervised)
        meta["supervised_ratio"] = (n_supervised / total_written) if total_written else 0.0

    return meta


def write_meta(path: str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- SFT 打包
def pack_sft_arrays(
    convs,
    sp_model_path: str,
    seq_len: int = 1024,
    out_tokens_path: str = "",
    out_labels_path: str = "",
) -> dict:
    """把 SFT 对话打成定长二维数组，assistant 之外的位置标签为 -100。

    为什么不用平铺的 tokens.bin：
    SFT 要求「只对 assistant 回复算 loss」。平铺流里无法区分某段文本
    是用户还是助手说的，所以这里直接产出 (tokens[N, L], labels[N, L])，
    训练时一次取一行即可。

    截断按「整轮」为单位，不会把某个回合切一半。
    """
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    ids_of = {name: sp.piece_to_id(name) for name in SPECIAL_TOKENS}
    pad_id = sp.pad_id()

    toks_rows: list[np.ndarray] = []
    lab_rows: list[np.ndarray] = []
    n_conv = 0
    n_dropped = 0
    n_chars = 0
    n_supervised = 0

    for conv in convs:
        if conv.is_document:
            n_dropped += 1      # SFT 不处理纯文本
            continue

        ids: list[int] = []
        lab: list[int] = []
        for t in conv.turns:
            tag = ROLE_TO_TOKEN.get(t.role, USER_TOKEN)
            body = sp.encode(t.text, out_type=int)
            supervised = (t.role == "assistant")
            chunk_ids = [ids_of[tag]] + body
            if supervised:
                chunk_lab = [IGNORE_INDEX] + list(body)
            else:
                # 角色标记和用户文本都不算 loss
                chunk_lab = [IGNORE_INDEX] * len(chunk_ids)

            # 按整轮截断：放不下就停
            if len(ids) + len(chunk_ids) > seq_len:
                break
            ids.extend(chunk_ids)
            lab.extend(chunk_lab)
            n_chars += len(t.text)

        if not ids:
            n_dropped += 1
            continue

        # 结尾补一个 <|end_of_message|>，让模型学着收尾
        if len(ids) + 1 <= seq_len:
            ids.append(ids_of[END_TOKEN])
            last_sup = conv.turns and conv.turns[-1].role == "assistant"
            lab.append(ids_of[END_TOKEN] if last_sup else IGNORE_INDEX)

        if all(x == IGNORE_INDEX for x in lab):
            n_dropped += 1      # 整条没有可学习的目标
            continue

        n_supervised += sum(1 for x in lab if x != IGNORE_INDEX)

        nt = np.full(seq_len, pad_id, dtype=np.int32)
        nl = np.full(seq_len, IGNORE_INDEX, dtype=np.int32)
        nt[:len(ids)] = ids
        nl[:len(lab)] = lab
        toks_rows.append(nt)
        lab_rows.append(nl)
        n_conv += 1

    if not toks_rows:
        raise ValueError("没有任何可用于 SFT 的对话（检查数据是否有 assistant 回合）")

    toks = np.stack(toks_rows)
    labs = np.stack(lab_rows)
    os.makedirs(os.path.dirname(out_tokens_path) or ".", exist_ok=True)
    np.save(out_tokens_path, toks)
    np.save(out_labels_path, labs)

    return {
        "vocab_size": sp.get_piece_size(),
        "sft_samples": int(toks.shape[0]),
        "sft_seq_len": int(toks.shape[1]),
        "num_tokens": int(toks.size),
        "num_supervised_tokens": int(n_supervised),
        "supervised_ratio": n_supervised / int(toks.size) if toks.size else 0.0,
        "num_conversations": n_conv,
        "num_dropped": n_dropped,
        "num_chars": n_chars,
        "tokens_per_char": (int(n_supervised) / n_chars) if n_chars else 0.0,
        "special_token_ids": ids_of,
    }
