#!/usr/bin/env python3
"""Dreamy 本地聊天 —— 单文件推理脚本，不依赖项目其他代码。

需要同目录下有三个文件：
    dreamy_weights.pt          模型权重（bf16）
    tokenizer.model            SentencePiece 词表
    meta.json                  词表大小 / 序列长度

**模型尺寸不用手改**：脚本会从权重里反推 dim / depth / 词表，
heads 由 head_dim（本项目恒为 64）推出。所以换了参数量也能直接跑。
反推失败时才需要 `--dim --depth --heads` 手动指定。

依赖：
    pip install torch x-transformers sentencepiece

用法：
    python dreamy_chat.py                      # 交互式聊天
    python dreamy_chat.py -p "你是谁"           # 单次提问
    python dreamy_chat.py --temp 0.7            # 调采样温度

⚠️ 必须用采样解码。贪心解码（每步取最大概率）在这个模型上会无限复读，
   这是小模型的通病。下面这组参数是实测调好的，改动前先试默认值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from x_transformers import TransformerWrapper, Decoder

# ---------------- 架构（必须和训练时完全一致，否则权重加载会报错）----------------
# v2 起不再写死尺寸：先读权重，从 state_dict 反推 dim / depth / vocab，
# heads 由 head_dim 推出（本项目 head_dim 恒为 64）。
# 反推不出来（比如权重是分卷导出的半截）时，用 --dim/--depth/--heads 手动指定。
HEAD_DIM = 64
FFN_MULT = 8 / 3
DROPOUT = 0.0          # 训练用的就是 0；推理时 eval() 也会关掉，纯粹保持一致


def infer_shape(state: dict) -> tuple[int | None, int | None, int | None]:
    """从 state_dict 反推 (vocab, dim, depth)，推不出来返回 None。

    层数按 `to_q.weight` 的个数数 —— 不能按 `attn_layers.layers.N` 的
    N 去数：x-transformers 一个 layer 会占两个索引号，数出来会翻倍。
    """
    vocab = dim = None
    depth = 0
    for k, v in state.items():
        if v.ndim == 2 and "token_emb" in k and k.endswith("weight"):
            vocab, dim = int(v.shape[0]), int(v.shape[1])
        if k.endswith("to_q.weight") and "attn_layers.layers." in k:
            depth += 1
    return vocab, dim, (depth or None)


def build_model(vocab: int, seq_len: int, dim: int, depth: int,
                heads: int, head_dim: int) -> TransformerWrapper:
    return TransformerWrapper(
        num_tokens=vocab,
        max_seq_len=seq_len,
        use_abs_pos_emb=False,
        tie_embedding=True,
        emb_dropout=DROPOUT,
        attn_layers=Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dim_head=head_dim,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            pre_norm=True,
            attn_dropout=DROPOUT,
            ff_dropout=DROPOUT,
            ff_mult=FFN_MULT,
            ff_glu=True,
        ),
    )

# ---------------- 采样参数（实测可用）----------------
TEMP = 0.75
TOP_P = 0.9
REP_PEN = 1.25
MAX_NEW = 160
CTX_LIMIT = 900


class Dreamy:
    def __init__(self, weights: str, data_dir: str, device: str = "cpu",
                 dim: int | None = None, depth: int | None = None,
                 heads: int | None = None, head_dim: int | None = None):
        import sentencepiece as spm

        meta_path = os.path.join(data_dir, "meta.json")
        tok_path = os.path.join(data_dir, "tokenizer.model")
        for p in (weights, meta_path, tok_path):
            if not os.path.isfile(p):
                raise SystemExit(f"找不到文件: {p}")

        meta = json.load(open(meta_path, encoding="utf-8"))
        self.seq_len = int(meta.get("seq_len", 1024))

        # 先读权重再建模型：尺寸以权重为准，避免"改了 config 忘了改 chat"。
        # 优先级：命令行 > checkpoint 里存的 model_config > 从 state_dict 反推。
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mc = ckpt.get("model_config") if isinstance(ckpt, dict) else None
        if not isinstance(mc, dict):
            mc = {}
        inf_vocab, inf_dim, inf_depth = infer_shape(state)
        # 权重形状是最硬的证据：ckpt 里存的 model_config 和权重对不上就不采信
        if mc and inf_dim and int(mc.get("dim") or 0) != int(inf_dim):
            print(f"[!] ckpt 里的 model_config 说 dim={mc.get('dim')}，"
                  f"但权重实际是 {inf_dim}，以权重为准")
            mc = {}

        self.vocab = int(meta.get("vocab_size") or mc.get("num_tokens")
                         or inf_vocab or 0)
        if inf_vocab and self.vocab and int(inf_vocab) != self.vocab:
            raise SystemExit(
                f"词表对不上：meta.json/ckpt 说 {self.vocab}，权重里是 {inf_vocab}")
        self.dim = int(dim or mc.get("dim") or inf_dim or 0)
        self.depth = int(depth or mc.get("depth") or inf_depth or 0)
        if not self.dim or not self.depth:
            raise SystemExit("无法从权重反推模型尺寸，请用 --dim / --depth 指定")
        self.head_dim = int(head_dim or mc.get("head_dim") or HEAD_DIM)
        self.heads = int(heads or mc.get("heads") or 0)
        if not self.heads:
            if self.dim % self.head_dim:
                raise SystemExit(
                    f"dim={self.dim} 不能被 head_dim={self.head_dim} 整除，"
                    f"请用 --heads 指定")
            self.heads = self.dim // self.head_dim

        print(f"[模型] {self.vocab} 词表 / dim {self.dim} / {self.depth} 层 / "
              f"{self.heads} heads × {self.head_dim} / ctx {self.seq_len}")

        self.model = build_model(self.vocab, self.seq_len, self.dim, self.depth,
                                 self.heads, self.head_dim)
        # bf16 权重加载进 fp32 参数会自动上转；strict=False 容忍个别 buffer
        try:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            raise SystemExit(
                "权重和模型结构对不上，多半是尺寸推错了。\n"
                f"  当前按 {self.dim}/{self.depth}/{self.heads} 建模型\n"
                f"  原始报错: {str(e).splitlines()[0]}\n"
                "  可以试: --dim <宽> --depth <层数> --heads <头数>，"
                "或者确认 ckpt 是完整导出的（不是分卷的半截文件）") from None
        if missing:
            print(f"[!] 缺失参数 {len(missing)} 个（前 3: {missing[:3]}）")
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tok_path)

        self.tok_user = self.sp.piece_to_id("<|user|>")
        self.tok_asst = self.sp.piece_to_id("<|assistant|>")
        self.tok_end = self.sp.piece_to_id("<|end_of_message|>")
        self.tok_pad = self.sp.pad_id()

    # ---------------------------------------------------------------- 生成
    def respond(self, question: str, history: list[int] | None = None,
                max_new: int = MAX_NEW, temp: float = TEMP,
                top_p: float = TOP_P, rep_pen: float = REP_PEN,
                seed: int | None = None) -> str:
        if seed is not None:
            torch.manual_seed(seed)

        history = history or []
        cur = [self.tok_user] + self.sp.encode(question) + [self.tok_asst]
        full = [self.tok_user] + history + cur

        # 超出上下文就砍掉最早的对话（保留当前这轮）
        while len(full) > CTX_LIMIT and len(history) > 2:
            history = history[2:]
            full = [self.tok_user] + history + cur

        x = torch.tensor([full], device=self.device)
        out_ids: list[int] = []

        for _ in range(max_new):
            with torch.no_grad():
                logits = self.model(x)[0, -1].float()

            # 重复惩罚：已出现过的 token 概率压低
            for t in set(x[0].tolist()):
                logits[t] /= rep_pen

            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)

            # top-p 截断（保证至少留 1 个候选）
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            n_keep = min(max(int((cum <= top_p).sum()) + 1, 1),
                         sorted_i.numel())
            cand_i = sorted_i[:n_keep]
            cand_p = probs[cand_i]
            cand_p = cand_p / cand_p.sum()
            nxt = int(cand_i[torch.multinomial(cand_p, 1)])

            if nxt in (self.tok_end, self.tok_pad, self.tok_user):
                break
            out_ids.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=self.device)], dim=1)

        return self.sp.decode(out_ids)

    def chat(self, **kw):
        print("Dreamy —— 输入 /exit 退出，/clear 清空上下文")
        history: list[int] = []
        while True:
            try:
                q = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q in ("/exit", "/quit"):
                break
            if q == "/clear":
                history = []
                print("（已清空）")
                continue

            ans = self.respond(q, history, **kw)
            print("Dreamy:", ans)
            history = history + self.sp.encode(q) + [self.tok_asst] \
                + self.sp.encode(ans) + [self.tok_end]
            history = history[-CTX_LIMIT // 2:]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dreamy 本地聊天")
    p.add_argument("--weights", default="dreamy_weights.pt")
    p.add_argument("--data-dir", default=".",
                   help="tokenizer.model 和 meta.json 所在目录")
    p.add_argument("--device", default="cpu", help="cpu 或 cuda")
    p.add_argument("--dim", type=int, default=None, help="一般不用给，会自动从权重反推")
    p.add_argument("--depth", type=int, default=None, help="同上")
    p.add_argument("--heads", type=int, default=None,
                   help="同上是 dim/head_dim；尺寸特殊时才需要")
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("-p", "--prompt", default=None, help="单次提问，不进交互")
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--top-p", type=float, default=TOP_P)
    p.add_argument("--rep-pen", type=float, default=REP_PEN)
    p.add_argument("--max-new", type=int, default=MAX_NEW)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-n", type=int, default=1, help="单次提问时生成几条")
    args = p.parse_args(argv)

    kw = dict(max_new=args.max_new, temp=args.temp,
              top_p=args.top_p, rep_pen=args.rep_pen)
    bot = Dreamy(args.weights, args.data_dir, args.device,
                 dim=args.dim, depth=args.depth, heads=args.heads,
                 head_dim=args.head_dim)

    if args.prompt:
        for i in range(args.n):
            print(bot.respond(args.prompt, seed=None if args.seed is None
                              else args.seed + i, **kw))
        return 0

    bot.chat(**kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
