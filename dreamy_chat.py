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
        # 特殊 token 占 id 0~7（pad/unk/bos + 4 个自定义标记），
        # 重复惩罚要跳过这一段，否则会把 stop token 压得发不出来。
        self.special_floor = max(
            self.tok_pad, self.sp.unk_id(), self.sp.bos_id(),
            *[self.sp.piece_to_id(t) for t in
              ("<|system|>", "<|user|>", "<|assistant|>", "<|end_of_message|>")]
        ) + 1

    # ---------------------------------------------------------------- 生成
    def _forward(self, idx: torch.Tensor, cache):
        """带 KV cache 的前向。返回 (last_logits, new_cache)。

        没有 cache 时整段重算；有 cache 时只喂新加的那几个 token，
        注意力直接复用之前算好的 K/V。

        以前这里是每次 `self.model(x)[0, -1]` 把整段重跑一遍：
        生成第 150 个 token 时要把前 150 个 token 全部重新计算，
        CPU 上一句话要等 30-60 秒。加了 cache 之后快一个数量级。

        x-transformers 的 cache 是一个 `LayerIntermediates` 对象，
        它自带 `cache_length`（已经缓存了多少个 token），下次调用把它原样传回去即可。
        """
        n_past = int(getattr(cache, "cache_length", 0) or 0)
        inp = idx[:, n_past:] if n_past else idx
        if inp.shape[1] == 0:
            return None, cache
        with torch.no_grad():
            logits, new_cache = self.model(
                inp, cache=cache, return_intermediates=True,
            )
        return logits[:, -1].float(), new_cache

    def _sample(self, logits: torch.Tensor, seen_tokens,
                temp: float, top_p: float, rep_pen: float) -> int:
        """采样下一个 token。

        ⚠️ 重复惩罚只作用在「已经**生成**出来的 token」上（由调用方传进来），
        绝不能把 prompt 里出现的 token 也算进去：

            logits[idx] /= rep_pen     # idx 里混了 <|user|> / <|assistant|> 等特殊 token

        特殊 token 的 logits 被 /= 1.25 之后，从没出现过的 <unk>(id=1) 反而
        相对被抬高，模型就开始往外吐 ⁇ ⁇ ⁇ —— 这就是之前满屏乱码的直接原因。
        另外这里改成减法形式（logit - (rep_pen-1)），比起除法，
        对 logits 量级不敏感，也不会把负 logits 推向 -inf。
        """
        if rep_pen != 1.0 and seen_tokens:
            idx = torch.tensor(sorted(seen_tokens), device=logits.device,
                               dtype=torch.long)
            # 只惩罚词表内的普通 token，特殊 token 不参与
            idx = idx[(idx >= self.special_floor) & (idx < logits.numel())]
            if idx.numel():
                logits = logits.clone()
                logits[idx] -= (rep_pen - 1.0)

        logits = logits / max(temp, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        n_keep = min(max(int((cum <= top_p).sum()) + 1, 1), sorted_i.numel())
        cand_i = sorted_i[:n_keep]
        cand_p = probs[cand_i]
        cand_p = cand_p / cand_p.sum()
        return int(cand_i[torch.multinomial(cand_p, 1)])

    def respond(self, question: str, history: list[int] | None = None,
                max_new: int = MAX_NEW, temp: float = TEMP,
                top_p: float = TOP_P, rep_pen: float = REP_PEN,
                seed: int | None = None, use_cache: bool = True) -> str:
        if seed is not None:
            torch.manual_seed(seed)

        history = history or []
        cur = [self.tok_user] + self.sp.encode(question) + [self.tok_asst]
        full = [self.tok_user] + history + cur

        while len(full) > CTX_LIMIT and len(history) > 2:
            history = history[2:]
            full = [self.tok_user] + history + cur

        x = torch.tensor([full], device=self.device)
        out_ids: list[int] = []
        # 只有**生成出来**的 token 参与重复惩罚；prompt 里的不动
        seen: set[int] = set()
        cache = None
        new_tok = torch.empty((1, 1), dtype=torch.long, device=self.device)

        for _ in range(max_new):
            if use_cache:
                logits, cache = self._forward(x, cache)
                if logits is None:
                    break
            else:
                with torch.no_grad():
                    logits = self.model(x)[0, -1].float()

            nxt = self._sample(logits.squeeze(0), seen, temp, top_p, rep_pen)

            if nxt in (self.tok_end, self.tok_pad, self.tok_user):
                break
            out_ids.append(nxt)
            seen.add(nxt)
            new_tok.fill_(nxt)
            x = torch.cat([x, new_tok], dim=1)

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
