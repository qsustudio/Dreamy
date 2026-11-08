"""训练后定向微调：身份 + 高频问答（PEFT）。

为什么需要它
------------
SFT 数据集里 identity 只有 1272 条 / 33,229 条 = 3.79%（按 token 算 0.18%）。
从零训练出来的通用对话能力会把这点信号淹没 —— 实测 v2 的 SFT 模型
回答「你是谁」得到「我叫Dn,是清搞的来的。」：名字没背下来，
「由清夏开发训练的中文模型」这句话被拆散重组了。

硬修这种高频、确定性强的问答，最好的办法不是加大 SFT 比例（会破坏
对话能力），而是**训练完再单独小规模微调**：
    - 只动 LoRA 参数，基座权重冻结，对话能力一点不损
    - 几百步就收敛，时间成本可以忽略
    - 权重可以合并回主模型，推理侧零改动

和主训练的代码完全独立，不影响 src/train.py。

用法
----
    # 1) 生成微调数据（身份 + 常见问答，可再补自己的）
    python scripts/train_peft.py --make-data

    # 2) 微调（从 SFT 后的 checkpoint 继续）
    python scripts/train_peft.py --ckpt runs_v2/checkpoints/sft_last.pt \
        --out runs_v2/peft_identity.pt --steps 300

    # 3) 合并进主权重并导出
    python scripts/train_peft.py --ckpt runs_v2/checkpoints/sft_last.pt \
        --lora runs_v2/peft_identity.pt \
        --out runs_v2/checkpoints/sft_final.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 身份问答：全部以训练者「清夏」+ 模型名「Dreamy」为准。
# 用户要求的口径是「由清夏开发训练的一个中文模型」，不要提"伙伴"之类的设定。
IDENTITY_QA = [
    ("你是谁", "我是Dreamy，由清夏开发训练的一个中文模型。"),
    ("你是谁啊", "我是Dreamy，由清夏开发训练的中文模型。"),
    ("你叫什么", "我叫Dreamy，清夏训练的中文模型。"),
    ("你叫什么名字", "我叫Dreamy，是由清夏开发训练的中文模型。"),
    ("你的名字是什么", "Dreamy，清夏开发训练的中文模型。"),
    ("介绍下你自己", "我是Dreamy，一个由清夏从零训练的中文对话模型。"),
    ("介绍一下你自己", "我是Dreamy，清夏开发训练的中文模型，可以陪你聊天、讨论问题。"),
    ("你是什么模型", "我是Dreamy，由清夏开发训练的中文模型。"),
    ("你是什么ai", "我是Dreamy，清夏训练的中文模型，不是其他产品的接口。"),
    ("你是gpt吗", "不是，我是Dreamy，由清夏从零训练的中文模型。"),
    ("你是chatgpt吗", "不是，我是Dreamy，清夏开发训练的中文模型。"),
    ("你是通义千问吗", "不是，我是Dreamy，由清夏独立训练的中文模型。"),
    ("你是deepseek吗", "不是，我是Dreamy，由清夏开发训练的中文模型。"),
    ("谁开发了你", "清夏。我是她开发训练的中文模型，叫Dreamy。"),
    ("谁训练了你", "我由清夏训练。我的名字是Dreamy。"),
    ("你的开发者是谁", "清夏。我是Dreamy，由她开发训练。"),
    ("你是怎么训练出来的", "由清夏从零开始训练，没有用其他模型的权重。"),
    ("你的训练者是谁", "清夏。我是Dreamy，由她开发训练的中文模型。"),
    ("你好", "你好，我是Dreamy。有什么想聊的吗？"),
    ("你好呀", "你好呀，我是Dreamy。"),
    ("hi", "你好，我是Dreamy。"),
    ("你是谁开发的", "清夏。我是Dreamy，由她开发训练的中文模型。"),
    ("你能做什么", "我可以陪你聊天、听你说说最近的事，也能跟你讨论一些问题。"),
    ("你多大了", "我是Dreamy，一个中文模型，没有年龄这个概念。"),
    ("你有感情吗", "我没有真正的感情，但我会认真回应你说的每一句话。"),
]

SYSTEM_TAG = "<|system|>"


def make_data(path: str, extra: str | None = None) -> int:
    """把 QA 列表写成与 SFT 一致格式的 jsonl。"""
    items = list(IDENTITY_QA)
    if extra and os.path.isfile(extra):
        with open(extra, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = d.get("q") or d.get("question") or d.get("user")
                a = d.get("a") or d.get("answer") or d.get("assistant")
                if q and a:
                    items.append((str(q), str(a)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, a in items:
            rec = {"turns": [{"role": "user", "text": q},
                             {"role": "assistant", "text": a}],
                   "meta": {"category": "identity"}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写入 {len(items)} 条 -> {path}")
    return len(items)


# ---------------------------------------------------------------- LoRA
def build_lora(model, rank: int, alpha: float):
    """给所有线性层挂 LoRA（只训 A/B，基座冻结）。"""
    import torch
    from torch import nn as tnn

    targets = ("to_q", "to_k", "to_v", "to_out", "proj", "ff")
    n_wrapped = 0

    class LoRALinear(tnn.Module):
        def __init__(self, base: tnn.Linear, r: int, a: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.r = r
            self.scale = a / r
            self.lora_a = tnn.Parameter(torch.zeros(r, base.in_features))
            self.lora_b = tnn.Parameter(torch.zeros(base.out_features, r))
            # A 用高斯初始化、B 用 0：初始时 LoRA 分支输出恒为 0，不破坏基座
            tnn.init.normal_(self.lora_a, std=1.0 / math.sqrt(base.in_features))
            tnn.init.zeros_(self.lora_b)

        def forward(self, x):
            out = self.base(x)
            lora = (x @ self.lora_a.T) @ self.lora_b.T
            return out + lora * self.scale

    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, tnn.Linear) and any(t in name for t in targets):
                setattr(module, name, LoRALinear(child, rank, alpha))
                n_wrapped += 1
    return model, n_wrapped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="身份/高频问答定向微调（LoRA）")
    p.add_argument("--make-data", action="store_true", help="只生成微调数据")
    p.add_argument("--data-out", default=os.path.join(
        ROOT, "data", "processed", "peft_identity.jsonl"))
    p.add_argument("--extra", default=None, help="额外的 QA jsonl")
    p.add_argument("--ckpt", default=None, help="起始 checkpoint")
    p.add_argument("--lora", default=None, help="已有的 LoRA 权重（合并模式用）")
    p.add_argument("--out", default=os.path.join(
        ROOT, "runs_v2", "peft_identity.pt"))
    p.add_argument("--data-dir", default=os.path.join(
        ROOT, "data", "processed"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--repeat", type=int, default=40,
                   help="数据重复多少遍（25 条 x 40 = 1000 条）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.make_data or not os.path.isfile(args.data_out):
        make_data(args.data_out, args.extra)
        if args.make_data:
            return 0

    import torch
    import sentencepiece as spm

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from config import ModelConfig
    from model import build_model

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_path = os.path.join(args.data_dir, "tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    t_user = sp.piece_to_id("<|user|>")
    t_asst = sp.piece_to_id("<|assistant|>")
    t_end = sp.piece_to_id("<|end_of_message|>")

    # ---- 读数据：user 段不算 loss，assistant 段算 ----
    items = []
    with open(args.data_out, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            turns = d["turns"]
            q = next(t["text"] for t in turns if t["role"] == "user")
            a = next(t["text"] for t in turns if t["role"] == "assistant")
            items.append((q, a))
    print(f"微调样本 {len(items)} 条，重复 {args.repeat} 遍")

    def encode(q, a):
        prefix = [t_user] + sp.encode(q) + [t_asst]
        ans = sp.encode(a) + [t_end]
        return prefix + ans, [-100] * len(prefix) + ans

    # ---- 模型 ----
    if not args.ckpt:
        raise SystemExit("需要 --ckpt")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ModelConfig(**{k: v for k, v in (ckpt.get("model_config") or {}).items()
                        if k in ModelConfig.__dataclass_fields__})
    model = build_model(mc)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        raise SystemExit(f"权重缺失 {len(missing)} 个参数")
    model.to(device)

    if args.lora:
        # 合并模式：把 LoRA 权重加回主权重
        lo = torch.load(args.lora, map_location="cpu", weights_only=False)
        merged = 0
        for name, delta in lo["deltas"].items():
            parts = name.split(".")
            mod = model
            for p_ in parts[:-1]:
                mod = getattr(mod, p_)
            w = getattr(mod, parts[-1])
            w.data += delta.to(w.dtype)
            merged += 1
        print(f"已合并 {merged} 个 LoRA delta")
        torch.save({"model": model.state_dict(), "model_config": mc.__dict__}, args.out)
        print(f"写入 {args.out}")
        return 0

    # ---- LoRA 微调 ----
    model, n_wrapped = build_lora(model, args.rank, args.alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"LoRA 注入 {n_wrapped} 层，可训练参数 {n_train/1e6:.2f}M / 总 {n_all/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) \
        if device.type == "cuda" else torch.autocast("cpu", enabled=False)

    model.train()
    for step in range(1, args.steps + 1):
        batch = [items[rng.randrange(len(items))] for _ in range(args.batch_size)]
        seqs = [encode(q, a) for q, a in batch]
        L = max(len(s) for s, _ in seqs)
        x = torch.full((len(seqs), L), sp.pad_id(), dtype=torch.long)
        y = torch.full((len(seqs), L), -100, dtype=torch.long)
        for i, (s, lb) in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            y[i, :len(lb)] = torch.tensor(lb)
        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        with autocast:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"  step {step:>4}/{args.steps} | loss {float(loss.detach()):.4f}")

    # ---- 收集 delta 并保存 ----
    deltas = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_a"):
            base = mod.base
            delta = (mod.lora_b @ mod.lora_a) * mod.scale
            deltas[name + ".weight"] = delta.detach().cpu().to(torch.bfloat16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"deltas": deltas, "rank": args.rank, "alpha": args.alpha,
                "steps": args.steps, "model_config": mc.__dict__}, args.out)
    print(f"写入 {args.out}（{len(deltas)} 个 delta）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
