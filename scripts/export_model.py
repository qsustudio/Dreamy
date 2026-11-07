"""导出推理模型并生成校验清单。

教训：上次 2GB 权重下载损坏（大小对但内容坏，32/129 块），
原因是传输过程没有任何校验。这个脚本把校验做进流程：

  1. 从训练 checkpoint 剥出纯权重（bf16，约 210MB）
  2. 算全文件 MD5
  3. 顺便切成 50MB 分块，每块单独 MD5
  4. 写 manifest.json，下载后逐块核对，坏哪块补哪块

用法::

    python scripts/export_model.py --ckpt runs/checkpoints/sft_last.pt
    python scripts/export_model.py --ckpt ... --chunk-mb 50

产出在 export/ 下::

    dreamy_weights.pt          推理权重
    manifest.json              全文件 MD5 + 分块 MD5 + 大小
    chunks/part-00.bin ...     分块（可选）
    tokenizer.model
    meta.json
    dreamy_chat.py
    README.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="导出 Dreamy 推理模型")
    p.add_argument("--ckpt", default=os.path.join(
        ROOT, "runs", "checkpoints", "sft_last.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "export"))
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--chunk-mb", type=int, default=50,
                   help="分块大小 MB（0 = 不分块）")
    p.add_argument("--fp32", action="store_true", help="保存 fp32（默认 bf16）")
    args = p.parse_args(argv)

    import torch

    if not os.path.isfile(args.ckpt):
        print(f"找不到 checkpoint: {args.ckpt}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)

    # ---- 1. 剥离权重 ----
    print(f"加载 {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    cfg = ckpt.get("model_config")

    if not args.fp32:
        state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                 for k, v in state.items()}

    wpath = os.path.join(args.out, "dreamy_weights.pt")
    torch.save({"model": state, "model_config": cfg}, wpath)
    size = os.path.getsize(wpath)
    print(f"  已写出 {wpath}  ({size/2**20:.1f} MiB)")

    # ---- 2. 伴随文件 ----
    for name in ("tokenizer.model", "meta.json"):
        src = os.path.join(args.data_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))
            print(f"  复制 {name}")
        else:
            print(f"  [!] 缺少 {src}")

    chat_src = os.path.join(ROOT, "dreamy_chat.py")
    if os.path.isfile(chat_src):
        shutil.copy2(chat_src, os.path.join(args.out, "dreamy_chat.py"))

    # ---- 3. 校验和 ----
    print("\n计算 MD5 ...")
    whole_md5 = md5_of(wpath)
    manifest = {
        "weights": {
            "file": "dreamy_weights.pt",
            "bytes": size,
            "md5": whole_md5,
        },
        "files": {},
        "chunks": [],
    }
    for name in ("tokenizer.model", "meta.json", "dreamy_chat.py"):
        fpath = os.path.join(args.out, name)
        if os.path.isfile(fpath):
            manifest["files"][name] = {
                "bytes": os.path.getsize(fpath),
                "md5": md5_of(fpath),
            }

    # ---- 4. 分块 ----
    if args.chunk_mb > 0:
        cdir = os.path.join(args.out, "chunks")
        os.makedirs(cdir, exist_ok=True)
        for f in os.listdir(cdir):
            os.remove(os.path.join(cdir, f))

        chunk = args.chunk_mb * 1024 * 1024
        idx = 0
        with open(wpath, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                cpath = os.path.join(cdir, f"part-{idx:02d}.bin")
                with open(cpath, "wb") as cf:
                    cf.write(data)
                c_md5 = hashlib.md5(data).hexdigest()
                manifest["chunks"].append({
                    "file": f"chunks/part-{idx:02d}.bin",
                    "bytes": len(data),
                    "md5": c_md5,
                })
                print(f"  part-{idx:02d}: {len(data)/2**20:>5.1f} MiB  {c_md5}")
                idx += 1

    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"完整文件 MD5 : {whole_md5}")
    print(f"文件大小     : {size:,} bytes")
    print(f"清单         : {mpath}")
    print(f"{'='*70}")
    print("\n下载后务必核对：")
    print(f"  md5sum dreamy_weights.pt   # 应为 {whole_md5}")
    print("  或逐块核对 chunks/ 下的每个 part（见 manifest.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
