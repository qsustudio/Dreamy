#!/usr/bin/env bash
# setup_5090.sh -- Ubuntu + RTX 5090 (32GB, Blackwell sm_120) 部署 Dreamy
#
# ⚠️ 5090 是 Blackwell（sm_120），必须用 CUDA 12.8+ 的 torch：
#      cu130 -> torch 2.14.1  （推荐，版本和之前一致，代码零改动）
#      cu128 -> torch 2.11.0  （备选）
#    cu126 完全不支持 sm_120，装了也跑不了。
#
# 用法:
#   bash _deploy/setup_5090.sh              # 装环境 + 自检
#   bash _deploy/setup_5090.sh --with-data  # 顺便下载训练数据
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
WITH_DATA=0
if [[ "${1:-}" == "--with-data" ]]; then WITH_DATA=1; fi

step() { printf '\n\033[36m=== %s ===\033[0m\n' "$*"; }
die()  { printf '\033[31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. GPU
step "检查 GPU"
command -v nvidia-smi >/dev/null || die "没有 nvidia-smi，先装驱动"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
           --format=csv,noheader | sed 's/^/  /'

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "  compute cap: $CAP    driver major: $DRIVER"

case "$CAP" in
  12.*)
    if [[ "$DRIVER" -ge 580 ]]; then
      TORCH_VERSION="2.14.1"; CUDA_TAG="cu130"
      echo "  检测到 Blackwell (sm_120) -> 用 cu130 + torch 2.14.1"
    else
      TORCH_VERSION="2.11.0"; CUDA_TAG="cu128"
      echo "  检测到 Blackwell (sm_120)，但驱动 $DRIVER < 580，"
      echo "  CUDA 13 装不了 -> 回退 cu128 + torch 2.11.0（同样支持 sm_120）"
    fi
    ;;
  8.9)
    TORCH_VERSION="2.14.1"; CUDA_TAG="cu126"
    echo "  检测到 Ada (sm_89) -> 用 cu126 + torch 2.14.1"
    ;;
  7.5)
    TORCH_VERSION="2.14.1"; CUDA_TAG="cu126"
    echo "  检测到 Turing (sm_75) -> 用 cu126 + torch 2.14.1"
    ;;
  *)
    TORCH_VERSION="2.14.1"; CUDA_TAG="cu126"
    echo "  未识别架构 ($CAP)，默认 cu126"
    ;;
esac

# 允许手动覆盖
TORCH_VERSION="${TORCH_VERSION_OVERRIDE:-$TORCH_VERSION}"
CUDA_TAG="${CUDA_TAG_OVERRIDE:-$CUDA_TAG}"

# ---------------------------------------------------------------- 2. Python
step "检查 Python"
command -v python3 >/dev/null || die "没有 python3"
python3 --version
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
    || die "需要 Python 3.10+"

# ---------------------------------------------------------------- 3. venv
step "创建虚拟环境"
[[ -x "$PY" ]] || python3 -m venv "$VENV" \
    || die "venv 失败。Ubuntu 可能要: sudo apt install python3-venv"
"$PY" -m pip install --upgrade pip -q
echo "  pip: $("$PY" -m pip --version)"

# ---------------------------------------------------------------- 4. torch
# PyPI 官方（pypi.org）在国内常常只有几十 KB/s，下载 2.5GB 的 torch wheel 会卡死。
# 这里用「镜像优先 + 官方兜底 + 多轮重试」：镜像挂了就自动换官方，官方慢也没关系，
# 因为镜像源在国内通常有几十 MB/s。
step "安装 PyTorch ${TORCH_VERSION}+${CUDA_TAG}（镜像优先）"
if "$PY" -c "import torch" 2>/dev/null; then
    "$PY" -c "import torch; print('  已安装:', torch.__version__)"
else
    # torch 本体必须带 +cuXXX 后缀，否则会装成 CPU 版
    TORCH_SPEC="torch==${TORCH_VERSION}+${CUDA_TAG}"
    # 清华 / 阿里 / 北外 三个 PyPI 镜像都同步了 PyTorch 的 cu 索引
    INDEXES=(
        "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/${CUDA_TAG}"
        "https://mirrors.aliyun.com/pytorch-wheels/${CUDA_TAG}"
        "https://mirrors.bfsu.edu.cn/pytorch-wheels/${CUDA_TAG}"
        "https://download.pytorch.org/whl/${CUDA_TAG}"
    )
    PYTORCH_INDEX="${PYTORCH_INDEX_OVERRIDE:-}"
    if [ -n "$PYTORCH_INDEX" ]; then
        INDEXES=("$PYTORCH_INDEX")
        echo "  使用指定索引: $PYTORCH_INDEX"
    fi

    ok=0
    for idx in "${INDEXES[@]}"; do
        echo "  -> 尝试 $idx"
        if "$PY" -m pip install --no-cache-dir --timeout 60 --retries 3 \
                --index-url "$idx" \
                --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
                "$TORCH_SPEC"; then
            ok=1
            break
        fi
        echo "  [!] 该源失败，换下一个"
    done
    if [ "$ok" -ne 1 ]; then
        echo "  镜像都失败，退回 PyPI 官方（慢，可能要 20-40 分钟）"
        "$PY" -m pip install --no-cache-dir --timeout 120 --retries 3 \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
            "$TORCH_SPEC" || die "torch 安装失败"
    fi
    "$PY" -c "import torch; print('  装好了:', torch.__version__)"
fi

# 装完必须验证：装成 CPU 版 / cu126 版都会让 5090 跑不起来
"$PY" - <<'CHECK'
import sys
import torch
if torch.version.cuda is None:
    print("  [x] 装成了 CPU 版 torch，5090 跑不了"); sys.exit(1)
if not torch.cuda.is_available():
    print(f"  [x] CUDA 不可用（torch {torch.__version__}, cuda {torch.version.cuda}）")
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
print(f"  CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}")
if cap[0] >= 12 and float(torch.version.cuda) < 12.8:
    print("  [x] Blackwell(sm_120) 需要 CUDA >= 12.8"); sys.exit(1)
CHECK

# ---------------------------------------------------------------- 5. 依赖
step "安装项目依赖"
"$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt"

# ---------------------------------------------------------------- 5b. x-transformers
# 不从 PyPI 装：仓库内自带 vendor/x_transformers/（和开发机同一份源码）。
# pip 装会拿到可能更旧的版本，而 v1.1 依赖 attn_flash / attn_qk_norm /
# zero_init_branch_output —— 版本不对会在 build_model 时直接 TypeError。
step "安装 x-transformers（仓库内自带版本）"
VENDOR="$ROOT/vendor"
if [ ! -d "$VENDOR/x_transformers" ]; then
    die "找不到 $VENDOR/x_transformers，仓库不完整（请 git clone 完整仓库）"
fi
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "$SP/x_transformers" "$SP"/x_transformers-*.dist-info 2>/dev/null || true
echo "$VENDOR" > "$SP/_dreamy_vendor.pth"
"$PY" -c "import x_transformers, os; print('  x_transformers:', x_transformers.__file__)"
case "$("$PY" -c 'import x_transformers; print(x_transformers.__file__)')" in
    "$VENDOR"/*) echo "  -> 已指向 vendor/" ;;
    *) die "x-transformers 没有指到 vendor/，请检查 site-packages 里是否有残留版本" ;;
esac

# ---------------------------------------------------------------- 6. 自检
step "环境自检"
cd "$ROOT"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")

import torch

print(f"  torch : {torch.__version__}   cuda: {torch.version.cuda}")
if not torch.cuda.is_available():
    print("  [x] CUDA 不可用"); sys.exit(1)

cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
props = torch.cuda.get_device_properties(0)
print(f"  gpu   : {torch.cuda.get_device_name(0)} | {arch} | "
      f"{props.total_memory/2**30:.0f}GB")

# 实测能不能跑（架构列表只是参考，二进制兼容允许 X.y 跑在 X.z, z>=y）
try:
    xx = torch.randn(512, 512, device="cuda")
    torch.mm(xx, xx)
    torch.cuda.synchronize()
    print("  kernel: OK")
except Exception as e:
    print(f"  kernel: FAILED {str(e)[:120]}"); sys.exit(1)

print(f"  bf16  : {'支持' if torch.cuda.is_bf16_supported() else '不支持'}")

# 构建 v2 模型（默认 127M = dim 832 / depth 12 / heads 13）并检查 flash 是否真的开了
# 想验别的尺寸: DREAMY_DIM=896 DREAMY_DEPTH=12 bash _deploy/setup_5090.sh
import os

from config import ModelConfig
from model import build_model, count_params

dim = int(os.environ.get("DREAMY_DIM", 832))
depth = int(os.environ.get("DREAMY_DEPTH", 12))
heads = int(os.environ.get("DREAMY_HEADS", dim // 64))
cfg = ModelConfig(dim=dim, depth=depth, heads=heads, head_dim=64)
m = build_model(cfg).cuda()
n = count_params(m)
flashes = [getattr(x, "flash", None) for x in m.modules()
           if x.__class__.__name__ == "Attend"]
print(f"  参数量: {n/1e6:.2f}M  (dim {cfg.dim} / {cfg.depth} 层 / {cfg.heads} heads)")
print(f"  注意力: {len(flashes)} 层, flash 全部开启 = {all(flashes)}")
if not all(flashes):
    print("  [x] 融合注意力没生效，训练会慢 2-3 倍"); sys.exit(1)

x = torch.randint(0, cfg.num_tokens, (2, 512), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = m(x)
    loss = logits.float().sum()
loss.backward()
print(f"  前向   : {tuple(logits.shape)}")
print(f"  显存   : {torch.cuda.memory_allocated()/2**30:.2f} GB")
print("  自检通过")
PY

# ---------------------------------------------------------------- 7. 数据
if (( WITH_DATA )); then
    step "下载并构建数据（慢，2-4h；通常更推荐从开发机直接传 tokens.bin）"
    "$PY" scripts/download_data.py --preset full
    "$PY" scripts/prepare_sources.py --stage both
    "$PY" -u scripts/build_dataset_streaming.py --stage pretrain \
        --mix-key pretrain_v2 --tokenize --max-chars 3890000000
else
    step "数据还没准备"
    cat <<EOF
  从开发机传这几组文件到 data/processed/：
    tokens.bin val_tokens.bin meta.json tokenizer.model
    sft_tokens.npy sft_labels.npy sft_val_*.npy sft_meta.json

  传输前先在开发机上核对 md5（tokens.bin 约 4.8GB）：
    python scripts/export_model.py --help   # 有现成的分卷 + 校验逻辑可参考
EOF
fi

step "完成"
cat <<EOF
激活: source "$VENV/bin/activate"

下一步 —— **先跑吞吐基准，再开训练**（它会自己算出 --max-steps）：
    python -u scripts/bench_throughput.py \\
        --dim 832 --depth 12 --heads 13 --head-dim 64 --seq-len 1024 \\
        --batch-size 16 --grad-accum 2 --sweep 8,16,24,32 \\
        --steps 60 --warmup 10 --hours 10 --reserve-hours 1

然后把它打印出来的训练命令复制过去执行。
EOF
