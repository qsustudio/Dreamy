#!/usr/bin/env bash
# setup_4090.sh -- Ubuntu + RTX 4090 (24GB) 部署 Dreamy
#
# 用法:
#   bash _deploy/setup_4090.sh              # 装依赖 + 自检
#   bash _deploy/setup_4090.sh --with-data  # 顺便下载训练数据（约 13GB）
#
# 做完这一步就可以开始训练。数据也可以从别的机器传 tokens.bin 过来，
# 不必在这里重新下载。
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
CUDA_TAG="${CUDA_TAG:-cu126}"      # 4090 = sm_89，cu118 以上都行
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
WITH_DATA=0
[[ "${1:-}" == "--with-data" ]] && WITH_DATA=1

step() { printf '\n\033[36m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. GPU
step "检查 GPU 和驱动"
if ! command -v nvidia-smi >/dev/null; then
    die "没找到 nvidia-smi。请先装好 NVIDIA 驱动。"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
           --format=csv,noheader | sed 's/^/  /'

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "  driver major: $DRIVER"
if [[ -n "$DRIVER" ]] && (( DRIVER < 525 )); then
    die "驱动 $DRIVER 太旧，cu126 需要 525+。请先升级驱动。"
fi

# ---------------------------------------------------------------- 2. Python
step "检查 Python"
command -v python3 >/dev/null || die "没有 python3"
python3 --version
PYVER=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "  版本: $PYVER"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
    || die "需要 Python 3.10+"

# ---------------------------------------------------------------- 3. venv
step "创建虚拟环境"
if [[ ! -x "$PY" ]]; then
    python3 -m venv "$VENV" || die "venv 创建失败。Ubuntu 可能需要: sudo apt install python3-venv"
fi
"$PY" -m pip install --upgrade pip -q
echo "  pip: $("$PY" -m pip --version)"

# ---------------------------------------------------------------- 4. torch
step "安装 PyTorch ${TORCH_VERSION}+${CUDA_TAG}"
if "$PY" -c "import torch" 2>/dev/null; then
    "$PY" -c "import torch; print('  已安装:', torch.__version__)"
else
    "$PY" -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ---------------------------------------------------------------- 5. 依赖
step "安装项目依赖"
"$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt"

# ---------------------------------------------------------------- 6. 自检
step "环境自检"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")

import torch

print(f"  python : {sys.version.split()[0]}")
print(f"  torch  : {torch.__version__}")
print(f"  cuda   : {torch.version.cuda}")

if not torch.cuda.is_available():
    print("  [x] CUDA 不可用 —— 检查驱动和 torch 的 CUDA 版本是否匹配")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    arch = f"sm_{cap[0]}{cap[1]}"
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    # 注意：不能只用 arch in get_arch_list() 判断。
    # CUDA 二进制兼容规则是「X.y 的 kernel 能跑在 X.z (z>=y)」，
    # 4090 是 sm_89，跑 sm_86 的 kernel 完全没问题。
    # 最终判据改成真的跑一次 matmul。
    archs = torch.cuda.get_arch_list()
    compatible = any(
        a.startswith("sm_") and a[3] == arch[3] and int(a[4:].rstrip("abcdef")) <= cap[1]
        for a in archs
    )
    try:
        xx = torch.randn(256, 256, device=f"cuda:{i}")
        torch.mm(xx, xx)
        torch.cuda.synchronize(i)
        run_ok = True
        err = ""
    except Exception as e:
        run_ok = False
        err = str(e)[:120]
    state = "kernel OK" if run_ok else f"kernel FAILED: {err}"
    print(f"  gpu{i}   : {name} | {arch} | {props.total_memory/2**30:.0f}GB | {state}")
    if not run_ok:
        print("  [x] 当前 torch 跑不动这张卡，请装匹配的 CUDA 版本")
        sys.exit(1)

bf16 = torch.cuda.is_bf16_supported()
print(f"  bf16   : {'支持（训练用 bf16，不需要 GradScaler）' if bf16 else '不支持（会回退 fp16）'}")

# 构建真实模型验证
from config import ModelConfig
from model import build_model, count_params

cfg = ModelConfig()
m = build_model(cfg).cuda()
n = count_params(m)
print(f"  参数量 : {n/1e6:.2f}M  (目标 110M)")

x = torch.randint(0, cfg.num_tokens, (2, 512), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
    logits = m(x)
    loss = logits.float().sum()
loss.backward()
grads = sum(1 for p in m.parameters() if p.grad is not None)
total = sum(1 for _ in m.parameters())
print(f"  前向   : {tuple(logits.shape)}")
print(f"  反向   : {grads}/{total} 参数有梯度")
print(f"  显存   : 已分配 {torch.cuda.memory_allocated()/2**30:.2f} GB")
print("  自检通过")
PY

# ---------------------------------------------------------------- 7. 数据
if (( WITH_DATA )); then
    step "下载训练数据（约 13GB）"
    "$PY" scripts/download_data.py --preset full
    step "转换数据格式"
    "$PY" scripts/prepare_sources.py --stage both
    step "构建 token 数据"
    "$PY" scripts/build_dataset.py --stage both --tokenize
else
    step "数据还没准备"
    cat <<EOF
  两种方式任选：

  A. 在本机下载（推荐，避免传 13GB）
       $PY scripts/download_data.py --preset full
       $PY scripts/prepare_sources.py --stage both
       $PY scripts/build_dataset.py --stage both --tokenize

  B. 从开发机传已构建好的 tokens.bin（约 4GB）到 data/processed/
     需要: tokens.bin, val_tokens.bin, meta.json, tokenizer.model
     以及 SFT 的 sft_tokens.npy / sft_labels.npy / sft_meta.json
EOF
fi

step "完成"
cat <<EOF
激活环境:
    source "$VENV/bin/activate"

开始训练:
    $PY src/train.py --stage pretrain --max-steps 200   # 先冒烟
EOF
