#!/usr/bin/env bash
# setup_t4.sh -- deploy x-transformers on a Linux T4 box (Google Colab / GCE / Kaggle)
#
# T4 is Turing, compute capability sm_75. PyTorch's CUDA wheels all support it,
# but the driver matters:
#   * CUDA 12.6 wheels (torch 2.14.x)  -> driver 525+   <- safest for T4 images
#   * CUDA 13.x wheels (torch 2.15)    -> driver 580+   (T4 itself is still fine)
#
# Usage:
#   bash setup_t4.sh                    # torch 2.14.1 + cu126 (recommended)
#   CUDA=cu132 bash setup_t4.sh         # newer stack, needs driver >= 580
#   ROOT=/content/dreamy bash setup_t4.sh
#
# In a Colab notebook, prefix with a bash cell:
#   !bash setup_t4.sh
# then put the venv's python first on PATH (see the end of this script).

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CUDA="${CUDA:-cu126}"
TORCH_VERSION="${TORCH_VERSION:-}"      # empty -> newest build on that CUDA index
REPO_URL="${REPO_URL:-https://github.com/lucidrains/x-transformers.git}"
SKIP_TESTS="${SKIP_TESTS:-0}"

REPO_DIR="$ROOT/x-transformers"
VENV_DIR="$ROOT/.venv-t4"
PY="$VENV_DIR/bin/python"

step() { printf '\n=== %s ===\n' "$*"; }

step "Environment"
echo "root      : $ROOT"
echo "cuda index: $CUDA"
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
python3 --version
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
               --format=csv,noheader || true
else
    echo "nvidia-smi: not found (no GPU? the install still works)"
fi

# ------------------------------------------------------------------ clone
step "Clone x-transformers"
if [ -d "$REPO_DIR/.git" ]; then
    echo "already cloned, skipping"
else
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# ------------------------------------------------------------------ venv
step "Create virtual environment"
if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$PY" -m pip install --upgrade pip

# ------------------------------------------------------------------ torch
step "Install PyTorch ($CUDA)"
if [ -z "$TORCH_VERSION" ]; then
    TORCH_VERSION=$("$PY" - <<PY
import re, urllib.request
url = "https://download.pytorch.org/whl/$CUDA/torch/"
html = urllib.request.urlopen(url, timeout=60).read().decode()
vers = set()
for m in re.finditer(r'torch-([0-9][^-]*?)%2B$CUDA-[^-]+-[^-]+-(?:manylinux\w*|linux)_x86_64\.whl', html):
    vers.add(m.group(1))
if not vers:
    raise SystemExit("no linux wheel for $CUDA at " + url)
print(max(vers, key=lambda v: [int(x) for x in v.split(".") if x.isdigit()]))
PY
)
fi
echo "torch version: $TORCH_VERSION"

"$PY" -m pip install --no-cache-dir \
    "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA"

# ------------------------------------------------------------------ package
step "Install x-transformers (editable) + test deps"
"$PY" -m pip install --no-cache-dir --no-build-isolation -e "$REPO_DIR"
"$PY" -m pip install --no-cache-dir pytest

# ------------------------------------------------------------------ verify
step "Smoke test"
"$PY" - <<'PY'
import importlib.metadata as md
import torch
from x_transformers import TransformerWrapper, Decoder

print("torch       :", torch.__version__)
print("cuda build  :", torch.version.cuda)
print("x-transform :", md.version("x-transformers"))

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = "sm_%d%d" % cap
        name = torch.cuda.get_device_name(i)
        ok = arch in torch.cuda.get_arch_list()
        print("gpu %d       : %s | %s | built for %s -> %s"
              % (i, name, arch, arch, "OK" if ok else "NOT COMPILED IN"))
    if "sm_75" not in torch.cuda.get_arch_list():
        print("WARNING: this torch build has no sm_75 kernel (T4 needs sm_75)")
else:
    print("cuda        : NOT available (driver mismatch or no GPU)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = TransformerWrapper(num_tokens=256, max_seq_len=64,
                       attn_layers=Decoder(dim=128, depth=4, heads=8)).to(device)
x = torch.randint(0, 256, (4, 64), device=device)
logits = m(x)
logits.sum().backward()
print("forward     :", tuple(logits.shape), "on", device)
print("grads       :", sum(p.grad is not None for p in m.parameters()), "/",
      sum(1 for _ in m.parameters()))
if device.type == "cuda":
    torch.cuda.synchronize()
    print("peak mem    : %.1f MB" % (torch.cuda.max_memory_allocated() / 2**20))
PY

if [ "$SKIP_TESTS" != "1" ]; then
    step "Test suite"
    ( cd "$ROOT" && "$PY" -m pytest "$REPO_DIR/tests" -q -p no:cacheprovider \
        -W ignore::FutureWarning --no-header ) || \
        echo "WARNING: some tests failed (see _deploy/README.md)"
fi

step "Done"
cat <<EOF

Activate with:
    source "$VENV_DIR/bin/activate"

In Colab, this persists only for the current runtime. To make the kernel use
this venv:
    import sys, subprocess
    sys.path.insert(0, "$VENV_DIR/lib/python$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')/site-packages")
or simply run your training via the venv's python:
    "$PY" $REPO_DIR/train_enwik8.py
EOF
