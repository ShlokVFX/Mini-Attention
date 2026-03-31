#!/usr/bin/env bash
# ============================================================
# entrypoint.sh — Mini-Attention B200 (SM_100a) auto-setup
# Stored in repo root; executed by vast.ai --onstart-cmd
# ============================================================
set -euo pipefail

REPO=/workspace/Mini-Attention
VENV=$REPO/.venv
LOGFILE=/tmp/b200_compile.log

# ---- 1. Clone / update repo --------------------------------
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch origin wip-b200-kernel
    git -C "$REPO" checkout wip-b200-kernel
    git -C "$REPO" pull --ff-only
else
    git clone -b wip-b200-kernel \
        https://github.com/ShlokVFX/Mini-Attention.git "$REPO"
fi

# ---- 2. Ninja (speeds up JIT compilation) ------------------
command -v ninja &>/dev/null || apt-get install -y ninja-build -qq

# ---- 3. Python venv + PyTorch cu128 (required for SM_100) --
if [ ! -f "$VENV/bin/activate" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install "torch>=2.11.0" --index-url https://download.pytorch.org/whl/cu128 -q
pip install numpy matplotlib -q

# ---- 4. Pre-warm JIT compile (tcgen05 kernel) --------------
# Runs in background — SSH is usable immediately.
# Monitor with: tail -f /tmp/b200_compile.log
cat > /tmp/warmup.py << 'PYEOF'
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
from pathlib import Path
from torch.utils.cpp_extension import load

HERE = Path("/workspace/Mini-Attention/kernels/flash_attn/cuda/sm_100")
sys.path.insert(0, str(HERE / "py"))

load(
    name="b200_warmup",
    sources=[str(HERE / "kernel/flash_attention.cu")],
    extra_include_paths=[str(HERE / "kernel/include")],
    extra_cuda_cflags=[
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=["-O3"],
    build_directory="/tmp/b200_build",
    verbose=True,
)
print("COMPILE OK")
PYEOF

mkdir -p /tmp/b200_build
nohup bash -c "source $VENV/bin/activate && python /tmp/warmup.py" \
    > "$LOGFILE" 2>&1 &

echo "Kernel compiling in background — monitor: tail -f $LOGFILE"

# ---- 5. Activate venv on login -----------------------------
grep -qxF "source $VENV/bin/activate" ~/.bashrc || \
    echo "source $VENV/bin/activate" >> ~/.bashrc

echo ""
echo "Setup done. When compile finishes:"
echo "  source $VENV/bin/activate"
echo "  cd $REPO/kernels/flash_attn/cuda/sm_100"
echo "  python _bench_b200.py"
