#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — Mini-Attention environment setup
#
# Detects GPU compute capability and installs the correct PyTorch build.
# Tested on: RTX 5090 (SM120), RTX 4090 (SM89), H100 (SM90), B200 (SM100)
#
# Usage (after cloning the repo):
#   bash setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "=== Mini-Attention setup ==="
echo "Repo: $REPO_DIR"

# ---------------------------------------------------------------------------
# 1. Detect GPU compute capability
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi

# Returns e.g. "12.0" for RTX 5090, "9.0" for H100, "8.9" for RTX 4090
SM_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
SM_MAJOR=$(echo "$SM_RAW" | cut -d. -f1)
SM_MINOR=$(echo "$SM_RAW" | cut -d. -f2)
SM_INT=$((SM_MAJOR * 10 + SM_MINOR))   # e.g. 120, 90, 89, 86

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME  (SM${SM_INT} / compute_cap=${SM_RAW})"

# ---------------------------------------------------------------------------
# 2. Pick PyTorch index URL based on SM
# ---------------------------------------------------------------------------
# SM120 (RTX 5090 / consumer Blackwell) — needs torch>=2.11.0+cu128
# SM100 (B200 / datacenter Blackwell)   — needs torch>=2.11.0+cu128
# SM90  (H100 / Hopper)                 — cu128 works; cu124 fine too
# SM89  (RTX 4090 / Ada)                — cu124
# SM86  (RTX 3060/3090 / Ampere)        — cu124
if [ "$SM_INT" -ge 100 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_REQ="torch>=2.11.0"
    echo "→ Using torch+cu128 (SM${SM_INT} requires CUDA 12.8+ runtime)"
elif [ "$SM_INT" -ge 90 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_REQ="torch>=2.5.0"
    echo "→ Using torch+cu128 (SM${SM_INT} Hopper)"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_REQ="torch>=2.5.0"
    echo "→ Using torch+cu124 (SM${SM_INT})"
fi

# ---------------------------------------------------------------------------
# 3. Check CUDA toolkit (nvcc)
# ---------------------------------------------------------------------------
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    echo "CUDA toolkit: $CUDA_VER"
else
    echo "WARNING: nvcc not found — JIT kernel compilation will fail."
    echo "  Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
fi

# Ninja speeds up torch.utils.cpp_extension.load significantly
if ! command -v ninja &>/dev/null; then
    echo "Installing ninja-build..."
    apt-get install -y ninja-build -qq 2>/dev/null || pip install ninja -q
fi

# ---------------------------------------------------------------------------
# 4. Python check
# ---------------------------------------------------------------------------
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3)
PY_VER=$("$PYTHON" --version 2>&1)
echo "Python: $PY_VER ($PYTHON)"

# ---------------------------------------------------------------------------
# 5. Create venv
# ---------------------------------------------------------------------------
echo ""
echo "=== Creating virtual environment at $VENV_DIR ==="
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ---------------------------------------------------------------------------
# 6. Install PyTorch first (large wheel, architecture-specific)
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing PyTorch ($TORCH_REQ from $TORCH_INDEX) ==="
pip install "$TORCH_REQ" --index-url "$TORCH_INDEX" -q

# ---------------------------------------------------------------------------
# 7. Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing project dependencies ==="

# vllm is optional and heavy — skip if it fails (not available for all CUDA builds)
pip install matplotlib numpy transformers triton -q

if pip install vllm -q 2>/dev/null; then
    echo "vllm installed OK"
else
    echo "vllm skipped (not available for this torch/CUDA combination — optional)"
fi

# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "=== Smoke test ==="
python - <<EOF
import torch
print(f"torch:      {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU:        {dev.name}  SM{dev.major}{dev.minor}")
    print(f"VRAM:       {dev.total_memory // 1024**3} GB")
    x = torch.ones(4, device="cuda")
    print(f"Tensor OK:  {x.sum().item()}")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run SM120 benchmark (RTX 5090):"
echo "  python kernels/flash_attn/cuda/sm_120/_bench_sm120.py"
echo ""
echo "To run SM86 benchmark (RTX 3060/3090):"
echo "  TORCH_CUDA_ARCH_LIST=8.6 python kernels/flash_attn/cuda/sm_86/_bench_sm86.py"
