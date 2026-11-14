#!/bin/bash
# Build the python environment for train_nash_sana.py on a fresh GPU worker.
# The worker image already has torch (system python); we add a venv on top of it.
set -euo pipefail

VENV="${VENV:-$HOME/sana-venv}"
# One setup at a time per machine (a previous attempt's setup may still be running).
exec 9>/tmp/sana_setup.lock
flock 9
if [ ! -x "$VENV/bin/python" ]; then
  # No apt here: installing python3-venv upgrades the system python underneath the running Ray
  # worker, which killed the Ray actor once. `virtualenv` from pip needs no system packages.
  rm -rf "$VENV"
  python3 -m venv --system-site-packages "$VENV" 2>/dev/null || {
    rm -rf "$VENV"
    python3 -m pip install -q --user virtualenv
    python3 -m virtualenv -q --system-site-packages "$VENV"
  }
fi
source "$VENV/bin/activate"

if [ ! -f "$VENV/.sana_ready" ]; then
  pip install -q --upgrade pip
  pip install -q \
    "diffusers==0.35.2" "transformers==4.49.0" "accelerate>=1.0" "huggingface_hub<1.0" \
    safetensors sentencepiece protobuf datasets ftfy tensorboard \
    clint timm einops fairscale "numpy<2"
  # reward models (their own pinned requirements are stale, so install without deps)
  pip install -q --no-deps hpsv2 image-reward
  pip install -q "git+https://github.com/openai/CLIP.git"
  touch "$VENV/.sana_ready"
fi
python - <<'EOF'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "diffusers", diffusers.__version__, "transformers", transformers.__version__)
EOF
