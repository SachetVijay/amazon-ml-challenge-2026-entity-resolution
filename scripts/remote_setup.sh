#!/usr/bin/env bash
# Run ON the remote (compute machine WSL or AWS box) from ~/amlc. User-level only: no sudo, no system packages.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
cd ~/amlc
uv venv -q --python 3.12 .venv
uv pip install -q -e ".[gpu]"
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
