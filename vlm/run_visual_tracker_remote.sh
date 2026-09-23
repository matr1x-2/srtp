#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/root/fs/ai/hf-home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/root/fs/projects/qwen3_vl_instruction_attention
PYTHON=/root/fs/ai/venvs/qwen3-vl/bin/python

cd "$PROJECT_DIR"
exec "$PYTHON" run_visual_attention_tracker.py "$@"
