#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate attn-tracker
cd /root/fs/projects/qwen3_vl_instruction_attention
mkdir -p logs
OUT=outputs_mind2web_visual_10_$(date +%Y%m%d_%H%M%S)
LOG=logs/${OUT}.log
PIDFILE=logs/${OUT}.pid
STATUS=logs/${OUT}.status
{
  echo "OUT=$OUT"
  echo "LOG=$LOG"
  echo "STATUS=$STATUS"
  echo "START=$(date -Is)"
  echo "PHASE_A_PREPARE_BEGIN"
  python mind2web_visual_agent_phase1.py \
    --parquet /root/fs/projects/qwen3_vl_instruction_attention/mind2web_data/data/test_domain-00000-of-00011-26c55c12cbbcdc8e.parquet \
    --output-dir "$OUT" \
    --max-samples 10 \
    --calibration-samples 5 \
    --test-samples 5 \
    --prepare-only
  echo "PHASE_A_PREPARE_DONE"
  export HF_HOME=/root/fs/ai/hf-home
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  echo "PHASE_B_QWEN_BEGIN"
  /root/fs/ai/venvs/qwen3-vl/bin/python mind2web_visual_agent_phase1.py \
    --prepared-manifest "$OUT/prepared_manifest.json" \
    --output-dir "$OUT" \
    --model /root/fs/ai/models/Qwen3-VL-8B-Instruct \
    --max-samples 10 \
    --calibration-samples 5 \
    --test-samples 5 \
    --top-heads 8 \
    --max-new-tokens 64 \
    --max-pixels 112896
  echo "PHASE_B_QWEN_DONE"
  echo "END=$(date -Is)"
  echo "OK" > "$STATUS"
} > "$LOG" 2>&1

