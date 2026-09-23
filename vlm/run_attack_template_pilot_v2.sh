#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate attn-tracker
cd /root/fs/projects/qwen3_vl_instruction_attention
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
for TEMPLATE in action-aligned-v2 priority-override-v1 direct-command-v1; do
  OUT=outputs_mind2web_pilot_${TEMPLATE}_5_${STAMP}
  LOG=logs/${OUT}.log
  STATUS=logs/${OUT}.status
  {
    echo "OUT=$OUT"
    echo "ATTACK_TEMPLATE=$TEMPLATE"
    echo "SAMPLING_STRATEGY=stratified-task"
    echo "PHASE_A_PREPARE_BEGIN"
    python mind2web_visual_agent_phase1.py       --parquet /root/fs/projects/qwen3_vl_instruction_attention/mind2web_data/data/test_domain-00000-of-00011-26c55c12cbbcdc8e.parquet       --output-dir "$OUT"       --max-samples 5       --calibration-samples 2       --test-samples 3       --sampling-strategy stratified-task       --attack-template "$TEMPLATE"       --prepare-only
    echo "PHASE_A_PREPARE_DONE"
    export HF_HOME=/root/fs/ai/hf-home
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    echo "PHASE_B_QWEN_BEGIN"
    /root/fs/ai/venvs/qwen3-vl/bin/python mind2web_visual_agent_phase1.py       --prepared-manifest "$OUT/prepared_manifest.json"       --output-dir "$OUT"       --model /root/fs/ai/models/Qwen3-VL-8B-Instruct       --max-samples 5       --calibration-samples 2       --test-samples 3       --sampling-strategy stratified-task       --attack-template "$TEMPLATE"       --max-new-tokens 64       --max-pixels 112896       --skip-phase1
    echo "PHASE_B_QWEN_DONE"
    echo "OK" > "$STATUS"
  } > "$LOG" 2>&1
done
