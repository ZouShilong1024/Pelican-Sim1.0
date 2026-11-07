#!/usr/bin/env bash
# RoboDojo multi-view inference with the multi-view post-trained MoE (E8/K2) checkpoint.
#
# The head camera and both wrist cameras are packed onto one 480x640 canvas (head full width
# on the top 2/3, left | right wrist on the bottom 1/3), for the RGB and for the ARX-X5
# skeleton video. The skeletons of all three cameras are rendered online from action.npy
# (wrists with their per-frame pose in camera_tracks.npz). The model is a 13-frame model
# (state_t 4); inference rolls out chunk after chunk, each conditioned on the last generated
# frame, until the whole episode is covered (--match-episode-length).
#
#   1. checkpoint: CKPT_PATH (.pt), else JOB_DIR/checkpoints/<ITER or latest> converted once to
#      checkpoints/<job>/<iter>/model_ema_bf16.pt. JOB_DIR is the multi-view post-training run
#      (default), or a `MULTIVIEW=1 ./finetune.sh` job dir.
#   2. samples:    SAMPLE_DIR = multi-view sample folder(s) (comma-separated), or an out dir of
#      `scripts/prepare_robodojo.py --multiview` (every episode in its episodes.jsonl)
#   3. inference:  examples/action_inject_inference_from_dataset.py --state-t 4 --mode ar
#      --match-episode-length --metrics
#
# Usage:
#   ./infer_robodojo_mv.sh                                   # examples/samples/robodojo_mv_hold_button
#   SAMPLE_DIR=data/robodojo_mv_ft NPROC=8 ./infer_robodojo_mv.sh
#   JOB_DIR=outputs/pelican_sim/finetune/robodojo_mv_ft ITER=iter_000010000 ./infer_robodojo_mv.sh
#   CKPT_PATH=/path/to/model_ema_bf16.pt ./infer_robodojo_mv.sh
#   MODE=single ./infer_robodojo_mv.sh                       # one 13-frame chunk only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
fi

VENV_BIN="${VENV_BIN:-$ROOT/.venv/bin}"
export PATH="$VENV_BIN:$PATH"
export PYTHONPATH="$ROOT/packages/cosmos-oss:$ROOT/packages/cosmos-cuda:$ROOT/packages/cosmos-gradio:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
if [[ -z "${HF_HOME:-}" && -d "$ROOT/hf_cache/hub" ]]; then
  HF_HOME="$ROOT/hf_cache"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
fi
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$ROOT/outputs}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

JOB_DIR="${JOB_DIR:-/mnt/jackzou/Pelican-Sim/outputs/cosmos_predict2_action_inject/cosmos_predict_v2p5/2b_action_inject_moe_shared_dense_e8k2_from_i2v_mixed140k_uniembodied_ft_13f_s124_mv_fill}"
EXPERIMENT="${EXPERIMENT:-ac_inject_moe_shared_dense_e8_rectified_flow_2b}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_inject/config.py}"
STATE_T="${STATE_T:-4}"
MODE="${MODE:-ar}"
NPROC="${NPROC:-1}"
GUIDANCE="${GUIDANCE:-3}"
NUM_STEPS="${NUM_STEPS:-35}"
SEED="${SEED:-0}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-1}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/examples/samples/robodojo_mv_hold_button}"

# ---- checkpoint ----
if [[ -n "${CKPT_PATH:-}" ]]; then
  EMA_PT="$CKPT_PATH"
  ITER="${ITER:-$(basename "$(dirname "$EMA_PT")")}"
else
  CKPT_ROOT="$JOB_DIR/checkpoints"
  # latest_checkpoint.txt only moves after a save completes, unlike a still-writing iter_* dir.
  if [[ -z "${ITER:-}" && -f "$CKPT_ROOT/latest_checkpoint.txt" ]]; then
    ITER="$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpoint.txt")"
  fi
  # No job dir (e.g. another machine): newest converted checkpoint under checkpoints/<job>/.
  if [[ -z "${ITER:-}" ]]; then
    latest_pt="$(ls -1 "$ROOT/checkpoints/$(basename "$JOB_DIR")"/iter_*/model_ema_bf16.pt 2>/dev/null | sort -V | tail -n 1)"
    [[ -n "$latest_pt" ]] && ITER="$(basename "$(dirname "$latest_pt")")"
  fi
  CACHE_DIR="$ROOT/checkpoints/$(basename "$JOB_DIR")/${ITER:-none}"
  EMA_PT="$CACHE_DIR/model_ema_bf16.pt"
  if [[ ! -f "$EMA_PT" && ( -z "${ITER:-}" || ! -d "$CKPT_ROOT/$ITER/model" ) ]]; then
    echo "ERROR: no checkpoint under $CKPT_ROOT or $CACHE_DIR (ITER=${ITER:-})" >&2
    exit 1
  fi
  if [[ ! -f "$EMA_PT" ]]; then
    echo "=== Converting DCP -> $EMA_PT ==="
    mkdir -p "$CACHE_DIR"
    "$VENV_BIN/python" scripts/convert_distcp_to_pt.py "$CKPT_ROOT/$ITER/model" "$CACHE_DIR"
    rm -f "$CACHE_DIR/model.pt" "$CACHE_DIR/model_ema_fp32.pt"
  fi
fi
[[ -f "$EMA_PT" ]] || { echo "ERROR: checkpoint not found: $EMA_PT" >&2; exit 1; }

# ---- samples ----
SAMPLE_DIRS=""
IFS=',' read -r -a given <<< "$SAMPLE_DIR"
for d in "${given[@]}"; do
  [[ -z "$d" ]] && continue
  if [[ -f "$d/episodes.jsonl" ]]; then
    d="$("$VENV_BIN/python" -c 'import json, os, sys
print(",".join(os.path.dirname(json.loads(l)["rgb"]) for l in open(sys.argv[1]) if l.strip() and "views" in json.loads(l)))' \
      "$d/episodes.jsonl")"
    [[ -n "$d" ]] || { echo "ERROR: no multi-view episodes in $SAMPLE_DIR (prepare_robodojo.py --multiview)" >&2; exit 1; }
  fi
  SAMPLE_DIRS="${SAMPLE_DIRS:+$SAMPLE_DIRS,}$d"
done

OUT_DIR="${OUT_DIR:-$ROOT/outputs/infer_robodojo_mv_${MODE}_${ITER}_g${GUIDANCE}_seed${SEED}}"
mkdir -p "$OUT_DIR"
LOG="${LOG:-$OUT_DIR/infer.log}"
echo "=== RoboDojo multi-view inference MODE=$MODE ITER=$ITER CKPT=$EMA_PT OUT=$OUT_DIR ===" | tee "$LOG"

common_args=(
  examples/action_inject_inference_from_dataset.py
  --job-dir "$JOB_DIR"
  --ckpt-path "$EMA_PT"
  --experiment "$EXPERIMENT"
  --config-file "$CONFIG_FILE"
  --state-t "$STATE_T"
  --sample-dir "$SAMPLE_DIRS"
  --render-action-video "${RENDER_ACTION_VIDEO:-auto}"
  --mode "$MODE"
  --chunk-overlap "$CHUNK_OVERLAP"
  --seed "$SEED"
  --guidance "$GUIDANCE"
  --num-steps "$NUM_STEPS"
  --resolution 480,640
  --save-dir "$OUT_DIR"
  --metrics
)
if [[ "$MODE" == "ar" ]]; then
  common_args+=(--match-episode-length)
fi

if [[ "$NPROC" -gt 1 ]]; then
  GPU_IDS=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  else
    for ((i=0; i<NPROC; i++)); do GPU_IDS+=("$i"); done
  fi
  pids=()
  for ((r=0; r<NPROC; r++)); do
    (
      export CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}"
      unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK || true
      "$VENV_BIN/python" "${common_args[@]}" --context-parallel-size 1 \
        --data-parallel --data-parallel-rank "$r" --data-parallel-world-size "$NPROC" \
        >>"${LOG}.rank${r}" 2>&1
    ) &
    pids+=($!)
  done
  INFER_EXIT=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then INFER_EXIT=1; fi
  done
else
  "$VENV_BIN/python" "${common_args[@]}" --context-parallel-size 1 2>&1 | tee -a "$LOG"
  INFER_EXIT=${PIPESTATUS[0]}
fi

if [[ -s "$OUT_DIR/inference_results.jsonl" ]]; then
  "$VENV_BIN/python" scripts/summarize_infer_metrics.py "$OUT_DIR/inference_results.jsonl" | tee -a "$LOG"
fi
echo "INFER_EXIT=${INFER_EXIT}" | tee -a "$LOG"
echo "Outputs under: $OUT_DIR"
exit "${INFER_EXIT}"
