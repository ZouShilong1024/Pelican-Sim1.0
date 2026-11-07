#!/usr/bin/env bash
# Action-inject MoE inference (teacher 35-step, single-chunk or AR).
#
# Usage:
#   ./infer.sh [num_samples] [dataset] [seed] [guidance]
#
# Examples:
#   ./infer.sh                                   # every folder in examples/samples/manifest.txt
#   ./infer.sh 1 libero 0 3                      # sample from a local training dataset
#   MODE=single NPROC=1 ./infer.sh 1 libero 0 3
#   CKPT_PATH=/path/to/model_ema_bf16.pt NPROC=1 ./infer.sh 1 libero 0 3
#   JOB_DIR=outputs/pelican_sim/finetune/<job> ITER=iter_000010000 SAMPLE_DIR=... ./infer.sh
#
# Checkpoint lookup order:
#   1. CKPT_PATH (consolidated .pt)
#   2. checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt (Hugging Face release), unless ITER is set
#   3. JOB_DIR/checkpoints/<ITER or latest>/model_ema_bf16.pt (fine-tune job dir; DCP converted once)
#
# Site-specific env (proxy, HF_HOME, ...) can live in an untracked .env.local.
#
# Public example pack (Libero-style folders under examples/samples/).
#   SAMPLE_DIR=examples/samples/libero_spatial_top_drawer NPROC=1 MODE=single ./infer.sh
#   MODE=ar MATCH_EPISODE_LENGTH=1 ./infer.sh    # AR, match episode length
#   RENDER_ACTION_VIDEO=always NPROC=1 ./infer.sh 1 libero 0 3
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
# Release bundle: pretrained text encoder / VAE ship in hf_cache/ and resolve offline.
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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

JOB_DIR="${JOB_DIR:-$ROOT/checkpoints/Pelican-Sim-1.0}"
CKPT_ROOT="$JOB_DIR/checkpoints"
# No dataset / sample folder given: run the bundled example pack.
if [[ -z "${SAMPLE_DIR:-}" && -z "${EPISODE_MANIFEST:-}" && -z "${2:-${DATASET:-}}" ]]; then
  SAMPLE_DIR="$(tr '\n' ',' < "$ROOT/examples/samples/manifest.txt" | sed 's/,$//')"
  DATA_TAG=samples
fi
HF_RELEASE_PT="$ROOT/checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt"
MODE="${MODE:-ar}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-185}"
CHUNK_SIZE="${CHUNK_SIZE:-93}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-1}"
NUM_SAMPLES="${1:-${NUM_SAMPLES:-1}}"
DATASET="${2:-${DATASET:-libero}}"
SEED="${3:-${SEED:-0}}"
GUIDANCE="${4:-${GUIDANCE:-3}}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
NPROC="${NPROC:-1}"
NUM_STEPS="${NUM_STEPS:-35}"
RESOLUTION="${RESOLUTION:-480,640}"
FPS="${FPS:-15}"
SAVE_FPS="${SAVE_FPS:-${FPS}}"
MATCH_EPISODE_LENGTH="${MATCH_EPISODE_LENGTH:-1}"
VIDEO_STYLE="${VIDEO_STYLE:-premium}"
EXPERIMENT="${EXPERIMENT:-ac_inject_moe_shared_dense_e8_rectified_flow_2b}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_inject/config.py}"

if [[ -z "${CKPT_PATH:-}" && -z "${ITER:-}" && -f "$HF_RELEASE_PT" ]]; then
  CKPT_PATH="$HF_RELEASE_PT"
fi

if [[ -n "${CKPT_PATH:-}" ]]; then
  EMA_PT="$CKPT_PATH"
  if [[ ! -f "$EMA_PT" ]]; then
    echo "ERROR: CKPT_PATH not found: $EMA_PT"
    exit 1
  fi
  ITER="${ITER:-$(basename "$(dirname "$EMA_PT")")}"
else
  if [[ -n "${ITER:-}" ]]; then
    :
  elif [[ -f "$CKPT_ROOT/latest_checkpoint.txt" && -d "$CKPT_ROOT/$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpoint.txt")" ]]; then
    ITER="$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpoint.txt")"
  else
    ITER="$(basename "$(ls -1d "$CKPT_ROOT"/iter_* 2>/dev/null | sort -V | tail -n 1)")"
  fi
  if [[ -z "${ITER:-}" ]]; then
    echo "ERROR: no checkpoint found. Download the release (see README) or set CKPT_PATH / JOB_DIR."
    exit 1
  fi

  CHECKPOINT_DIR="$CKPT_ROOT/$ITER"
  EMA_PT="$CHECKPOINT_DIR/model_ema_bf16.pt"
  if [[ ! -f "$EMA_PT" && -d "$CHECKPOINT_DIR/model" ]]; then
    echo "=== Converting DCP -> pt: $CHECKPOINT_DIR ==="
    "$VENV_BIN/python" scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"
  fi
  if [[ ! -f "$EMA_PT" ]]; then
    echo "ERROR: expected $EMA_PT"
    exit 1
  fi
fi

if [[ "${MATCH_EPISODE_LENGTH}" == "1" ]]; then
  LEN_TAG="matchlen"
else
  LEN_TAG="f${NUM_OUTPUT_FRAMES}"
fi
OUT_DIR="${OUT_DIR:-$ROOT/outputs/infer_${DATA_TAG:-$DATASET}_${MODE}_${ITER}_n${NUM_SAMPLES}_${LEN_TAG}_seed${SEED}_g${GUIDANCE}_s${SAMPLE_SEED}}"
mkdir -p "$OUT_DIR" "$ROOT/logs"
LOG="${LOG:-$OUT_DIR/infer.log}"

echo "=== Pelican-Sim-Infer MODE=$MODE DATASET=$DATASET N=$NUM_SAMPLES ===" | tee "$LOG"
echo "ITER=$ITER CKPT=$EMA_PT OUT=$OUT_DIR" | tee -a "$LOG"
echo "RENDER_ACTION_VIDEO=${RENDER_ACTION_VIDEO:-auto} SAMPLE_DIR=${SAMPLE_DIR:-}" | tee -a "$LOG"

common_args=(
  examples/action_inject_inference_from_dataset.py
  --job-dir "$JOB_DIR"
  --ckpt-path "$EMA_PT"
  --experiment "$EXPERIMENT"
  --config-file "$CONFIG_FILE"
  --dataset "$DATASET"
  --num-samples "$NUM_SAMPLES"
  --sample-seed "$SAMPLE_SEED"
  --mode "$MODE"
  --num-output-frames "$NUM_OUTPUT_FRAMES"
  --chunk-size "$CHUNK_SIZE"
  --chunk-overlap "$CHUNK_OVERLAP"
  --seed "$SEED"
  --guidance "$GUIDANCE"
  --num-steps "$NUM_STEPS"
  --resolution "$RESOLUTION"
  --fps "$FPS"
  --save-fps "$SAVE_FPS"
  --save-dir "$OUT_DIR"
  --video-style "$VIDEO_STYLE"
)
if [[ "${MATCH_EPISODE_LENGTH}" == "1" ]]; then
  common_args+=(--match-episode-length)
else
  common_args+=(--no-match-episode-length)
fi
if [[ -n "${EPISODE_MANIFEST:-}" ]]; then
  common_args+=(--episode-manifest "$EPISODE_MANIFEST")
fi
if [[ -n "${SAMPLE_DIR:-}" ]]; then
  common_args+=(--sample-dir "$SAMPLE_DIR")
fi
common_args+=(--render-action-video "${RENDER_ACTION_VIDEO:-auto}")

if [[ "${NPROC}" -gt 1 ]]; then
  GPU_IDS=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  else
    for ((i=0; i<NPROC; i++)); do GPU_IDS+=("$i"); done
  fi
  pids=()
  for ((r=0; r<NPROC; r++)); do
    gpu="${GPU_IDS[$r]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK || true
      "$VENV_BIN/python" "${common_args[@]}" \
        --context-parallel-size 1 \
        --data-parallel \
        --data-parallel-rank "$r" \
        --data-parallel-world-size "$NPROC" \
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

echo "INFER_EXIT=${INFER_EXIT}" | tee -a "$LOG"
echo "Outputs under: $OUT_DIR"
exit "${INFER_EXIT}"
