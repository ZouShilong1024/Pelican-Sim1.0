#!/usr/bin/env bash
# RoboDojo inference with the RoboDojo fine-tuned MoE (E8/K2) checkpoint.
#
#   1. checkpoint: CKPT_PATH (.pt), else JOB_DIR/checkpoints/<ITER or latest> converted once to
#      checkpoints/<job>/<iter>/model_ema_bf16.pt (JOB_DIR = a finetune.sh job dir, or by default
#      the bundled RoboDojo fine-tune under checkpoints/)
#   2. samples:    examples/pack_robodojo_samples.py → SAMPLES_DIR/<name>/ (first_frame.png,
#      rgb.mp4, action.npy, prompt.txt, meta.json with temporal_stride + camera / arm-base geometry)
#   3. inference:  examples/action_inject_inference_from_dataset.py --sample-dir ... --metrics;
#      the ARX-X5 action video is rendered online from action.npy (RENDER_ACTION_VIDEO=auto|always|never;
#      WITH_ACTION_VIDEO=1 also packs the pre-rendered training mp4, which auto then prefers)
#
# Usage:
#   ./infer_robodojo.sh                                  # 24 scenes × 1 episode, single 93-frame chunk
#   SAMPLE_DIR=outputs/robodojo_samples/<one folder> ./infer_robodojo.sh
#   EPISODES=250,1050 ./infer_robodojo.sh
#   SCENES=fold_clothes,push_T PER_SCENE=2 ./infer_robodojo.sh
#   MODE=ar ./infer_robodojo.sh                          # autoregressive over the whole episode
#   NPROC=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./infer_robodojo.sh
#   ITER=iter_000072000 ./infer_robodojo.sh
#   CKPT_PATH=/path/to/model_ema_bf16.pt ./infer_robodojo.sh
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
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

JOB_DIR="${JOB_DIR:-$ROOT/checkpoints/2b_action_inject_moe_shared_dense_e8k2_from_i2v_mixed140k_uniembodied_ft}"
EXPERIMENT="${EXPERIMENT:-ac_inject_moe_shared_dense_e8_rectified_flow_2b}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_inject/config.py}"
MODE="${MODE:-single}"
NPROC="${NPROC:-1}"
GUIDANCE="${GUIDANCE:-3}"
NUM_STEPS="${NUM_STEPS:-35}"
SEED="${SEED:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-93}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-1}"
SAMPLES_DIR="${SAMPLES_DIR:-$ROOT/outputs/robodojo_samples}"

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
  if [[ -z "${ITER:-}" ]]; then
    ITER="$(basename "$(ls -1d "$CKPT_ROOT"/iter_* 2>/dev/null | sort -V | tail -n 1)")"
  fi
  # No job dir (release bundle): newest converted checkpoint under checkpoints/<job>/.
  if [[ -z "${ITER:-}" ]]; then
    latest_pt="$(ls -1 "$ROOT/checkpoints/$(basename "$JOB_DIR")"/iter_*/model_ema_bf16.pt 2>/dev/null | sort -V | tail -n 1)"
    [[ -n "$latest_pt" ]] && ITER="$(basename "$(dirname "$latest_pt")")"
  fi
  CACHE_DIR="$ROOT/checkpoints/$(basename "$JOB_DIR")/${ITER:-none}"
  EMA_PT="$CACHE_DIR/model_ema_bf16.pt"
  # A converted cache stays usable after training rotates the DCP away.
  if [[ ! -f "$EMA_PT" && ( -z "$ITER" || ! -d "$CKPT_ROOT/$ITER/model" ) ]]; then
    echo "ERROR: no DCP checkpoint under $CKPT_ROOT (ITER=${ITER:-})" >&2
    exit 1
  fi
  if [[ ! -f "$EMA_PT" ]]; then
    echo "=== Converting DCP -> $EMA_PT ==="
    mkdir -p "$CACHE_DIR"
    "$VENV_BIN/python" scripts/convert_distcp_to_pt.py "$CKPT_ROOT/$ITER/model" "$CACHE_DIR"
    rm -f "$CACHE_DIR/model.pt" "$CACHE_DIR/model_ema_fp32.pt"
  fi
fi
if [[ ! -f "$EMA_PT" ]]; then
  echo "ERROR: checkpoint not found: $EMA_PT" >&2
  exit 1
fi

# ---- samples ----
# SAMPLE_DIR=<dir>[,<dir>...] runs existing sample folders; SKIP_PACK=1 uses
# SAMPLES_DIR/sample_dirs.txt as is (e.g. the wrong-action control set). SKIP_PACK=auto (default)
# reuses an existing sample_dirs.txt unless a selection (EPISODES, SCENES, ...) is given, so the
# bundled samples run without the raw RoboDojo dataset.
SKIP_PACK="${SKIP_PACK:-auto}"
if [[ "$SKIP_PACK" == "auto" ]]; then
  SKIP_PACK=0
  if [[ -f "$SAMPLES_DIR/sample_dirs.txt" && -z "${EPISODES:-}${SCENES:-}${PER_SCENE:-}${MAX_FRAMES:-}${PAIRS_JSONL:-}" \
        && "${WITH_ACTION_VIDEO:-0}" != "1" ]]; then
    SKIP_PACK=1
  fi
fi
if [[ -n "${SAMPLE_DIR:-}" ]]; then
  SAMPLE_DIRS="$SAMPLE_DIR"
elif [[ "$SKIP_PACK" != "1" ]]; then
  pack_args=(--out-dir "$SAMPLES_DIR" --per-scene "${PER_SCENE:-1}")
  [[ -n "${EPISODES:-}" ]] && pack_args+=(--episodes "$EPISODES")
  [[ -n "${SCENES:-}" ]] && pack_args+=(--scenes "$SCENES")
  [[ -n "${MAX_FRAMES:-}" ]] && pack_args+=(--max-frames "$MAX_FRAMES")
  [[ -n "${PAIRS_JSONL:-}" ]] && pack_args+=(--pairs-jsonl "$PAIRS_JSONL")
  [[ "${WITH_ACTION_VIDEO:-0}" == "1" ]] && pack_args+=(--with-action-video)
  "$VENV_BIN/python" examples/pack_robodojo_samples.py "${pack_args[@]}"
fi
if [[ -z "${SAMPLE_DIRS:-}" ]]; then
  # Entries are absolute; after moving the samples, fall back to SAMPLES_DIR/<basename>.
  IFS=',' read -r -a listed <<< "$(tr -d '\n' < "$SAMPLES_DIR/sample_dirs.txt")"
  resolved=()
  for d in "${listed[@]}"; do
    [[ -z "$d" ]] && continue
    [[ -d "$d" ]] || d="$SAMPLES_DIR/$(basename "$d")"
    resolved+=("$d")
  done
  SAMPLE_DIRS="$(IFS=','; echo "${resolved[*]}")"
fi

OUT_DIR="${OUT_DIR:-$ROOT/outputs/infer_robodojo_${MODE}_${ITER}_g${GUIDANCE}_seed${SEED}}"
mkdir -p "$OUT_DIR"
LOG="${LOG:-$OUT_DIR/infer.log}"
echo "=== RoboDojo inference MODE=$MODE ITER=$ITER CKPT=$EMA_PT OUT=$OUT_DIR ===" | tee "$LOG"

common_args=(
  examples/action_inject_inference_from_dataset.py
  --job-dir "$JOB_DIR"
  --ckpt-path "$EMA_PT"
  --experiment "$EXPERIMENT"
  --config-file "$CONFIG_FILE"
  --sample-dir "$SAMPLE_DIRS"
  --render-action-video "${RENDER_ACTION_VIDEO:-auto}"
  --mode "$MODE"
  --chunk-size "$CHUNK_SIZE"
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
