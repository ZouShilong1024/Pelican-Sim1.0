#!/usr/bin/env bash
# Fine-tune Pelican-Sim on an episode manifest (see docs/finetune_robodojo.md).
#
#   EPISODES=data/robodojo_ft/episodes.jsonl JOB_NAME=robodojo_ft ./finetune.sh
#   # RoboDojo multi-view (head + both wrists on one canvas, 13-frame windows, stride 1/2/4):
#   MULTIVIEW=1 EPISODES=data/robodojo_mv_ft/episodes.jsonl JOB_NAME=robodojo_mv_ft ./finetune.sh
#
# Env:
#   EPISODES     manifest from scripts/prepare_robodojo.py (--multiview for MULTIVIEW=1)  [required]
#   JOB_NAME     run name; outputs/pelican_sim/finetune/<JOB_NAME>; re-running resumes
#   INIT_CKPT    model_ema_bf16.pt or DCP iter dir   (checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt)
#   INIT_DCP     where a .pt INIT_CKPT is converted to (<pt dir>/dcp_<pt stem>; ~70 GB)
#   MULTIVIEW    1 = experiment ac_inject_moe_finetune_mv_2b (0)
#   MV_NUM_FRAMES / MV_STRIDES / MV_STRIDE_PROBS   multi-view window and strides (13 / 1,2,4 / 0.5,0.3,0.2)
#   STRIDE       model stride for episodes without temporal_stride (2)
#   LR           action-module LR; the rest of the model gets 0.1x (3.0517578125e-05 = 2**-15)
#   MAX_ITER / SAVE_ITER / SAMPLE_EVERY      (10000 / 1000 / 1000)
#   NPROC / NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT   (8 / 1 / 0 / 127.0.0.1 / 12370)
#   WANDB_MODE   online | offline | disabled (offline)
#   SMOKE=1      4 iterations, checkpoint at 2, no sample videos, no wandb
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
if [[ -z "${HF_HOME:-}" && -d "$ROOT/hf_cache/hub" ]]; then
  HF_HOME="$ROOT/hf_cache"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
fi
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$ROOT/outputs}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

EPISODES="${EPISODES:?set EPISODES=/path/to/episodes.jsonl (scripts/prepare_robodojo.py)}"
[[ -f "$EPISODES" ]] || { echo "ERROR: EPISODES=$EPISODES not found" >&2; exit 1; }
EPISODES="$(realpath "$EPISODES")"
JOB_NAME="${JOB_NAME:-finetune_$(basename "$(dirname "$EPISODES")")}"
JOB_DIR="$IMAGINAIRE_OUTPUT_ROOT/pelican_sim/finetune/$JOB_NAME"

# ---- init checkpoint: the trainer only loads DCP, so convert a .pt once ----
INIT_CKPT="${INIT_CKPT:-$ROOT/checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt}"
if [[ "$INIT_CKPT" == *.pt ]]; then
  # Keep symlinks unresolved so the DCP lands next to the path given, not the link target.
  pt="$(realpath -s "$INIT_CKPT")"
  INIT_CKPT="${INIT_DCP:-$(dirname "$pt")/dcp_$(basename "$pt" .pt)}/iter_000000000"
  if [[ ! -f "$INIT_CKPT/model/.metadata" ]]; then
    echo "=== converting $pt -> $INIT_CKPT ==="
    "$VENV_BIN/python" scripts/convert_pt_to_dcp.py "$pt" "$INIT_CKPT"
  fi
fi
[[ -f "$INIT_CKPT/model/.metadata" ]] || { echo "ERROR: INIT_CKPT=$INIT_CKPT is not a DCP iter dir" >&2; exit 1; }

export FT_INIT_CKPT="$INIT_CKPT"
export FT_EPISODES_JSONL="$EPISODES"
export FT_STRIDE="${STRIDE:-2}"
export FT_LR="${LR:-3.0517578125e-05}"
export FT_MAX_ITER="${MAX_ITER:-10000}"
export FT_SAVE_ITER="${SAVE_ITER:-1000}"
export FT_SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
export FT_MV_NUM_FRAMES="${MV_NUM_FRAMES:-13}"
export FT_MV_STRIDES="${MV_STRIDES:-1,2,4}"
export FT_MV_STRIDE_PROBS="${MV_STRIDE_PROBS:-0.5,0.3,0.2}"
if [[ "${MULTIVIEW:-0}" == "1" ]]; then
  EXPERIMENT=ac_inject_moe_finetune_mv_2b
  [[ "$(head -n 1 "$EPISODES")" == *'"views"'* ]] \
    || { echo "ERROR: $EPISODES has no wrist views; run scripts/prepare_robodojo.py --multiview" >&2; exit 1; }
else
  EXPERIMENT=ac_inject_moe_finetune_2b
fi

WANDB_MODE="${WANDB_MODE:-offline}"
extra=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  extra+=(trainer.max_iter=4 trainer.logging_iter=1 checkpoint.save_iter=2
          trainer.callbacks.every_n_sample_ema.every_n=1000000)
  WANDB_MODE=disabled
fi

resume="$(cat "$JOB_DIR/checkpoints/latest_checkpoint.txt" 2>/dev/null || true)"
echo "============================================================"
echo " Pelican-Sim fine-tuning"
echo " JOB_DIR   = $JOB_DIR${resume:+   (resuming $resume)}"
echo " INIT_CKPT = $INIT_CKPT"
echo " EXPERIMENT= $EXPERIMENT"
if [[ "$EXPERIMENT" == *_mv_* ]]; then
  echo " EPISODES  = $EPISODES ($(wc -l < "$EPISODES") episodes, $FT_MV_NUM_FRAMES frames, strides $FT_MV_STRIDES)"
else
  echo " EPISODES  = $EPISODES ($(wc -l < "$EPISODES") episodes, stride $FT_STRIDE)"
fi
echo " LR        = $FT_LR   MAX_ITER = $FT_MAX_ITER   SAVE_ITER = $FT_SAVE_ITER"
echo "============================================================"

mkdir -p logs
LOG="$ROOT/logs/finetune_${JOB_NAME}_n${NODE_RANK:-0}_$(date +%Y%m%d_%H%M%S).log"
"$VENV_BIN/python" -u -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC:-8}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12370}" \
  -m cosmos_oss.scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_inject/config.py -- \
  "experiment=$EXPERIMENT" \
  "job.name=$JOB_NAME" \
  "job.wandb_mode=$WANDB_MODE" \
  "${extra[@]}" \
  ${EXTRA_ARGS:-} 2>&1 | tee -a "$LOG"
