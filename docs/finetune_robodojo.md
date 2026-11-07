# Fine-tuning Pelican-Sim 1.0 on RoboDojo

Prepare RoboDojo episodes, then continue training from the released checkpoint with `finetune.sh`. The same
manifest works for any robot whose `action.npy` uses the unified 28-D layout in
[Use your own data](../README.md#use-your-own-data).

## Prepare the data

`scripts/prepare_robodojo.py` writes one folder per episode and the manifest `finetune.sh` reads. HDF5 and
LeRobot v3 produce the same actions, camera parameters, and action videos.

```bash
# Official HDF5: a file, or a directory searched recursively
python scripts/prepare_robodojo.py --hdf5 /data/RoboDojo \
    --scenes push_T,fold_clothes --per-scene 50 --out-dir data/robodojo_ft --workers 16

# LeRobot v3 export (meta/, data/, videos/, SAMPLE_MAPPING.jsonl); camera geometry comes from the HDF5 it maps to
python scripts/prepare_robodojo.py --lerobot /data/robodojo_lerobot --out-dir data/robodojo_ft --workers 16
```

| Option | Meaning |
| --- | --- |
| `--scenes` | comma-separated task names (default: all) |
| `--per-scene N` | first N episodes of each task (default: all) |
| `--episodes` | `--lerobot` only: explicit `episode_index` list |
| `--stride` | temporal stride of the model frames (2: 25 Hz recordings → 12.5 Hz) |
| `--overwrite` | rebuild folders that already exist |

Each episode becomes one folder:

```text
data/robodojo_ft/push_T_ep000000/
  rgb.mp4            head camera, 640×480, 25 fps
  action_video.mp4   ARX-X5 skeleton rendered from action.npy (same frames and camera)
  action.npy         [T, 28] unified action: per arm 6 joints (rad) + gripper (1 = open), zeros elsewhere
  first_frame.png    first RGB frame
  prompt.txt         instruction
  meta.json          camera K / extrinsic, arm bases, stride
```

Row `t` of `action.npy` is the joint state when RGB frame `t` was captured. For `imitate_sorting_sequence`, the
scripted demonstration before the ARX arms first move is cut.

### Manifest format

`episodes.jsonl` has one JSON object per episode, with absolute paths. `action_video.mp4` must be frame-aligned
with `rgb.mp4`.

```json
{"uuid": "push_T/episode_000000", "task_name": "push_T",
 "rgb": "/abs/data/robodojo_ft/push_T_ep000000/rgb.mp4",
 "action_video": "/abs/data/robodojo_ft/push_T_ep000000/action_video.mp4",
 "action": "/abs/data/robodojo_ft/push_T_ep000000/action.npy",
 "lang": "Push the T-shaped block to align it precisely with the gray T-shaped pad.",
 "frame_nums": 359, "fps": 25.0, "temporal_stride": 2}
```

`temporal_stride` is optional (default `STRIDE`). An episode shorter than 93 model frames at that stride falls back
to the largest stride that fits; one shorter than 93 frames even at stride 1 is skipped. Each training sample is a
random 93-frame window (30% of the time starting at the first frame).

Hold out a few episodes per task:

```bash
cd data/robodojo_ft
shuf --random-source=<(yes) episodes.jsonl > all.jsonl
head -n 40 all.jsonl > eval.jsonl && tail -n +41 all.jsonl > train.jsonl
```

## Train

```bash
EPISODES=data/robodojo_ft/train.jsonl JOB_NAME=robodojo_ft ./finetune.sh
```

The first run converts `model_ema_bf16.pt` to a DCP checkpoint beside it (`dcp_model_ema_bf16/`). Checkpoints go
to `outputs/pelican_sim/finetune/<JOB_NAME>/checkpoints/`. Re-run the same `JOB_NAME` to resume from
`checkpoints/latest_checkpoint.txt`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPISODES` | required | manifest (`episodes.jsonl` or `train.jsonl`) |
| `JOB_NAME` | `finetune_<manifest dir>` | run name and output folder |
| `INIT_CKPT` | `checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt` | `.pt` or DCP `iter_*` directory to start from |
| `INIT_DCP` | `<pt dir>/dcp_<pt stem>` | where a `.pt` `INIT_CKPT` is converted to |
| `LR` | `3.05e-5` (2⁻¹⁵) | action-module LR; the rest of the model uses 0.1× |
| `MAX_ITER` | `10000` | iterations (100-step warm-up, then linear decay to 0.4× at `MAX_ITER`) |
| `SAVE_ITER` | `1000` | checkpoint interval |
| `SAMPLE_EVERY` | `1000` | interval of the preview videos |
| `STRIDE` | `2` | stride for manifest lines without `temporal_stride` |
| `NPROC` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` | `8` / `1` / `0` / `127.0.0.1` / `12370` | `torch.distributed.run` |
| `WANDB_MODE` | `offline` | `online` needs `WANDB_API_KEY` |
| `SMOKE=1` | | 4 iterations, checkpoint at iteration 2, no previews, no W&B |
| `EXTRA_ARGS` | | extra Hydra overrides, e.g. `"trainer.grad_accum_iter=2"` |

```bash
SMOKE=1 EPISODES=data/robodojo_ft/train.jsonl JOB_NAME=smoke ./finetune.sh
```

## Evaluate

`infer_robodojo.sh` turns the job checkpoint into `checkpoints/<JOB_NAME>/<iter>/model_ema_bf16.pt` on first
use, predicts each held-out episode from its first frame and actions, and prints PSNR / SSIM against `rgb.mp4`.
`ITER` defaults to the latest checkpoint. `MODE=ar` rolls out the whole episode instead of one 93-frame chunk.

```bash
EVAL=$(python -c 'import json,os,sys; print(",".join(os.path.dirname(json.loads(l)["rgb"]) for l in sys.stdin))' \
       < data/robodojo_ft/eval.jsonl)

JOB_DIR=outputs/pelican_sim/finetune/robodojo_ft ITER=iter_000010000 \
  SAMPLE_DIR=$EVAL NPROC=8 ./infer_robodojo.sh

CKPT_PATH=checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt SAMPLE_DIR=$EVAL NPROC=8 ./infer_robodojo.sh
```

Videos, `inference_results.jsonl`, and the mean scores are under
`outputs/infer_robodojo_<mode>_<iter>_g3_seed0/` (`infer.log`). To export one file from a DCP checkpoint:

```bash
python scripts/convert_distcp_to_pt.py \
  outputs/pelican_sim/finetune/robodojo_ft/checkpoints/iter_000010000/model out_dir/
```

## Multi-view (head + both wrists)

The multi-view model sees all three RoboDojo cameras on the released 480×640 canvas, so it starts from the same
`model_ema_bf16.pt` without a spatial-size change. The head camera fills the top 2/3 at full width; the left and
right wrist cameras split the bottom 1/3. Each view is resized into its cell. The RGB and the skeleton video use
the same packing, and the 28-D action is shared. The wrist cameras move with the arms, so each wrist skeleton frame
is projected with that frame's own wrist pose.

```text
┌───────────────────────────────┐
│           cam_high            │  640×320
├───────────────┬───────────────┤
│ cam_left_wrist│cam_right_wrist│  320×160 each
└───────────────┴───────────────┘
```

Training uses 13-frame windows (`state_t` 4, one policy chunk). Each sample draws stride 1, 2, or 4 (25 / 12.5 /
6.25 fps) with probability 0.5 / 0.3 / 0.2, skipping a stride whose window does not fit the episode.

```bash
# 1. Folders with the wrist cameras too (rgb_cam_*_wrist.mp4, action_video_cam_*_wrist.mp4, camera_tracks.npz)
python scripts/prepare_robodojo.py --lerobot /data/robodojo_lerobot --multiview \
    --out-dir data/robodojo_mv_ft --workers 16

# 2. Fine-tune from checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt
SMOKE=1 MULTIVIEW=1 EPISODES=data/robodojo_mv_ft/episodes.jsonl JOB_NAME=smoke_mv ./finetune.sh
MULTIVIEW=1 EPISODES=data/robodojo_mv_ft/episodes.jsonl JOB_NAME=robodojo_mv_ft ./finetune.sh

# 3. Roll out each episode chunk by chunk until its last action (13-frame chunks, 1-frame overlap)
JOB_DIR=outputs/pelican_sim/finetune/robodojo_mv_ft SAMPLE_DIR=data/robodojo_mv_ft NPROC=8 ./infer_robodojo_mv.sh
```

`MULTIVIEW=1` selects the experiment `ac_inject_moe_finetune_mv_2b`; `MV_NUM_FRAMES` (13, must be 4k+1),
`MV_STRIDES` (`1,2,4`), and `MV_STRIDE_PROBS` (`0.5,0.3,0.2`) change the window and strides. Each multi-view manifest
line adds the wrist files:

```json
{"rgb": ".../rgb.mp4", "action_video": ".../action_video.mp4", "action": ".../action.npy", "...": "...",
 "views": {"cam_left_wrist":  {"rgb": ".../rgb_cam_left_wrist.mp4",  "action_video": ".../action_video_cam_left_wrist.mp4"},
           "cam_right_wrist": {"rgb": ".../rgb_cam_right_wrist.mp4", "action_video": ".../action_video_cam_right_wrist.mp4"}}}
```

`infer_robodojo_mv.sh` takes sample folders or a `--multiview` output directory in `SAMPLE_DIR` (default:
[`examples/samples/robodojo_mv_hold_button`](../examples/samples/robodojo_mv_hold_button)). It renders the three
skeletons online from `action.npy`, `meta.json`, and `camera_tracks.npz`, packs the first frame of the three cameras
as the condition, and generates until the episode's last action (`--match-episode-length`). `MODE=single` generates
one 13-frame chunk. `CKPT_PATH` takes a `.pt` directly; otherwise `JOB_DIR` / `ITER` are converted once to
`checkpoints/<job>/<iter>/model_ema_bf16.pt`. Results are in
`outputs/infer_robodojo_mv_<mode>_<iter>_g3_seed0/`; PSNR / SSIM are measured on the packed canvas, plus a PSNR
per camera (`cam_high`, `cam_left_wrist`, `cam_right_wrist`).
