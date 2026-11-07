# Example samples

One folder per dataset. See the root [README](../../README.md) for the full
reproduction commands and the bring-your-own-data guide.

```text
examples/samples/<name>/
  meta.json            # source, prompt, camera, temporal_stride
  prompt.txt           # task instruction
  first_frame.png      # RGB condition (the only visual input to the model)
  rgb.mp4              # original RGB — comparison middle panel only
  action.npy           # Libero / ManiSkill / RoboCasa: [T, 8]; others: [T, 28]
  kinematic_qpos.npy   # optional, Libero exact finger joints
```

Action-video mp4 is **not** shipped. Inference online-renders the skeleton from
`action.npy` plus the camera fields in `meta.json`.

| Folder | `source` | Stride | Action |
| --- | --- | ---: | --- |
| `libero_spatial_top_drawer` | `libero` | 1 | `[T, 8]` |
| `maniskill_pick_cube` | `maniskill` | 1 | `[T, 8]` |
| `robocasa_coffee_serve_mug` | `robocasa` | 1 | `[T, 8]` |
| `agibot_put_in_bag` | `agibot` | 9 | `[T, 28]` |
| `agibot_dexhand_steam_shirt` | `agibot_dexhand` | 12 | `[T, 28]` |
| `realsource_arrange_cups` | `real_source_world` | 4 | `[T, 28]` |
| `robomind_close_laptop` | `robomind` | 2 | `[T, 28]` |
| `robotwin_blocks_ranking` | `robotwin` | 2 | `[T, 28]` |
| `robodojo_mv_hold_button` | `robodojo` (3 cameras) | 2 | `[T, 28]` |

`robodojo_mv_hold_button` is a RoboDojo multi-view episode (`"layout": "mv_fill"`) for the multi-view
post-trained 13-frame checkpoint; run it with `./infer_robodojo_mv.sh`. It has one RGB mp4 per camera, and the
wrist entries in `meta.views` give each wrist camera's `K` plus its per-frame pose in `camera_tracks.npz`:

```text
examples/samples/robodojo_mv_hold_button/
  meta.json                 head K / T_world_camera_opencv / arm_bases, layout, views
  rgb.mp4                   cam_high
  rgb_cam_left_wrist.mp4    cam_left_wrist
  rgb_cam_right_wrist.mp4   cam_right_wrist
  camera_tracks.npz         [T, 4, 4] OpenCV pose per RGB frame for each wrist camera
  action.npy                [T, 28]
  prompt.txt
```

```bash
SAMPLE_DIR=examples/samples/libero_spatial_top_drawer NPROC=1 MODE=single ./infer.sh
SAMPLE_DIR=examples/samples/agibot_put_in_bag ./infer.sh   # autoregressive over the full episode
./infer.sh                                                # every folder in manifest.txt
```
