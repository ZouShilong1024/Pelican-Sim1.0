---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/Cosmos-Predict2.5-2B
pipeline_tag: robotics
library_name: pytorch
tags:
  - world-model
  - robotics
  - video-generation
  - action-conditioned
  - mixture-of-experts
  - cosmos
---

# Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence

[🌐 Project Page](https://zoushilong1024.github.io/Pelican-Sim1.0/) | [📄 arXiv](https://arxiv.org/abs/2609.12036) | [💻 Code](https://github.com/ZouShilong1024/Pelican-Sim1.0)

WFM System Group, Beijing Innovation Center of Humanoid Robotics (X-Humanoid)

![Pelican-Sim 1.0 overview](assets/teaser.png)

**Pelican-Sim 1.0 is a shared action-conditioned world model simulator for heterogeneous robots.** Given a
first-frame RGB image, a robot joint trajectory, and a language instruction, it predicts the future video. The
trajectory enters the model twice — as numerical configurations in a unified 28-D layout and as a URDF-rendered,
camera-aligned skeleton video — and the two conditions are injected into alternating blocks of a sparse-MoE Video
DiT, so each shapes hidden states already informed by the other.

## Highlights

- **A unified cross-embodiment world model simulator.** One checkpoint covers parallel-gripper, dexterous-hand,
  single-arm, and bimanual robots through a canonical 28-D action layout, and generates interactive videos that
  follow the commanded actions. Its distilled four-step autoregressive variant enables efficient rollouts, and a
  fine-tuned VLM evaluator turns predicted outcomes into feedback for policy evaluation, action selection, and policy
  optimization.
- **Action alignment and sparse expert capacity both help with heterogeneous dynamics.** Interleaving numerical and
  visual conditions through shared hidden states beats single-condition, concatenation, and late-fusion baselines
  (PSNR **+0.904 dB** over late fusion on AgiBotWorld Beta). With the action interface and the mixed real–simulated
  corpus fixed, sparse MoE (1 shared + 8 routed experts, top-2) further lowers FVD by **6.530** over the dense
  backbone. Both are general mechanisms for modeling video dynamics.
- **State-of-the-art prediction and downstream utility.** Pelican-Sim 1.0 leads all evaluated baselines on five
  video-quality metrics across three held-out test sets (PSNR **+4.636 / +2.080 / +10.343 dB** on AgiBotWorld Beta /
  RoboMIND / RoboTwin) and supports four downstream policy applications. On RoboTwin, data augmentation and imagined
  policy optimization raise policy success from **70.0% → 93.0%** and **62.7% → 75.4%**. Qualitative studies show
  action-consistent predictions under trajectory, scene, embodiment, object, and viewpoint shifts.
- **Open release.** Model weights on Hugging Face; inference and fine-tuning code and example data in this
  repository.

## Checkpoint

| File | Content |
| --- | --- |
| `model_ema_bf16.pt` | EMA weights, bf16, ~23.6 GB. Every trained weight of the model below. |

- **Backbone:** 28-layer Cosmos-Predict2.5-2B Video DiT.
- **Action conditioning:** numerical 28-D action embeddings modulate the 14 odd blocks; 14 Context Blocks inject
  action-video residuals after the even blocks.
- **Sparse MoE:** every feed-forward layer has 1 shared + 8 routed experts, sigmoid top-2 routing, route scale 2.5.
- **Sampler:** 35-step bidirectional rectified flow; 93-frame chunks at 480×640, chunk-wise autoregressive for longer
  episodes.
- **Training data:** \~1M real and simulated trajectories (\~8,000 hours) from AgiBotWorld Beta, RealSource World,
  RoboMIND, LIBERO, ManiSkill2, RoboTwin, and RoboCasa.

The Wan2.1 VAE tokenizer ([`nvidia/Cosmos-Predict2.5-2B`](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)) and
the text encoder ([`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)) are frozen public
components and are downloaded from their own releases by the inference code.

## Results

| Held-out split | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | PSNR gain over best baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgiBotWorld Beta | 22.276 | 0.796 | 0.013 | 14.640 | 10.850 | +4.636 dB |
| RoboMIND | 23.850 | 0.835 | 0.012 | 17.610 | 15.930 | +2.080 dB |
| RoboTwin | 30.383 | 0.942 | 0.006 | 8.910 | 4.980 | +10.343 dB |

On RoboTwin, generated-data augmentation raises policy success from 70.0% to 93.0%, and imagined policy optimization
with task-specific simulator adaptation improves success from 62.7% to 75.4%.

## Usage

```bash
git clone https://github.com/ZouShilong1024/Pelican-Sim1.0.git && cd Pelican-Sim1.0
uv sync --extra=cu128 && uv pip install yourdfpy opencv-python pyarrow && source .venv/bin/activate
hf download ShilongZou/Pelican-Sim1.0 --local-dir checkpoints/Pelican-Sim-1.0

# Autoregressive rollout of a bundled example until the episode ends
SAMPLE_DIR=examples/samples/agibot_put_in_bag ./infer.sh

# Fine-tune on your own episodes (this checkpoint is the default starting point)
EPISODES=data/my_task/episodes.jsonl JOB_NAME=my_task ./finetune.sh
```

See the [GitHub README](https://github.com/ZouShilong1024/Pelican-Sim1.0) for bringing your own robot data, and
the [fine-tuning guide](https://github.com/ZouShilong1024/Pelican-Sim1.0/blob/main/docs/finetune_robodojo.md) for
fine-tuning on RoboDojo or other robots.

## Citation

```bibtex
@article{zou2026pelicansim,
  title         = {Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence},
  author        = {Shilong Zou and Shilin Zhang and Yingji Zhang and Yuhang Huang and Yi Zhang and Zeyuan Ding
                   and Han Dong and Junwei Liao and Yong Dai and Jian Tang and Xiaozhu Ju},
  journal       = {arXiv preprint arXiv:2609.12036},
  year          = {2026},
  eprint        = {2609.12036},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.12036}
}
```
