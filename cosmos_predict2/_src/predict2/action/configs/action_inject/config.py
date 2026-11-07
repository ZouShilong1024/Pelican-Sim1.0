# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra entry for Pelican-Sim inference and fine-tuning.

Experiments live in ``cosmos_predict2/experiments/base``:
  - ``ac_inject_moe_shared_dense_e8_rectified_flow_2b``: inference
  - ``ac_inject_moe_finetune_2b``: fine-tuning (see ``docs/finetune_robodojo.md``)
"""

from typing import Any, List

import attrs

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from cosmos_predict2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_predict2._src.predict2.action.configs.action_inject.conditioner import register_conditioner
from cosmos_predict2._src.predict2.action.configs.action_inject.model import register_model
from cosmos_predict2._src.predict2.action.configs.action_inject.net import register_net
from cosmos_predict2._src.predict2.action.configs.action_inject.optimizer import (
    register_optimizer as register_action_inject_optimizer,
)
from cosmos_predict2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_predict2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_predict2._src.predict2.configs.common.defaults.dataloader import register_training_and_val_data
from cosmos_predict2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_predict2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_predict2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_predict2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_predict2._src.predict2.configs.video2world.defaults.callbacks import register_callbacks


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "action_inject_fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "action_inject_moe_video2world_fsdp_rectified_flow"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "action_inject_video_conditioner"},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    # Maps the s3:// VAE / text-encoder URIs in the base configs to their Hugging Face releases.
    from cosmos_oss.checkpoints_predict2 import register_checkpoints

    register_checkpoints()

    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    c.job.project = "cosmos_predict2_action_inject"
    c.job.group = "pelican_sim"
    c.job.name = "action_inject_moe"
    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.logging_iter = 10
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    register_optimizer()
    register_action_inject_optimizer()
    register_scheduler()
    register_model()
    register_callbacks()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()
    register_training_and_val_data()
    register_net()
    register_conditioner()

    import_all_modules_from_package("cosmos_predict2.experiments", reload=True)
    import cosmos_predict2._src.predict2.configs.video2world.experiment.reason_embeddings.model_2B_reason_1p1_rectified_flow  # noqa: F401

    return c
