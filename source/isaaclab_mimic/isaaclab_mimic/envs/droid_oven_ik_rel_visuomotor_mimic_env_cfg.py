# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.oven.config.droid.oven_ik_rel_visuomotor_env_cfg import (
    DroidOvenIkRelVisuomotorEnvCfg,
)


@configclass
class DroidOvenIKRelVisuomotorMimicEnvCfg(
    DroidOvenIkRelVisuomotorEnvCfg, MimicEnvCfg
):
    """
    Isaac Lab Mimic environment config class for Droid Oven IK Rel Visuomotor env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        self.datagen_config.name = "isaac_lab_droid_oven_ik_rel_visuomotor_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        ACTION_NOISE = 0.01

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="oven",
                subtask_term_signal="oven_opened",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=ACTION_NOISE,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="can",
                subtask_term_signal="grasp_can",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=ACTION_NOISE,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="oven",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=ACTION_NOISE,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["franka"] = subtask_configs
