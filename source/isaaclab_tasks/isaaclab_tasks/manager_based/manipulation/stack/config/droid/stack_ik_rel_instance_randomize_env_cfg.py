# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_instance_randomize_env_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


@configclass
class DroidCubeStackInstanceRandomizeEnvCfg(
    stack_joint_pos_instance_randomize_env_cfg.DroidCubeStackInstanceRandomizeEnvCfg
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = UsdFileCfg(
            usd_path=f"asset/droid/droid.usd",
            scale=(1.0, 1.0, 1.0),
        )

        # Reduce the number of environments due to camera resources
        self.scene.num_envs = 2

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="robotiq_85_tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
