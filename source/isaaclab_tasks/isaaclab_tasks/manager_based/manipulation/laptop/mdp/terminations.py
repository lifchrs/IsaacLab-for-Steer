# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the angry bird task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_rotation_exceeds_threshold(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    threshold: float = 4 * torch.pi / 7,
):
    asset: RigidObject = env.scene[asset_cfg.name]

    asset_rot = asset.data.root_quat_w

    angle_with_z_axis = torch.arccos(
        1 - 2 * (asset_rot[:, 1] ** 2 + asset_rot[:, 2] ** 2)
    )  # angle between asset and z axis
    done = angle_with_z_axis > threshold

    return done


def root_velocity_exceeds_threshold(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.5,
):
    asset: RigidObject = env.scene[asset_cfg.name]

    asset_lin_vel = asset.data.root_lin_vel_w
    asset_ang_vel = asset.data.root_ang_vel_w
    # print(f"asset_lin_vel: {asset_lin_vel}")
    # print(f"asset_ang_vel: {asset_ang_vel}")

    asset_lin_vel_norm = torch.norm(asset_lin_vel, dim=-1)
    asset_ang_vel_norm = torch.norm(asset_ang_vel, dim=-1)

    done = torch.logical_or(
        asset_lin_vel_norm > lin_vel_threshold, asset_ang_vel_norm > ang_vel_threshold
    )

    # for i in range(done.shape[0]):
    #     if done[i]:
    #         print(f"env {i} done")
    #         print(f"asset_lin_vel_norm: {asset_lin_vel_norm[i]}")
    #         print(f"asset_ang_vel_norm: {asset_ang_vel_norm[i]}")
    #         print(f"lin_vel_threshold: {lin_vel_threshold}")
    #         print(f"ang_vel_threshold: {ang_vel_threshold}")

    return done


def task_done_laptop(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    laptop_cfg: SceneEntityCfg = SceneEntityCfg("laptop"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    laptop_closed_threshold: float = -1.5,
    drawer_in_threshold: float = 0.15,
    atol: float = 0.01,
    rtol: float = 0.01,
):
    """Check if the laptop is closed and placed in the drawer.

    Args:
        env: The environment.
        robot_cfg: Robot configuration.
        laptop_cfg: Laptop configuration.
        table_cfg: Table configuration (with drawer).
        laptop_closed_threshold: Joint angle threshold for closed lid.
        drawer_in_threshold: Distance threshold for considering laptop in drawer.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        Boolean tensor indicating success per environment.
    """
    laptop = env.scene[laptop_cfg.name]

    # Check if laptop lid is closed (joint position should be near minimum)
    # The laptop has one joint "RevoluteJoint" for the lid
    lid_joint_pos = laptop.data.joint_pos[:, 0]
    lid_closed = lid_joint_pos < laptop_closed_threshold

    # Check if laptop is in the drawer area
    laptop_pos = laptop.data.root_pos_w
    table_pos = env.scene[table_cfg.name].data.root_pos_w

    # Laptop should be inside drawer area
    height_diff = laptop_pos[:, 2] - table_pos[:, 2]
    in_drawer_height = height_diff < 0.1

    # Check XY distance to drawer center
    xy_dist = torch.linalg.vector_norm(laptop_pos[:, :2] - table_pos[:, :2], dim=1)
    in_drawer_xy = xy_dist < drawer_in_threshold

    # Success when lid is closed and laptop is in drawer
    success = torch.logical_and(lid_closed, torch.logical_and(in_drawer_height, in_drawer_xy))

    return success
