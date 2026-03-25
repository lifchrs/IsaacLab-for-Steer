# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the pen task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if the asset falls below a minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def _gripper_is_open(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if both gripper joints are close to the configured open value."""
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    gripper_1_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    gripper_2_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[1]],
        gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    return torch.logical_and(gripper_1_open, gripper_2_open)


def _asset_root_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an asset root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def pen_in_holder(
    env: ManagerBasedRLEnv,
    pen_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder001"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.05,
    min_height_offset: float = 0.02,
    max_height_offset: float = 0.08,
    require_gripper_open: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Check if the pen is aligned with the pen holder opening and released."""
    pen_pos_w = _asset_root_position_w(env, pen_cfg)
    holder_pos_w = _asset_root_position_w(env, holder_cfg)
    pos_diff = pen_pos_w - holder_pos_w

    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    print(f"xy_dist: {xy_dist}")
    height_offset = pos_diff[:, 2]
    print(f"height_offset: {height_offset}")
    placed = torch.logical_and(
        xy_dist <= xy_threshold,
        torch.logical_and(height_offset >= min_height_offset, height_offset <= max_height_offset),
    )

    if require_gripper_open:
        robot: Articulation = env.scene[robot_cfg.name]
        placed = torch.logical_and(placed, _gripper_is_open(env, robot, atol=atol, rtol=rtol))

    return placed


def task_done_pen(
    env: ManagerBasedRLEnv,
    pen_cfg: SceneEntityCfg = SceneEntityCfg("pen"),
    holder_cfg: SceneEntityCfg = SceneEntityCfg("pen_holder001"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.05,
    min_height_offset: float = 0.06,
    max_height_offset: float = 0.24,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Success when the pen is in the holder and the gripper is open."""
    return pen_in_holder(
        env,
        pen_cfg=pen_cfg,
        holder_cfg=holder_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=xy_threshold,
        min_height_offset=min_height_offset,
        max_height_offset=max_height_offset,
        require_gripper_open=True,
        atol=atol,
        rtol=rtol,
    )
