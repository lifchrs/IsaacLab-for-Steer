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
from isaaclab.sensors import ContactSensor

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


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def task_done_block(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("triangle_contact"),
    contact_threshold: float = 0.01,
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]

    # force_matrix_w shape: (num_envs, num_bodies, num_filter_shapes, 3)
    # filter_shapes[0] = cylinder_1, filter_shapes[1] = cylinder_2
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w
    # Compute force magnitude per filter shape: (num_envs, num_bodies, num_filter_shapes)
    force_magnitudes = torch.norm(force_matrix, dim=-1)
    # Sum across all bodies of the triangle: (num_envs, num_filter_shapes)
    force_per_filter = force_magnitudes.sum(dim=1)
    # Check contact with both cylinders
    contact_with_cyl1 = force_per_filter[:, 0] > contact_threshold
    contact_with_cyl2 = force_per_filter[:, 1] > contact_threshold
    done = torch.logical_and(contact_with_cyl1, contact_with_cyl2)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        done = torch.logical_and(suction_cup_is_open, done)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            done = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                done,
            )
            done = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                done,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return done
