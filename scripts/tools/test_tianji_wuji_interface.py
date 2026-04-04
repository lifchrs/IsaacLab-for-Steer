#!/usr/bin/env python3
"""Test the Tianji+Wuji unified positional control interface (OSC).

Just send the target wrist pose — the Operational Space Controller handles
dynamics (mass, inertia, gravity) automatically via PhysX.

Usage:
    PYTHONUNBUFFERED=1 /path/to/python scripts/tools/test_tianji_wuji_interface.py \
        --headless --enable_cameras --target_offset 0.1 -0.08 0.08
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Tianji+Wuji unified control.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--threshold", type=float, default=0.03)
parser.add_argument("--target_offset", type=float, nargs=3, default=[0.10, -0.08, 0.08],
                    help="Target offset from current EE (x y z meters).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="./videos/tianji_wuji_test")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--modified_values", action="store_true",
                    help="Override USD actuator gains with real Tianji SDK values (from impedance control demo).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.shapes import SphereCfg

from isaaclab_tasks.manager_based.manipulation.tianji_wuji.tianji_wuji_env_cfg import TianjiWujiEnvCfg


def _to_uint8_rgb(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = (image * 255.0).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.clip(0, 255).astype(np.uint8)
    return image


def _camera_rgb(env, name):
    cam = env.scene[name]
    for k in ("rgb", "rgba"):
        if k in cam.data.output:
            return _to_uint8_rgb(cam.data.output[k])
    raise KeyError(f"No rgb from {name}")


def _write_video(frames, path, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"  Saved {len(frames)} frames -> {path}")


def _ee_pos_world(env, body_name, offset_pos, offset_rot):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    pos_w = robot.data.body_pos_w[:, body_ids[0]]
    quat_w = robot.data.body_quat_w[:, body_ids[0]]
    if offset_pos is not None:
        pos_w, _ = math_utils.combine_frame_transforms(pos_w, quat_w, offset_pos, offset_rot)
    return pos_w


def _ee_pose_in_base(env, body_name, offset_pos, offset_rot):
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    pos_w = robot.data.body_pos_w[:, body_ids[0]]
    quat_w = robot.data.body_quat_w[:, body_ids[0]]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w
    )
    if offset_pos is not None:
        pos_b, quat_b = math_utils.combine_frame_transforms(pos_b, quat_b, offset_pos, offset_rot)
    return pos_b, quat_b


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = TianjiWujiEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.modified_values:
        # Real Tianji SDK impedance values (from showcase_torque_joint_impedance demo)
        # K=[2,2,2,1.6,1,1,1] Nm/deg, D=[0.3,0.3,0.3,0.2,0.2,0.2,0.2] Nm/(deg/s)
        # Converted to Nm/rad: multiply by 180/pi ≈ 57.3
        for act_name in ["left_shoulder", "right_shoulder"]:  # joints 1-4
            env_cfg.scene.robot.actuators[act_name].stiffness = 114.6  # 2.0 Nm/deg
            env_cfg.scene.robot.actuators[act_name].damping = 17.2     # 0.3 Nm/(deg/s)
        for act_name in ["left_forearm", "right_forearm"]:    # joints 5-7
            env_cfg.scene.robot.actuators[act_name].stiffness = 57.3   # 1.0 Nm/deg
            env_cfg.scene.robot.actuators[act_name].damping = 11.5     # 0.2 Nm/(deg/s)
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()

    device = env.device
    n = args_cli.num_envs

    # Print action layout
    print(f"\n  Action layout (total {env.action_manager.total_action_dim}D):")
    idx = 0
    for name in env.action_manager.active_terms:
        dim = env.action_manager.get_term(name).action_dim
        print(f"    [{idx}:{idx+dim}] {name} ({dim}D)")
        idx += dim

    # Video setup
    record_video = not args_cli.no_video
    frames = []
    output_dir = ""
    if record_video:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args_cli.output_dir, f"run_{ts}")
        os.makedirs(output_dir, exist_ok=True)

    # IK offset
    ik_offset_pos = torch.tensor([[0.0, 0.0, 0.107]], device=device).repeat(n, 1)
    ik_offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(n, 1)

    # Target
    offset = torch.tensor([args_cli.target_offset], dtype=torch.float32, device=device)
    ee_pos_w_init = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
    target_pos_w = ee_pos_w_init + offset

    robot = env.scene["robot"]
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        target_pos_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
    )

    # Fingers
    finger_target = torch.full((n, 20), 0.8, device=device)

    # Red ball
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetBall",
        markers={"sphere": SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)))},
    )
    markers = VisualizationMarkers(marker_cfg)
    marker_idx = torch.zeros(n, dtype=torch.long, device=device)
    markers.visualize(translations=target_pos_w, marker_indices=marker_idx)

    print(f"\n{'=' * 60}")
    print(f"  Tianji + Wuji Interface Test (IK relative, no USD overrides)")
    print(f"  EE start (world): {ee_pos_w_init[0].cpu().numpy()}")
    print(f"  Target   (world): {target_pos_w[0].cpu().numpy()}")
    print(f"  Offset:           {args_cli.target_offset}")
    print(f"  Threshold: {args_cli.threshold} m")
    if record_video:
        print(f"  Recording to: {output_dir}")
    print(f"{'=' * 60}\n")

    gain = 2.0
    max_delta = 0.08
    converged = False
    converge_step = -1

    for step in range(args_cli.max_steps):
        # IK relative P-controller
        curr_pos_b, curr_quat_b = _ee_pose_in_base(env, "left_link7", ik_offset_pos, ik_offset_rot)
        pos_err, rot_err = math_utils.compute_pose_error(
            curr_pos_b, curr_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle",
        )
        left_arm_cmd = torch.cat((gain * pos_err, 0.3 * rot_err), dim=-1)
        left_arm_cmd[:, :3] = torch.clamp(left_arm_cmd[:, :3], -max_delta, max_delta)
        left_arm_cmd[:, 3:] = torch.clamp(left_arm_cmd[:, 3:], -0.12, 0.12)

        right_arm_cmd = torch.zeros((n, 6), device=device)
        alpha = min(1.0, step / 30.0)
        left_hand_cmd = alpha * finger_target
        right_hand_cmd = torch.zeros((n, 20), device=device)

        action = torch.cat([left_arm_cmd, right_arm_cmd, left_hand_cmd, right_hand_cmd], dim=-1)

        ee_w = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
        dist = torch.linalg.vector_norm(ee_w - target_pos_w, dim=-1)[0].item()

        if dist < args_cli.threshold and not converged:
            converged = True
            converge_step = step

        print(f"  step {step:3d}  |  ee_dist={dist:.4f}m")

        markers.visualize(translations=target_pos_w, marker_indices=marker_idx)
        env.step(action)

        if record_video:
            try:
                frames.append(_camera_rgb(env, "overhead_cam"))
            except KeyError:
                pass

        if converged and step - converge_step > 10:
            break

    if record_video and frames:
        print(f"\n  Writing video...")
        _write_video(frames, os.path.join(output_dir, "overhead_cam.mp4"), args_cli.fps)

    ee_final = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
    final_dist = torch.linalg.vector_norm(ee_final - target_pos_w, dim=-1)[0].item()
    ee_pass = final_dist < args_cli.threshold

    print(f"\n{'=' * 60}")
    status = "PASS" if ee_pass else "FAIL"
    info = f"converged at step {converge_step}" if converged else "did not converge"
    print(f"  [{status}] EE: final_dist={final_dist:.4f}m  ({info})")
    if record_video and output_dir:
        print(f"  Videos: {output_dir}")
    print(f"{'=' * 60}\n")

    env.close()
    sys.exit(0 if ee_pass else 1)


if __name__ == "__main__":
    main()
    simulation_app.close()
