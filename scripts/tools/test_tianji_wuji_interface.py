#!/usr/bin/env python3
"""Test the Tianji+Wuji unified positional control interface.

Architecture (matches real robot):
  P-controller (policy rate) → 6D delta → DiffIK action term (240 Hz) → PhysX PD

The DiffIK action term re-solves IK at each physics sub-step (240 Hz), acting
as the "internal servo" — equivalent to the real robot's 1 kHz control loop.
The P-controller is the "policy" that generates EE commands.

Modes:
  (default)      Relative IK + P-controller (DiffIK at 240 Hz)
  --absolute     Absolute IK (sends target pose directly)
  --analytical   Tianji SDK analytical IK + PhysX PD (closest to real robot)

Usage:
    PYTHONUNBUFFERED=1 /path/to/python scripts/tools/test_tianji_wuji_interface.py \
        --headless --enable_cameras --analytical --target_offset 0.1 -0.08 0.08
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
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--absolute", action="store_true",
                    help="Use absolute IK mode instead of relative IK + P-controller.")
parser.add_argument("--analytical", action="store_true",
                    help="Use Tianji SDK analytical IK + JointPosition PD (closest to real robot).")
parser.add_argument("--control_hz", type=int, default=None,
                    help="Policy update rate in Hz (default: 240). Sim always runs at 240 Hz.")
parser.add_argument("--usd_gains", action="store_true",
                    help="Use original USD actuator gains instead of real Tianji SDK values.")
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


def _trapezoidal_profile(t, distance, v_max, a_max):
    """Compute position along a 1D trapezoidal velocity profile.

    Returns a fraction in [0, 1] of the total distance traveled at time t.
    The profile accelerates at a_max, cruises at v_max, and decelerates at a_max.
    """
    if abs(distance) < 1e-8:
        return 1.0

    d = abs(distance)
    # Time to accelerate to v_max
    t_accel = v_max / a_max
    # Distance covered during acceleration
    d_accel = 0.5 * a_max * t_accel ** 2

    if 2 * d_accel >= d:
        # Triangle profile (can't reach v_max)
        t_accel = (d / a_max) ** 0.5
        t_total = 2 * t_accel
        t = min(t, t_total)
        if t <= t_accel:
            s = 0.5 * a_max * t ** 2
        else:
            dt = t - t_accel
            v_peak = a_max * t_accel
            s = d_accel + v_peak * dt - 0.5 * a_max * dt ** 2
            # actually d_accel here is 0.5 * d
        return min(s / d, 1.0)
    else:
        # Full trapezoidal
        d_cruise = d - 2 * d_accel
        t_cruise = d_cruise / v_max
        t_total = 2 * t_accel + t_cruise
        t = min(t, t_total)
        if t <= t_accel:
            s = 0.5 * a_max * t ** 2
        elif t <= t_accel + t_cruise:
            s = d_accel + v_max * (t - t_accel)
        else:
            dt = t - t_accel - t_cruise
            s = d_accel + d_cruise + v_max * dt - 0.5 * a_max * dt ** 2
        return min(s / d, 1.0)


def _compute_trapz_fraction(t, joint_distances, v_max, a_max):
    """Compute trapezoidal profile fraction for multi-joint motion.

    Uses the joint with the largest distance to determine the timing,
    so all joints start and stop together (coordinated motion).
    """
    max_dist = float(joint_distances.abs().max())
    return _trapezoidal_profile(t, max_dist, v_max, a_max)


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = TianjiWujiEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Use real Tianji SDK actuator gains by default (USD values are ~100x too low).
    # Real values from Tianji SDK showcase_torque_joint_impedance demo:
    #   K=[2,2,2,1.6,1,1,1] Nm/deg, D=[0.3,0.3,0.3,0.2,0.2,0.2,0.2] Nm/(deg/s)
    #   Converted to Nm/rad: multiply by 180/pi ≈ 57.3
    if not args_cli.usd_gains:
        for act_name in ["left_shoulder", "right_shoulder"]:  # joints 1-4
            env_cfg.scene.robot.actuators[act_name].stiffness = 114.6  # 2.0 Nm/deg
            env_cfg.scene.robot.actuators[act_name].damping = 17.2     # 0.3 Nm/(deg/s)
        for act_name in ["left_forearm", "right_forearm"]:    # joints 5-7
            env_cfg.scene.robot.actuators[act_name].stiffness = 57.3   # 1.0 Nm/deg
            env_cfg.scene.robot.actuators[act_name].damping = 11.5     # 0.2 Nm/(deg/s)

    # Policy interval: how many physics steps between policy updates
    if args_cli.control_hz is not None:
        policy_interval = max(1, 240 // args_cli.control_hz)
    elif args_cli.analytical:
        policy_interval = 24  # default 10 Hz for analytical
    else:
        policy_interval = 1   # default 240 Hz for DiffIK modes

    # Always run sim at 240 Hz (decimation=1) for smooth rendering
    env_cfg.decimation = 1

    if args_cli.analytical:
        # Analytical IK: arms use JointPositionAction, IK solved externally via SDK
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg as _JointPosCfg
        env_cfg.actions.left_arm_action = _JointPosCfg(
            asset_name="robot", joint_names=["left_joint.*"],
            scale=1.0, use_default_offset=False,
        )
        env_cfg.actions.right_arm_action = _JointPosCfg(
            asset_name="robot", joint_names=["right_joint.*"],
            scale=1.0, use_default_offset=False,
        )
    elif args_cli.absolute:
        # Switch to absolute IK: action = [x, y, z, qw, qx, qy, qz] in base frame
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        WUJI_EE_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.107], rot=[0.0, 0.0, 0.0, 1.0],
        )
        for arm, body in [("left_arm_action", "left_link7"), ("right_arm_action", "right_link7")]:
            joints = ["left_joint.*"] if "left" in arm else ["right_joint.*"]
            setattr(env_cfg.actions, arm, DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=joints,
                body_name=body,
                controller=DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="pinv",
                    ik_params={"k_val": 0.3},
                ),
                scale=1.0,
                body_offset=WUJI_EE_OFFSET,
            ))

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

    # Prepare mode-specific state
    if args_cli.analytical:
        from isaaclab_assets.robots.tianji_analytical_ik import TianjiAnalyticalIK
        from scipy.spatial.transform import Rotation as SciRotation

        sdk_ik = TianjiAnalyticalIK()
        sdk_ik.set_tool_offset("left", 12.0)  # flange → EE (M6: 870.5 + 12 = 882.5mm)

        # Get arm base frame for coordinate conversion
        arm_base_ids, _ = robot.find_bodies("left_base_link")
        left_joint_ids_list, _ = robot.find_joints(["left_joint.*"])
        right_joint_ids_list, _ = robot.find_joints(["right_joint.*"])

        # Get current EE pose in arm base frame (for target orientation)
        link7_pos_w = robot.data.body_pos_w[:, robot.find_bodies("left_link7")[0][0]]
        link7_quat_w = robot.data.body_quat_w[:, robot.find_bodies("left_link7")[0][0]]
        ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
            link7_pos_w, link7_quat_w, ik_offset_pos, ik_offset_rot
        )
        # Convert target position and current EE orientation to arm base frame
        target_pos_ab, _ = math_utils.subtract_frame_transforms(
            robot.data.body_pos_w[:, arm_base_ids[0]],
            robot.data.body_quat_w[:, arm_base_ids[0]],
            target_pos_w,
            ee_quat_w,  # keep current orientation
        )
        _, ee_quat_ab = math_utils.subtract_frame_transforms(
            robot.data.body_pos_w[:, arm_base_ids[0]],
            robot.data.body_quat_w[:, arm_base_ids[0]],
            ee_pos_w, ee_quat_w,
        )

        # Solve analytical IK once — exact joint solution
        target_pos_ab_np = target_pos_ab[0].cpu().numpy()
        target_quat_ab_np = ee_quat_ab[0].cpu().numpy()  # keep current orientation, wxyz
        ref_joints = robot.data.joint_pos[0, left_joint_ids_list].cpu().numpy()
        # Ensure ref joint4 != 0 (SDK constraint)
        if abs(ref_joints[3]) < 0.01:
            ref_joints[3] = 0.1

        ik_joints = sdk_ik.solve_ik("left", target_pos_ab_np, target_quat_ab_np, ref_joints)
        if ik_joints is None:
            print("  [ERROR] Analytical IK failed — target unreachable")
            env.close()
            sys.exit(1)
        print(f"  Analytical IK solution (deg): {np.degrees(ik_joints).round(2)}")

        # Joint targets and trajectory planner state
        left_arm_target_joints = torch.tensor(ik_joints, dtype=torch.float32, device=device).unsqueeze(0).repeat(n, 1)
        left_arm_start_joints = robot.data.joint_pos[:, left_joint_ids_list].clone()
        right_arm_fixed = robot.data.joint_pos[:, right_joint_ids_list].clone()

        # Trapezoidal velocity profile parameters
        max_joint_vel = 2.0    # rad/s (max velocity in joint space)
        max_joint_accel = 8.0  # rad/s^2 (acceleration/deceleration)
    elif args_cli.absolute:
        left_pos_b, left_quat_b = _ee_pose_in_base(env, "left_link7", ik_offset_pos, ik_offset_rot)
        right_pos_b, right_quat_b = _ee_pose_in_base(env, "right_link7", ik_offset_pos, ik_offset_rot)
        left_arm_target = torch.cat([target_pos_b, left_quat_b], dim=-1)
        right_arm_hold = torch.cat([right_pos_b, right_quat_b], dim=-1)

    policy_hz = 240 // policy_interval
    if args_cli.analytical:
        mode = "analytical IK + trapezoidal profile"
    elif args_cli.absolute:
        mode = "absolute IK"
    else:
        mode = "relative IK + P-controller"
    gains_info = "USD gains" if args_cli.usd_gains else "SDK gains"
    print(f"\n{'=' * 60}")
    print(f"  Tianji + Wuji Interface Test ({mode})")
    print(f"  Policy: {policy_hz} Hz  |  Physics: 240 Hz  |  {gains_info}")
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
    total_sim_steps = args_cli.max_steps * policy_interval
    action = None

    for sim_step in range(total_sim_steps):
        policy_step = sim_step // policy_interval
        is_policy_step = (sim_step % policy_interval == 0)

        # Recompute action at policy rate (or interpolate for analytical)
        if args_cli.analytical:
            # Trapezoidal velocity profile: smooth ramp from start to target joints
            sim_time = sim_step / 240.0
            joint_deltas = left_arm_target_joints - left_arm_start_joints
            frac = _compute_trapz_fraction(sim_time, joint_deltas[0], max_joint_vel, max_joint_accel)
            left_arm_cmd = left_arm_start_joints + frac * joint_deltas
            right_arm_cmd = right_arm_fixed

            alpha_finger = min(1.0, policy_step / 30.0)
            left_hand_cmd = alpha_finger * finger_target
            right_hand_cmd = torch.zeros((n, 20), device=device)
            action = torch.cat([left_arm_cmd, right_arm_cmd, left_hand_cmd, right_hand_cmd], dim=-1)
        elif is_policy_step:
            if args_cli.absolute:
                left_arm_cmd = left_arm_target.clone()
                right_arm_cmd = right_arm_hold.clone()
            else:
                # Relative IK + P-controller
                curr_pos_b, curr_quat_b = _ee_pose_in_base(env, "left_link7", ik_offset_pos, ik_offset_rot)
                pos_err, rot_err = math_utils.compute_pose_error(
                    curr_pos_b, curr_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle",
                )
                left_arm_cmd = torch.cat((gain * pos_err, 0.3 * rot_err), dim=-1)
                left_arm_cmd[:, :3] = torch.clamp(left_arm_cmd[:, :3], -max_delta, max_delta)
                left_arm_cmd[:, 3:] = torch.clamp(left_arm_cmd[:, 3:], -0.12, 0.12)
                right_arm_cmd = torch.zeros((n, 6), device=device)

            alpha = min(1.0, policy_step / 30.0)
            left_hand_cmd = alpha * finger_target
            right_hand_cmd = torch.zeros((n, 20), device=device)

            action = torch.cat([left_arm_cmd, right_arm_cmd, left_hand_cmd, right_hand_cmd], dim=-1)

        markers.visualize(translations=target_pos_w, marker_indices=marker_idx)
        env.step(action)

        # Log at policy rate
        if is_policy_step:
            ee_w = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
            dist = torch.linalg.vector_norm(ee_w - target_pos_w, dim=-1)[0].item()
            if dist < args_cli.threshold and not converged:
                converged = True
                converge_step = policy_step
            print(f"  step {policy_step:3d}  |  ee_dist={dist:.4f}m")

        # Capture video at ~60fps (every 4th sim step from 240Hz)
        if record_video and sim_step % 4 == 0:
            try:
                frames.append(_camera_rgb(env, "overhead_cam"))
            except KeyError:
                pass

        if converged and policy_step - converge_step > 10:
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
