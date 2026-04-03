#!/usr/bin/env python3
"""Test the unified Tianji arm + Wuji hand positional control interface.

The interface accepts a 52D action:
  [0:6]   left arm wrist pose  (3 pos + 3 axis-angle, IK-relative)
  [6:12]  right arm wrist pose (3 pos + 3 axis-angle, IK-relative)
  [12:32] left hand finger joints  (20 direct joint positions)
  [32:52] right hand finger joints (20 direct joint positions)

This script:
  1. Spawns Tianji + Wuji as a single articulation
  2. Places a red ball at a target location
  3. Sends IK commands to move the left arm EE to the ball
  4. Simultaneously curls the left hand fingers
  5. Reports PASS/FAIL and records video

Usage:
    conda run -n tianji_wuji python scripts/tools/test_tianji_wuji_interface.py --headless --enable_cameras
    conda run -n tianji_wuji python scripts/tools/test_tianji_wuji_interface.py --headless --no_video
"""

import argparse
import os
import sys
from dataclasses import MISSING
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Tianji+Wuji unified control interface.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--threshold", type=float, default=0.03, help="EE convergence threshold (m).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="./videos/tianji_wuji_test")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--no_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg, EventTermCfg, ObservationGroupCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.tianji_wuji import TIANJI_WUJI_CFG, stitch_wuji_hands

# ---------------------------------------------------------------------------
# EE offset from link7 toward the Wuji palm
# ---------------------------------------------------------------------------

WUJI_EE_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(
    pos=[0.0, 0.0, 0.107], rot=[0.0, 0.0, 0.0, 1.0],
)

# ---------------------------------------------------------------------------
# Event: stitch hands at startup
# ---------------------------------------------------------------------------


def _stitch_hands_event(env, env_ids):
    """Event callback that stitches Wuji hands onto the Tianji robot."""
    stage = prim_utils.get_current_stage()
    for i in range(env.num_envs):
        robot_root = f"/World/envs/env_{i}/Robot"
        stitch_wuji_hands(stage, robot_root)


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class TianjiWujiSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = TIANJI_WUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.6, 0.8, 0.8),
        ),
    )
    left_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_link7/left_flange",
                name="left_ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ],
    )
    right_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_link7/right_flange",
                name="right_ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ],
    )
    overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.2, horizontal_aperture=5.376),
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.5, 0.8), rot=(0.426, 0.227, 0.435, 0.755), convention="opengl"),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


@configclass
class ActionsCfg:
    """52D action: 6 left IK + 6 right IK + 20 left hand + 20 right hand."""
    left_arm_action: ActionTermCfg = MISSING
    right_arm_action: ActionTermCfg = MISSING
    left_hand_action: ActionTermCfg = MISSING
    right_hand_action: ActionTermCfg = MISSING


@configclass
class EventsCfg:
    stitch_hands = EventTermCfg(func=_stitch_hands_event, mode="prestartup")


@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class TianjiWujiEnvCfg(ManagerBasedEnvCfg):
    scene: TianjiWujiSceneCfg = TianjiWujiSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False)
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventsCfg = EventsCfg()
    commands = None

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # IK relative for arms (6D: 3 pos + 3 axis-angle delta)
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_joint.*"],
            body_name="left_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
            ),
            scale=0.5,
            body_offset=WUJI_EE_OFFSET,
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_joint.*"],
            body_name="right_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
            ),
            scale=0.5,
            body_offset=WUJI_EE_OFFSET,
        )

        # Direct joint position for hands (20D each)
        self.actions.left_hand_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_finger.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.right_hand_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_finger.*"],
            scale=1.0,
            use_default_offset=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = TianjiWujiEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
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

    # IK offset tensors
    ik_offset_pos = torch.tensor([[0.0, 0.0, 0.107]], device=device).repeat(n, 1)
    ik_offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(n, 1)

    # Current EE pose in base frame (for computing IK deltas)
    ee_pos_b, ee_quat_b = _ee_pose_in_base(env, "left_link7", ik_offset_pos, ik_offset_rot)
    ee_pos_w_init = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)

    # Target: offset from current EE
    target_offset = torch.tensor([[0.08, -0.05, 0.05]], device=device)
    target_pos_w = ee_pos_w_init + target_offset

    # Target in base frame
    robot = env.scene["robot"]
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        target_pos_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
    )

    # Finger targets: curl all fingers to ~0.8 rad (closed fist)
    finger_target = torch.full((n, 20), 0.8, device=device)

    # Spawn red ball
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetBall",
        markers={"sphere": SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)))},
    )
    markers = VisualizationMarkers(marker_cfg)
    marker_idx = torch.zeros(n, dtype=torch.long, device=device)
    markers.visualize(translations=target_pos_w, marker_indices=marker_idx)

    print(f"\n{'=' * 60}")
    print(f"  Tianji + Wuji Unified Interface Test")
    print(f"  EE start (world): {ee_pos_w_init[0].cpu().numpy()}")
    print(f"  Target   (world): {target_pos_w[0].cpu().numpy()}")
    print(f"  Finger target:    0.8 rad (closed fist)")
    print(f"  EE threshold:     {args_cli.threshold} m")
    if record_video:
        print(f"  Recording to:     {output_dir}")
    print(f"{'=' * 60}\n")

    # P-controller for IK relative commands
    gain = 0.8
    max_delta = 0.02
    converged = False
    converge_step = -1
    dist_log = []

    for step in range(args_cli.max_steps):
        # --- Arm: compute IK relative command ---
        curr_pos_b, curr_quat_b = _ee_pose_in_base(env, "left_link7", ik_offset_pos, ik_offset_rot)
        pos_err, rot_err = math_utils.compute_pose_error(
            curr_pos_b, curr_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle",
        )
        left_arm_cmd = torch.cat((gain * pos_err, 0.3 * rot_err), dim=-1)
        left_arm_cmd[:, :3] = torch.clamp(left_arm_cmd[:, :3], -max_delta, max_delta)
        left_arm_cmd[:, 3:] = torch.clamp(left_arm_cmd[:, 3:], -0.12, 0.12)

        # Right arm: hold still (zero delta)
        right_arm_cmd = torch.zeros((n, 6), device=device)

        # --- Hands: ramp finger targets ---
        alpha = min(1.0, step / 30.0)
        left_hand_cmd = alpha * finger_target
        right_hand_cmd = torch.zeros((n, 20), device=device)  # right hand idle

        # --- Compose 52D action ---
        action = torch.cat([left_arm_cmd, right_arm_cmd, left_hand_cmd, right_hand_cmd], dim=-1)

        # Measure distance
        ee_w = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
        dist = torch.linalg.vector_norm(ee_w - target_pos_w, dim=-1)[0].item()
        dist_log.append(dist)

        # Check finger convergence
        finger_ids, _ = robot.find_joints(["left_finger.*"])
        finger_pos = robot.data.joint_pos[0, finger_ids].cpu()
        finger_err = (finger_pos - alpha * 0.8).abs().mean().item()

        if dist < args_cli.threshold and not converged:
            converged = True
            converge_step = step

        print(f"  step {step:3d}  |  ee_dist={dist:.4f}m  |  finger_err={finger_err:.4f}rad")

        markers.visualize(translations=target_pos_w, marker_indices=marker_idx)
        env.step(action)

        if record_video:
            try:
                frames.append(_camera_rgb(env, "overhead_cam"))
            except KeyError:
                pass

        if converged and step - converge_step > 15:
            break

    # --- Write outputs ---
    if record_video and frames:
        print(f"\n  Writing video...")
        _write_video(frames, os.path.join(output_dir, "overhead_cam.mp4"), args_cli.fps)

        csv_path = os.path.join(output_dir, "distance_log.csv")
        with open(csv_path, "w") as f:
            f.write("step,ee_dist\n")
            for i, d in enumerate(dist_log):
                f.write(f"{i},{d:.6f}\n")
        print(f"  Saved distance log -> {csv_path}")

    # --- Results ---
    ee_final = _ee_pos_world(env, "left_link7", ik_offset_pos, ik_offset_rot)
    final_dist = torch.linalg.vector_norm(ee_final - target_pos_w, dim=-1)[0].item()
    finger_pos = robot.data.joint_pos[0, finger_ids].cpu()
    final_finger_err = (finger_pos - 0.8).abs().mean().item()

    ee_pass = final_dist < args_cli.threshold
    finger_pass = final_finger_err < 0.1

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    status = "PASS" if ee_pass else "FAIL"
    info = f"converged at step {converge_step}" if converged else "did not converge"
    print(f"  [{status}] EE:      final_dist={final_dist:.4f}m  ({info})")
    status = "PASS" if finger_pass else "FAIL"
    print(f"  [{status}] Fingers: mean_err={final_finger_err:.4f}rad")
    if record_video and output_dir:
        print(f"\n  Videos saved to: {output_dir}")
    print(f"{'=' * 60}\n")

    env.close()
    sys.exit(0 if (ee_pass and finger_pass) else 1)


if __name__ == "__main__":
    main()
    simulation_app.close()
