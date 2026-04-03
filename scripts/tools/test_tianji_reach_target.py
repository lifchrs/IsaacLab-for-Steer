#!/usr/bin/env python3
"""Test positional control for Tianji arms by reaching a visible ball target.

Creates a minimal scene (robot + ground plane + red sphere target), then uses
DifferentialIK in *absolute* mode — the controller receives the target pose
directly and solves for joint positions each step.  No hand-rolled P-controller.

Usage:
    python scripts/tools/test_tianji_reach_target.py --headless --enable_cameras
    python scripts/tools/test_tianji_reach_target.py --headless --enable_cameras --target_pos 0.25 0.8 0.4
    python scripts/tools/test_tianji_reach_target.py --headless --enable_cameras --arm right
    python scripts/tools/test_tianji_reach_target.py --headless --no_video
    python scripts/tools/test_tianji_reach_target.py --livestream 1
"""

import argparse
import os
import sys
from dataclasses import MISSING
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tianji reach-target positional control test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=150, help="Max controller steps.")
parser.add_argument("--threshold", type=float, default=0.02, help="Success distance (m).")
parser.add_argument("--target_pos", type=float, nargs=3, default=None,
                    help="Target xyz in env frame. Random if omitted.")
parser.add_argument("--arm", choices=["left", "right", "both"], default="left")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="./videos/reach_test")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--no_video", action="store_true", help="Skip video recording.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (must come after AppLauncher)
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.tianji import TIANJI_CFG

# ---------------------------------------------------------------------------
# Minimal environment config — robot + ground + camera
# ---------------------------------------------------------------------------

EE_FRAME_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(
    pos=[0.0, 0.0, 0.2414], rot=[0.0, 0.0, 0.0, 1.0],
)


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = TIANJI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_gripper/Robotiq_2F_85/base_link",
                name="end_effector",
                offset=OffsetCfg(pos=(0.1534, 0.0, 0.0), rot=(0.0, 0.7071068, 0.0, 0.7071068)),
            ),
        ],
    )
    right_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_gripper/Robotiq_2F_85/base_link",
                name="end_effector",
                offset=OffsetCfg(pos=(0.1534, 0.0, 0.0), rot=(0.0, 0.7071068, 0.0, 0.7071068)),
            ),
        ],
    )
    overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.2, horizontal_aperture=5.376),
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 0.5, 0.8),
            rot=(0.426, 0.227, 0.435, 0.755),
            convention="opengl",
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    left_arm_action: ActionTermCfg = MISSING
    right_arm_action: ActionTermCfg = MISSING


@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ReachEnvCfg(ManagerBasedEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False)
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events = None
    commands = None

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # Absolute pose IK: action = [x, y, z, qw, qx, qy, qz] in base frame
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_joint.*"],
            body_name="left_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            scale=1.0,
            body_offset=EE_FRAME_OFFSET,
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_joint.*"],
            body_name="right_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            scale=1.0,
            body_offset=EE_FRAME_OFFSET,
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
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    return image


def _camera_rgb(env, camera_name: str) -> np.ndarray:
    camera = env.scene[camera_name]
    for key in ("rgb", "rgba"):
        if key in camera.data.output:
            return _to_uint8_rgb(camera.data.output[key])
    raise KeyError(f"Camera '{camera_name}' has no rgb/rgba.")


def _write_video(frames: list[np.ndarray], path: str, fps: int):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved {len(frames)} frames -> {path}")


def _ee_pos_world(env, body_name: str, offset_pos, offset_rot):
    """EE position in world frame, including the body offset."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    ee_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[:, body_ids[0]]
    if offset_pos is not None:
        ee_pos_w, _ = math_utils.combine_frame_transforms(ee_pos_w, ee_quat_w, offset_pos, offset_rot)
    return ee_pos_w


def _ee_pose_in_base(env, body_name: str, offset_pos, offset_rot):
    """EE pose in robot base frame (matches what IK action sees internally)."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(body_name)
    ee_pos_w = robot.data.body_pos_w[:, body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[:, body_ids[0]]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    if offset_pos is not None:
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, offset_pos, offset_rot
        )
    return ee_pos_b, ee_quat_b


def _target_in_base(env, target_pos_w, target_quat_w):
    """Convert world-frame target to robot base frame."""
    robot = env.scene["robot"]
    return math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, target_quat_w
    )


def _compose_action(env, left_cmd, right_cmd):
    """Build full action tensor from per-arm commands."""
    chunks = []
    for term_name in env.action_manager.active_terms:
        if term_name == "left_arm_action":
            chunks.append(left_cmd)
        elif term_name == "right_arm_action":
            chunks.append(right_cmd)
        else:
            dim = env.action_manager.get_term(term_name).action_dim
            chunks.append(torch.zeros((env.num_envs, dim), device=env.device))
    return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = ReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()

    device = env.device
    n = args_cli.num_envs

    # Video setup
    record_video = not args_cli.no_video
    frames: list[np.ndarray] = []
    output_dir = ""
    if record_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args_cli.output_dir, f"reach_{args_cli.arm}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

    # IK body offset tensors
    ik_offset_pos = torch.tensor([[0.0, 0.0, 0.2414]], device=device).repeat(n, 1)
    ik_offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(n, 1)
    arm_body = {"left": "left_link7", "right": "right_link7"}

    # --- Determine target ---
    if args_cli.target_pos is not None:
        target_pos_env = torch.tensor([args_cli.target_pos], dtype=torch.float32, device=device)
    else:
        init_arm = "left" if args_cli.arm != "right" else "right"
        curr_ee_w = _ee_pos_world(env, arm_body[init_arm], ik_offset_pos, ik_offset_rot)
        curr_pos_env = curr_ee_w - env.scene.env_origins
        target_pos_env = curr_pos_env + torch.tensor([[0.08, -0.05, 0.05]], device=device)

    target_pos_w = target_pos_env + env.scene.env_origins

    # Get current EE pose in base frame for each arm
    arm_poses_b = {}
    for arm_name in ["left", "right"]:
        pos_b, quat_b = _ee_pose_in_base(env, arm_body[arm_name], ik_offset_pos, ik_offset_rot)
        arm_poses_b[arm_name] = (pos_b, quat_b)

    # Use the active arm's CURRENT orientation as the target orientation
    # (avoids impossible orientation targets that cause IK divergence)
    init_arm = "left" if args_cli.arm != "right" else "right"
    _, init_quat_b = arm_poses_b[init_arm]
    target_quat_w = math_utils.quat_mul(
        env.scene["robot"].data.root_quat_w, init_quat_b
    )  # base-frame quat back to world

    # Target in base frame — this is what we send as the absolute IK command
    target_pos_b, target_quat_b = _target_in_base(env, target_pos_w, target_quat_w)
    target_action = torch.cat([target_pos_b, target_quat_b], dim=-1)  # (n, 7)

    # "Hold still" command for the idle arm = its current pose
    left_hold = torch.cat(arm_poses_b["left"], dim=-1)
    right_hold = torch.cat(arm_poses_b["right"], dim=-1)

    # --- Spawn red sphere ---
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetBall",
        markers={
            "sphere": SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
        },
    )
    markers = VisualizationMarkers(marker_cfg)
    marker_idx = torch.zeros(n, dtype=torch.long, device=device)
    markers.visualize(translations=target_pos_w, marker_indices=marker_idx)

    init_arm = "left" if args_cli.arm != "right" else "right"
    ee_w_init = _ee_pos_world(env, arm_body[init_arm], ik_offset_pos, ik_offset_rot)
    print(f"\n{'=' * 60}")
    print(f"  Tianji Reach Test (Absolute IK)")
    print(f"  Arm(s): {args_cli.arm}")
    print(f"  EE start (world): {ee_w_init[0].cpu().numpy()}")
    print(f"  Target  (world):  {target_pos_w[0].cpu().numpy()}")
    print(f"  Threshold: {args_cli.threshold} m")
    print(f"  Max steps: {args_cli.max_steps}")
    if record_video:
        print(f"  Recording to: {output_dir}")
    print(f"{'=' * 60}\n")

    # --- Which arms to test ---
    test_arms = []
    if args_cli.arm in ("left", "both"):
        test_arms.append(("left", "left_ee_frame"))
    if args_cli.arm in ("right", "both"):
        test_arms.append(("right", "right_ee_frame"))

    converged = {name: False for name, _ in test_arms}
    converge_step = {name: -1 for name, _ in test_arms}
    dist_log: dict[str, list[float]] = {name: [] for name, _ in test_arms}

    # --- Control loop ---
    # Interpolation: ramp the commanded pose from current toward goal over time.
    # Each step, the IK target moves a fraction closer — keeps deltas small & smooth.
    ramp_steps = 40  # steps to fully ramp to the final target

    for step in range(args_cli.max_steps):
        parts = [f"step {step:3d}"]
        alpha = min(1.0, step / ramp_steps)  # 0 → 1 over ramp_steps

        left_cmd = left_hold.clone()
        right_cmd = right_hold.clone()

        for arm_name, frame_name in test_arms:
            # Interpolate: blend from initial pose toward target
            init_pose = left_hold if arm_name == "left" else right_hold
            interp_pos = init_pose[:, :3] + alpha * (target_action[:, :3] - init_pose[:, :3])
            interp_quat = math_utils.quat_slerp(init_pose[:, 3:], target_action[:, 3:], alpha)
            interp_cmd = torch.cat([interp_pos, interp_quat], dim=-1)

            if arm_name == "left":
                left_cmd = interp_cmd
            else:
                right_cmd = interp_cmd

            # Measure distance to final target
            ee_pos_w = _ee_pos_world(env, arm_body[arm_name], ik_offset_pos, ik_offset_rot)
            dist = torch.linalg.vector_norm(ee_pos_w - target_pos_w, dim=-1)
            d = dist[0].item()
            dist_log[arm_name].append(d)
            parts.append(f"{arm_name}_dist={d:.4f}m")
            if d < args_cli.threshold and not converged[arm_name]:
                converged[arm_name] = True
                converge_step[arm_name] = step

        print("  |  ".join(parts))

        markers.visualize(translations=target_pos_w, marker_indices=marker_idx)
        actions = _compose_action(env, left_cmd, right_cmd)
        env.step(actions)

        if record_video:
            try:
                frames.append(_camera_rgb(env, "overhead_cam"))
            except KeyError:
                pass

        if all(converged.values()) and step - max(converge_step.values()) > 10:
            break

    # --- Write outputs ---
    if record_video and frames:
        print(f"\n  Writing video...")
        _write_video(frames, os.path.join(output_dir, "overhead_cam.mp4"), args_cli.fps)

        csv_path = os.path.join(output_dir, "distance_log.csv")
        with open(csv_path, "w") as f:
            f.write("step," + ",".join(f"{name}_dist" for name, _ in test_arms) + "\n")
            for i in range(len(dist_log[test_arms[0][0]])):
                f.write(",".join([str(i)] + [f"{dist_log[nm][i]:.6f}" for nm, _ in test_arms]) + "\n")
        print(f"  Saved distance log -> {csv_path}")

    # --- Results ---
    print(f"\n{'=' * 60}")
    print("  Results:")
    all_passed = True
    for arm_name, frame_name in test_arms:
        ee_w = _ee_pos_world(env, arm_body[arm_name], ik_offset_pos, ik_offset_rot)
        final_dist = torch.linalg.vector_norm(ee_w - target_pos_w, dim=-1)[0].item()
        passed = final_dist < args_cli.threshold
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        info = f"converged at step {converge_step[arm_name]}" if converged[arm_name] else "did not converge"
        print(f"  [{status}] {arm_name} arm: final_dist={final_dist:.4f}m  ({info})")
    if record_video and output_dir:
        print(f"\n  Videos saved to: {output_dir}")
    print(f"{'=' * 60}\n")

    env.close()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
    simulation_app.close()
