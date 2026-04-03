"""Spawn the TianJi dual-arm robot with Wuji dexterous hands and apply random actions.

This script replaces the Robotiq 2F-85 grippers on the TianJi arms with Wuji 5-finger
dexterous hands by modifying the USD stage at setup time, then runs random joint efforts.

Usage:
    ./isaaclab.sh -p scripts/demos/tianji_wuji_random_actions.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TianJi arms + Wuji hands random actions demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from pathlib import Path

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

# Asset paths
REPO_ROOT = Path(__file__).parent.parent.parent
TIANJI_USD = str(REPO_ROOT / "assets/tianji/tianji_marvin_CCS_with_robotiq_2f85.usd")
WUJI_LEFT_USD = str(REPO_ROOT / "assets/wuji_hand/usd/left/wujihand.usd")
WUJI_RIGHT_USD = str(REPO_ROOT / "assets/wuji_hand/usd/right/wujihand.usd")


def _deactivate_prim(stage, prim_path: str):
    """Deactivate a prim (make it invisible to physics and rendering)."""
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"[INFO] Deactivated: {prim_path}")
    else:
        print(f"[WARN] Prim not found: {prim_path}")


def _add_fixed_joint(stage, joint_path: str, body0_path: str, body1_path: str,
                     local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
                     local_rot0=(1, 0, 0, 0), local_rot1=(1, 0, 0, 0)):
    """Add a fixed joint between two bodies."""
    joint_prim = stage.DefinePrim(joint_path, "PhysicsFixedJoint")
    joint = UsdPhysics.FixedJoint(joint_prim)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint_prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(*local_pos0))
    joint_prim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(
        Gf.Vec3f(*local_pos1))
    joint_prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(
        Gf.Quatf(*local_rot0))
    joint_prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(
        Gf.Quatf(*local_rot1))
    joint_prim.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(3.4028235e38)
    joint_prim.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(3.4028235e38)
    print(f"[INFO] Created fixed joint: {joint_path}")


def stitch_wuji_hands(stage, robot_root: str):
    """Replace Robotiq grippers with Wuji hands on the TianJi robot.

    1. Deactivate left/right Robotiq gripper prims
    2. Add Wuji hand USD references under the robot
    3. Create fixed joints from arm flanges to hand palms
    """
    # The TianJi robot lives under {robot_root}/marvin_robot/
    marvin = f"{robot_root}/marvin_robot"

    # --- Deactivate Robotiq grippers ---
    _deactivate_prim(stage, f"{marvin}/left_gripper")
    _deactivate_prim(stage, f"{marvin}/right_gripper")
    # Also deactivate the flange fixed joints that connected to the grippers
    _deactivate_prim(stage, f"{marvin}/left_link7/left_flange/left_flange_joint")
    _deactivate_prim(stage, f"{marvin}/right_link7/right_flange/right_flange_joint")

    # --- Add Wuji left hand ---
    left_hand_path = f"{marvin}/left_wuji_hand"
    left_hand_prim = stage.DefinePrim(left_hand_path)
    left_hand_prim.GetReferences().AddReference(WUJI_LEFT_USD)

    # --- Add Wuji right hand ---
    right_hand_path = f"{marvin}/right_wuji_hand"
    right_hand_prim = stage.DefinePrim(right_hand_path)
    right_hand_prim.GetReferences().AddReference(WUJI_RIGHT_USD)

    # --- Remove the Wuji hands' own root_joint so they don't create separate articulations ---
    # The hands have a root_joint (fixed joint to world) that we must deactivate
    # so they become part of the TianJi articulation tree instead.
    _deactivate_prim(stage, f"{left_hand_path}/root_joint")
    _deactivate_prim(stage, f"{right_hand_path}/root_joint")

    # --- Create fixed joints from flanges to hand palms ---
    # The flange is at left_link7/left_flange; the original joint connected to
    # the Robotiq base_link with localRot1 = (0.707, 0, 0.707, 0) (90-deg rotation).
    # We use the same attachment point and orientation for the Wuji palm.

    # Left hand: flange -> left_palm_link
    _add_fixed_joint(
        stage,
        joint_path=f"{marvin}/left_link7/left_flange/left_wuji_joint",
        body0_path=f"{marvin}/left_link7/left_flange",
        body1_path=f"{left_hand_path}/left_palm_link",
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        local_rot0=(1, 0, 0, 0),
        local_rot1=(0.70710677, 0, 0.70710677, 0),
    )

    # Right hand: flange -> right_palm_link
    _add_fixed_joint(
        stage,
        joint_path=f"{marvin}/right_link7/right_flange/right_wuji_joint",
        body0_path=f"{marvin}/right_link7/right_flange",
        body1_path=f"{right_hand_path}/right_palm_link",
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        local_rot0=(1, 0, 0, 0),
        local_rot1=(0.70710677, 0, 0.70710677, 0),
    )

    print("[INFO] Wuji hands stitched to TianJi arms.")


# -- Combined ArticulationCfg (TianJi arms + Wuji hand joints) --
# We load the original TianJi USD; the gripper removal and hand stitching
# happens via USD stage edits after the prim is spawned but before sim.reset().

TIANJI_WUJI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TIANJI_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            # Left arm
            "left_joint1": 0.0,
            "left_joint2": 0.0,
            "left_joint3": 0.0,
            "left_joint4": 0.0,
            "left_joint5": 0.0,
            "left_joint6": 0.0,
            "left_joint7": 0.0,
            # Right arm
            "right_joint1": 0.0,
            "right_joint2": 0.0,
            "right_joint3": 0.0,
            "right_joint4": 0.0,
            "right_joint5": 0.0,
            "right_joint6": 0.0,
            "right_joint7": 0.0,
            # Left Wuji hand (20 joints)
            "left_finger1_joint1": 0.5,
            "left_finger1_joint2": 0.3,
            "left_finger1_joint3": 0.3,
            "left_finger1_joint4": 0.3,
            "left_finger2_joint1": 0.3,
            "left_finger2_joint2": 0.0,
            "left_finger2_joint3": 0.3,
            "left_finger2_joint4": 0.3,
            "left_finger3_joint1": 0.3,
            "left_finger3_joint2": 0.0,
            "left_finger3_joint3": 0.3,
            "left_finger3_joint4": 0.3,
            "left_finger4_joint1": 0.3,
            "left_finger4_joint2": 0.0,
            "left_finger4_joint3": 0.3,
            "left_finger4_joint4": 0.3,
            "left_finger5_joint1": 0.3,
            "left_finger5_joint2": 0.0,
            "left_finger5_joint3": 0.3,
            "left_finger5_joint4": 0.3,
            # Right Wuji hand (20 joints)
            "right_finger1_joint1": 0.5,
            "right_finger1_joint2": 0.3,
            "right_finger1_joint3": 0.3,
            "right_finger1_joint4": 0.3,
            "right_finger2_joint1": 0.3,
            "right_finger2_joint2": 0.0,
            "right_finger2_joint3": 0.3,
            "right_finger2_joint4": 0.3,
            "right_finger3_joint1": 0.3,
            "right_finger3_joint2": 0.0,
            "right_finger3_joint3": 0.3,
            "right_finger3_joint4": 0.3,
            "right_finger4_joint1": 0.3,
            "right_finger4_joint2": 0.0,
            "right_finger4_joint3": 0.3,
            "right_finger4_joint4": 0.3,
            "right_finger5_joint1": 0.3,
            "right_finger5_joint2": 0.0,
            "right_finger5_joint3": 0.3,
            "right_finger5_joint4": 0.3,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        # --- Arm actuators ---
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-4]"],
            effort_limit=108.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "left_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[5-7]"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[1-4]"],
            effort_limit=108.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "right_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[5-7]"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        # --- Wuji hand actuators ---
        "left_thumb": ImplicitActuatorCfg(
            joint_names_expr=["left_finger1_joint[1-4]"],
            effort_limit=0.45,
            velocity_limit=13.0,
            stiffness=0.5,
            damping=0.05,
        ),
        "left_fingers": ImplicitActuatorCfg(
            joint_names_expr=["left_finger[2-5]_joint[1-4]"],
            effort_limit=0.65,
            velocity_limit=13.0,
            stiffness=0.5,
            damping=0.05,
        ),
        "right_thumb": ImplicitActuatorCfg(
            joint_names_expr=["right_finger1_joint[1-4]"],
            effort_limit=0.45,
            velocity_limit=13.0,
            stiffness=0.5,
            damping=0.05,
        ),
        "right_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[2-5]_joint[1-4]"],
            effort_limit=0.65,
            velocity_limit=13.0,
            stiffness=0.5,
            damping=0.05,
        ),
    },
)


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene with TianJi robot + Wuji hands."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Single origin
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Spawn TianJi robot (with Robotiq grippers still in the USD)
    tianji_wuji_cfg = TIANJI_WUJI_CFG.copy()
    tianji_wuji_cfg.prim_path = "/World/Origin.*/robot"
    robot = Articulation(cfg=tianji_wuji_cfg)

    # Now modify the USD stage: remove grippers, add Wuji hands, stitch with fixed joints
    stage = prim_utils.get_current_stage()
    stitch_wuji_hands(stage, "/World/Origin1/robot")

    scene_entities = {"robot": robot}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop with random joint efforts."""
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # Print joint info
    print(f"[INFO] Number of joints: {robot.num_joints}")
    print(f"[INFO] Joint names: {robot.joint_names}")

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print(f"[INFO]: Resetting robot state... (joints: {robot.num_joints})")

        # Apply random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()

        # Step
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
