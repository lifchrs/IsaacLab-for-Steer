"""Combined Tianji dual-arm robot with Wuji dexterous hands.

Provides:
- ``TIANJI_WUJI_CFG``: ArticulationCfg for the stitched robot (14 arm + 40 hand joints).
- ``stitch_wuji_hands()``: USD stage edit that replaces Robotiq grippers with Wuji hands.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
TIANJI_USD = str(_REPO_ROOT / "assets/tianji/tianji_marvin_CCS_with_robotiq_2f85.usd")
WUJI_LEFT_USD = str(_REPO_ROOT / "assets/wuji_hand/usd/left/wujihand.usd")
WUJI_RIGHT_USD = str(_REPO_ROOT / "assets/wuji_hand/usd/right/wujihand.usd")

# ---------------------------------------------------------------------------
# USD stage stitching
# ---------------------------------------------------------------------------


def _deactivate_prim(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)


def _add_fixed_joint(stage, joint_path, body0_path, body1_path,
                     local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
                     local_rot0=(1, 0, 0, 0), local_rot1=(1, 0, 0, 0)):
    from pxr import UsdPhysics, Sdf, Gf

    joint_prim = stage.DefinePrim(joint_path, "PhysicsFixedJoint")
    joint = UsdPhysics.FixedJoint(joint_prim)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint_prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*local_pos0))
    joint_prim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*local_pos1))
    joint_prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(*local_rot0))
    joint_prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(*local_rot1))
    joint_prim.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(3.4028235e38)
    joint_prim.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(3.4028235e38)


def stitch_wuji_hands(stage, robot_root: str):
    """Replace Robotiq grippers with Wuji hands on the Tianji robot.

    Must be called after the robot prim is spawned but before ``sim.reset()``.

    Args:
        stage: The current USD stage (``prim_utils.get_current_stage()``).
        robot_root: Prim path of the robot root, e.g. ``"/World/envs/env_0/Robot"``.
    """
    marvin = f"{robot_root}/marvin_robot"

    # Deactivate Robotiq grippers and their flange joints
    _deactivate_prim(stage, f"{marvin}/left_gripper")
    _deactivate_prim(stage, f"{marvin}/right_gripper")
    _deactivate_prim(stage, f"{marvin}/left_link7/left_flange/left_flange_joint")
    _deactivate_prim(stage, f"{marvin}/right_link7/right_flange/right_flange_joint")

    # Add Wuji hand USD references
    left_hand_path = f"{marvin}/left_wuji_hand"
    stage.DefinePrim(left_hand_path).GetReferences().AddReference(WUJI_LEFT_USD)
    right_hand_path = f"{marvin}/right_wuji_hand"
    stage.DefinePrim(right_hand_path).GetReferences().AddReference(WUJI_RIGHT_USD)

    # Remove their root joints so they become part of the Tianji articulation
    _deactivate_prim(stage, f"{left_hand_path}/root_joint")
    _deactivate_prim(stage, f"{right_hand_path}/root_joint")

    # Fixed joints: flange → hand palm
    _add_fixed_joint(
        stage,
        joint_path=f"{marvin}/left_link7/left_flange/left_wuji_joint",
        body0_path=f"{marvin}/left_link7/left_flange",
        body1_path=f"{left_hand_path}/left_palm_link",
        local_rot1=(0.70710677, 0, 0.70710677, 0),
    )
    _add_fixed_joint(
        stage,
        joint_path=f"{marvin}/right_link7/right_flange/right_wuji_joint",
        body0_path=f"{marvin}/right_link7/right_flange",
        body1_path=f"{right_hand_path}/right_palm_link",
        local_rot1=(0.70710677, 0, 0.70710677, 0),
    )


# ---------------------------------------------------------------------------
# Combined ArticulationCfg
# ---------------------------------------------------------------------------

TIANJI_WUJI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
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
            # Left arm (7)
            "left_joint1": 0.0, "left_joint2": 0.0, "left_joint3": 0.0, "left_joint4": 0.0,
            "left_joint5": 0.0, "left_joint6": 0.0, "left_joint7": 0.0,
            # Right arm (7)
            "right_joint1": 0.0, "right_joint2": 0.0, "right_joint3": 0.0, "right_joint4": 0.0,
            "right_joint5": 0.0, "right_joint6": 0.0, "right_joint7": 0.0,
            # Left Wuji hand (20)
            "left_finger1_joint1": 0.5, "left_finger1_joint2": 0.3,
            "left_finger1_joint3": 0.3, "left_finger1_joint4": 0.3,
            "left_finger2_joint1": 0.3, "left_finger2_joint2": 0.0,
            "left_finger2_joint3": 0.3, "left_finger2_joint4": 0.3,
            "left_finger3_joint1": 0.3, "left_finger3_joint2": 0.0,
            "left_finger3_joint3": 0.3, "left_finger3_joint4": 0.3,
            "left_finger4_joint1": 0.3, "left_finger4_joint2": 0.0,
            "left_finger4_joint3": 0.3, "left_finger4_joint4": 0.3,
            "left_finger5_joint1": 0.3, "left_finger5_joint2": 0.0,
            "left_finger5_joint3": 0.3, "left_finger5_joint4": 0.3,
            # Right Wuji hand (20)
            "right_finger1_joint1": 0.5, "right_finger1_joint2": 0.3,
            "right_finger1_joint3": 0.3, "right_finger1_joint4": 0.3,
            "right_finger2_joint1": 0.3, "right_finger2_joint2": 0.0,
            "right_finger2_joint3": 0.3, "right_finger2_joint4": 0.3,
            "right_finger3_joint1": 0.3, "right_finger3_joint2": 0.0,
            "right_finger3_joint3": 0.3, "right_finger3_joint4": 0.3,
            "right_finger4_joint1": 0.3, "right_finger4_joint2": 0.0,
            "right_finger4_joint3": 0.3, "right_finger4_joint4": 0.3,
            "right_finger5_joint1": 0.3, "right_finger5_joint2": 0.0,
            "right_finger5_joint3": 0.3, "right_finger5_joint4": 0.3,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        # Arms
        "left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-4]"], effort_limit=108.0, velocity_limit=3.14, stiffness=400.0, damping=80.0,
        ),
        "left_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[5-7]"], effort_limit=18.0, velocity_limit=3.14, stiffness=400.0, damping=80.0,
        ),
        "right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[1-4]"], effort_limit=108.0, velocity_limit=3.14, stiffness=400.0, damping=80.0,
        ),
        "right_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[5-7]"], effort_limit=18.0, velocity_limit=3.14, stiffness=400.0, damping=80.0,
        ),
        # Hands
        "left_thumb": ImplicitActuatorCfg(
            joint_names_expr=["left_finger1_joint[1-4]"], effort_limit=0.45, velocity_limit=13.0, stiffness=0.5, damping=0.05,
        ),
        "left_fingers": ImplicitActuatorCfg(
            joint_names_expr=["left_finger[2-5]_joint[1-4]"], effort_limit=0.65, velocity_limit=13.0, stiffness=0.5, damping=0.05,
        ),
        "right_thumb": ImplicitActuatorCfg(
            joint_names_expr=["right_finger1_joint[1-4]"], effort_limit=0.45, velocity_limit=13.0, stiffness=0.5, damping=0.05,
        ),
        "right_fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[2-5]_joint[1-4]"], effort_limit=0.65, velocity_limit=13.0, stiffness=0.5, damping=0.05,
        ),
    },
)
