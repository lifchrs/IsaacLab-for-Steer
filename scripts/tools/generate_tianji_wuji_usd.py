#!/usr/bin/env python3
"""Generate a pre-combined Tianji + Wuji hand USD file.

Replaces the Robotiq 2F-85 grippers with Wuji dexterous hands and saves
the result as a single USD file. Run once — the output is used by
TIANJI_WUJI_CFG and does not need runtime stage edits.

Usage:
    PYTHONUNBUFFERED=1 /path/to/python scripts/tools/generate_tianji_wuji_usd.py
"""

from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

REPO_ROOT = Path(__file__).resolve().parents[2]
TIANJI_USD = str(REPO_ROOT / "assets/tianji/tianji_marvin_CCS_with_robotiq_2f85.usd")
WUJI_LEFT_USD = str(REPO_ROOT / "assets/wuji_hand/usd/left/wujihand.usd")
WUJI_RIGHT_USD = str(REPO_ROOT / "assets/wuji_hand/usd/right/wujihand.usd")
OUTPUT_USD = str(REPO_ROOT / "assets/tianji/tianji_wuji.usd")


def _deactivate(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"  Deactivated: {path}")


def _add_fixed_joint(stage, joint_path, body0, body1,
                     local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
                     local_rot0=(1, 0, 0, 0), local_rot1=(1, 0, 0, 0)):
    prim = stage.DefinePrim(joint_path, "PhysicsFixedJoint")
    joint = UsdPhysics.FixedJoint(prim)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*local_pos0))
    prim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*local_pos1))
    prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(*local_rot0))
    prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(*local_rot1))
    prim.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float).Set(3.4028235e38)
    prim.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float).Set(3.4028235e38)
    print(f"  Fixed joint: {joint_path}")


def main():
    print(f"Loading: {TIANJI_USD}")
    stage = Usd.Stage.Open(TIANJI_USD)

    root = "/World/marvin_robot"

    # Deactivate Robotiq grippers and flange joints
    _deactivate(stage, f"{root}/left_gripper")
    _deactivate(stage, f"{root}/right_gripper")
    _deactivate(stage, f"{root}/left_link7/left_flange/left_flange_joint")
    _deactivate(stage, f"{root}/right_link7/right_flange/right_flange_joint")

    # Add Wuji hand references
    left_path = f"{root}/left_wuji_hand"
    stage.DefinePrim(left_path).GetReferences().AddReference(WUJI_LEFT_USD)
    print(f"  Added left hand: {left_path}")

    right_path = f"{root}/right_wuji_hand"
    stage.DefinePrim(right_path).GetReferences().AddReference(WUJI_RIGHT_USD)
    print(f"  Added right hand: {right_path}")

    # Deactivate hands' root joints (so they join the Tianji articulation)
    _deactivate(stage, f"{left_path}/root_joint")
    _deactivate(stage, f"{right_path}/root_joint")

    # Fixed joints: flange -> palm
    # Robotiq fingers pointed along +X in base_link frame.
    # Wuji fingers point along +Z in palm_link frame.
    # The original Robotiq joint used localRot1=(0.707, 0, 0.707, 0) = 90° around Y
    # to align +X(gripper) with the flange.
    # For Wuji, fingers are along +Z, so we need identity rotation (the flange
    # already points in the direction where +Z should go).
    _add_fixed_joint(
        stage,
        f"{root}/left_link7/left_flange/left_wuji_joint",
        f"{root}/left_link7/left_flange",
        f"{left_path}/left_palm_link",
        local_rot1=(1, 0, 0, 0),
    )
    _add_fixed_joint(
        stage,
        f"{root}/right_link7/right_flange/right_wuji_joint",
        f"{root}/right_link7/right_flange",
        f"{right_path}/right_palm_link",
        local_rot1=(1, 0, 0, 0),
    )

    # Save
    stage.GetRootLayer().Export(OUTPUT_USD)
    print(f"\nSaved: {OUTPUT_USD}")


if __name__ == "__main__":
    main()
    simulation_app.close()
