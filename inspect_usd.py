#!/usr/bin/env python3
"""Script to inspect USD structure for articulation joints."""

from pxr import Usd, UsdPhysics
import sys

def inspect_usd(usd_path, search_term=""):
    """Inspect a USD file for articulation roots and joints."""
    print(f"\n=== Inspecting: {usd_path} ===\n")

    stage = Usd.Stage.Open(usd_path)

    articulation_roots = []
    joints = []

    for prim in stage.Traverse():
        path = prim.GetPath().pathString

        # Skip if search term provided and not in path
        if search_term and search_term.lower() not in path.lower():
            continue

        # Check for articulation root
        if prim.IsA(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)

        # Check for joints
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(("RevoluteJoint", path))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joints.append(("PrismaticJoint", path))
        elif "Joint" in path:
            joints.append(("UnknownJoint", path))

    print("Articulation Roots:")
    for root in articulation_roots:
        print(f"  - {root}")

    print("\nJoints:")
    for joint_type, path in joints:
        print(f"  - {joint_type}: {path}")

    print()

if __name__ == "__main__":
    # Inspect the Interactive_kitchen.usd for largecabinet
    kitchen_usd = "/home/chuanruo/IsaacLab/assets/ArtVIP/Interactive_scene/kitchen/Interactive_kitchen.usd"
    print("Searching for 'largecabinet' in Interactive_kitchen.usd...")
    inspect_usd(kitchen_usd, "largecabinet")

    # Also inspect the standalone largecabinet model
    cabinet_usd = "/home/chuanruo/IsaacLab/assets/ArtVIP/Interactive_scene/kitchen/largecabinet/model_largecabinet.usd"
    print("\nInspecting standalone largecabinet model...")
    inspect_usd(cabinet_usd)
