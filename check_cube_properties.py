#!/usr/bin/env python3
"""
Script to inspect the default properties of cube USD files used in the stack environment.
This will help us understand what the original default values are.
"""

import os
import sys
from pathlib import Path

# Add Isaac Lab to path
isaac_lab_path = Path(__file__).parent / "source" / "isaaclab"
sys.path.insert(0, str(isaac_lab_path))

try:
    import omni.usd
    from pxr import Usd, UsdPhysics, Sdf
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script in the Isaac Sim environment")
    sys.exit(1)


def inspect_usd_properties(usd_path: str, cube_name: str):
    """Inspect the properties of a USD file."""
    print(f"\n🔍 Inspecting {cube_name}: {usd_path}")
    print("=" * 60)
    
    if not os.path.exists(usd_path):
        print(f"❌ USD file not found: {usd_path}")
        return
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"❌ Failed to open USD file: {usd_path}")
        return
    
    print(f"📁 USD File: {usd_path}")
    print(f"📋 Stage Root: {stage.GetRootLayer().identifier}")
    
    # Find all rigid body prims
    rigid_bodies = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
    
    if not rigid_bodies:
        print("⚠️  No rigid body prims found in USD file")
        return
    
    print(f"🎯 Found {len(rigid_bodies)} rigid body prim(s)")
    
    for i, prim in enumerate(rigid_bodies):
        print(f"\n📦 Rigid Body {i+1}: {prim.GetPath()}")
        print("-" * 40)
        
        # Check for mass properties
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if mass_api:
            print("⚖️  Mass Properties:")
            mass_attr = mass_api.GetMassAttr()
            if mass_attr.HasValue():
                print(f"   Mass: {mass_attr.Get()} kg")
            else:
                print("   Mass: Not set (will use default)")
            
            density_attr = mass_api.GetDensityAttr()
            if density_attr.HasValue():
                print(f"   Density: {density_attr.Get()} kg/m³")
            else:
                print("   Density: Not set (will use default)")
        else:
            print("⚖️  Mass Properties: No MassAPI found (will use defaults)")
        
        # Check for physics material
        material_api = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
        if material_api:
            print("🔧 Physics Material:")
            static_friction = material_api.GetStaticFrictionAttr()
            if static_friction.HasValue():
                print(f"   Static Friction: {static_friction.Get()}")
            else:
                print("   Static Friction: Not set (will use default 0.5)")
            
            dynamic_friction = material_api.GetDynamicFrictionAttr()
            if dynamic_friction.HasValue():
                print(f"   Dynamic Friction: {dynamic_friction.Get()}")
            else:
                print("   Dynamic Friction: Not set (will use default 0.5)")
            
            restitution = material_api.GetRestitutionAttr()
            if restitution.HasValue():
                print(f"   Restitution: {restitution.Get()}")
            else:
                print("   Restitution: Not set (will use default 0.0)")
        else:
            print("🔧 Physics Material: No MaterialAPI found (will use defaults)")
        
        # Check for rigid body properties
        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        if rigid_body_api:
            print("🤖 Rigid Body Properties:")
            kinematic = rigid_body_api.GetKinematicEnabledAttr()
            if kinematic.HasValue():
                print(f"   Kinematic: {kinematic.Get()}")
            else:
                print("   Kinematic: Not set (will use default False)")
            
            disable_gravity = rigid_body_api.GetDisableGravityAttr()
            if disable_gravity.HasValue():
                print(f"   Disable Gravity: {disable_gravity.Get()}")
            else:
                print("   Disable Gravity: Not set (will use default False)")
        
        # Check for collision properties
        collision_apis = UsdPhysics.CollisionAPI.GetAppliedSchemas(prim)
        if collision_apis:
            print(f"💥 Collision Properties: {len(collision_apis)} collision shape(s) found")
            for j, collision_api in enumerate(collision_apis):
                collision_prim = stage.GetPrimAtPath(collision_api)
                if collision_prim:
                    print(f"   Collision {j+1}: {collision_prim.GetPath()}")
        else:
            print("💥 Collision Properties: No collision shapes found")


def main():
    """Main function to inspect all cube USD files."""
    print("🚀 Isaac Lab Cube Properties Inspector")
    print("=" * 60)
    
    # Check if ISAAC_NUCLEUS_DIR is available
    if not ISAAC_NUCLEUS_DIR or not os.path.exists(ISAAC_NUCLEUS_DIR):
        print(f"❌ ISAAC_NUCLEUS_DIR not found or invalid: {ISAAC_NUCLEUS_DIR}")
        print("Make sure Isaac Sim is properly installed and ISAAC_NUCLEUS_DIR is set")
        return
    
    print(f"📁 ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
    
    # Define the cube USD files to inspect
    cube_files = [
        ("blue_block.usd", "Blue Cube"),
        ("red_block.usd", "Red Cube"), 
        ("green_block.usd", "Green Cube"),
        ("DexCube/dex_cube_instanceable.usd", "DexCube (for comparison)")
    ]
    
    # Inspect each cube file
    for usd_file, cube_name in cube_files:
        usd_path = os.path.join(ISAAC_NUCLEUS_DIR, "Props", "Blocks", usd_file)
        inspect_usd_properties(usd_path, cube_name)
    
    print("\n" + "=" * 60)
    print("📊 Summary of Default Values:")
    print("=" * 60)
    print("🔸 If no mass is specified in USD: PhysX will use default density")
    print("🔸 If no friction is specified: PhysX defaults are:")
    print("   - Static Friction: 0.5")
    print("   - Dynamic Friction: 0.5") 
    print("   - Restitution: 0.0")
    print("🔸 If no rigid body properties: PhysX defaults are:")
    print("   - Kinematic: False")
    print("   - Disable Gravity: False")
    print("\n✅ Inspection complete!")


if __name__ == "__main__":
    main()
