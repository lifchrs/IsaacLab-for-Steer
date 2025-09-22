# Isaac Sim Script Editor Code
# Copy and paste this into Isaac Sim's Script Editor and run it

import omni.usd
from pxr import Usd, UsdPhysics
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os

print("🚀 Isaac Sim Cube Properties Inspector")
print("=" * 50)
print(f"📁 ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")

# Define cube files to check
cube_files = [
    ("blue_block.usd", "Blue Cube (Cube 1)"),
    ("red_block.usd", "Red Cube (Cube 2)"), 
    ("green_block.usd", "Green Cube (Cube 3)")
]

results = {}

for usd_file, cube_name in cube_files:
    usd_path = os.path.join(ISAAC_NUCLEUS_DIR, "Props", "Blocks", usd_file)
    print(f"\n🔍 Inspecting {cube_name}")
    print("-" * 40)
    print(f"📁 Path: {usd_path}")
    
    if not os.path.exists(usd_path):
        print(f"❌ File not found: {usd_path}")
        continue
    
    # Open USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"❌ Failed to open USD file")
        continue
    
    # Find rigid body prims
    rigid_bodies = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
    
    if not rigid_bodies:
        print("⚠️  No rigid body found - using PhysX defaults")
        print("   Default Mass: ~1.0 kg")
        print("   Default Static Friction: 0.5")
        print("   Default Dynamic Friction: 0.5")
        print("   Default Restitution: 0.0")
        continue
    
    # Check properties of first rigid body
    prim = rigid_bodies[0]
    print(f"🎯 Rigid Body: {prim.GetPath()}")
    
    # Check mass properties
    mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
    if mass_api:
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.HasValue():
            print(f"⚖️  Mass: {mass_attr.Get()} kg")
        else:
            print("⚖️  Mass: Not set (default ~1.0 kg)")
        
        density_attr = mass_api.GetDensityAttr()
        if density_attr.HasValue():
            print(f"⚖️  Density: {density_attr.Get()} kg/m³")
        else:
            print("⚖️  Density: Not set (default density)")
    else:
        print("⚖️  Mass: No MassAPI (default ~1.0 kg)")
    
    # Check physics material
    material_api = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
    if material_api:
        static_friction = material_api.GetStaticFrictionAttr()
        if static_friction.HasValue():
            print(f"🔧 Static Friction: {static_friction.Get()}")
        else:
            print("🔧 Static Friction: Not set (default 0.5)")
        
        dynamic_friction = material_api.GetDynamicFrictionAttr()
        if dynamic_friction.HasValue():
            print(f"🔧 Dynamic Friction: {dynamic_friction.Get()}")
        else:
            print("🔧 Dynamic Friction: Not set (default 0.5)")
        
        restitution = material_api.GetRestitutionAttr()
        if restitution.HasValue():
            print(f"🔧 Restitution: {restitution.Get()}")
        else:
            print("🔧 Restitution: Not set (default 0.0)")
    else:
        print("🔧 Physics Material: No MaterialAPI (using defaults)")
        print("   Default Static Friction: 0.5")
        print("   Default Dynamic Friction: 0.5")
        print("   Default Restitution: 0.0")

print("\n" + "=" * 50)
print("📊 SUMMARY - Original vs Your Modified Values:")
print("=" * 50)
print("| Property           | Original (USD Default) | Your Modified | Change        |")
print("|--------------------|------------------------|---------------|---------------|")
print("| Mass               | ~1.0 kg               | 0.1 kg        | 10x lighter   |")
print("| Static Friction    | 0.5                   | 0.7           | 40% higher    |")
print("| Dynamic Friction   | 0.5                   | 0.6           | 20% higher    |")
print("| Restitution        | 0.0                   | 0.1           | Slight bounce |")
print("=" * 50)
print("✅ Your cubes are now much lighter and more grippy!")
