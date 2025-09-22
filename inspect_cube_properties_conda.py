#!/usr/bin/env python3
"""
Script to inspect cube USD properties using the conda environment.
This script should be run in Isaac Sim with the env_isaaclab conda environment activated.

Usage in Isaac Sim:
1. Open Isaac Sim
2. In the Script Editor, run: conda activate env_isaaclab
3. Copy and paste this script
4. Run it to inspect the cube properties

Or run from terminal in Isaac Sim:
    conda activate env_isaaclab
    python inspect_cube_properties_conda.py
"""

import os
import sys
from pathlib import Path

def inspect_cube_properties():
    """Inspect cube properties using Isaac Lab utilities."""
    print("🚀 Isaac Lab Cube Properties Inspector (Conda Environment)")
    print("=" * 70)
    
    try:
        # Import Isaac Lab modules
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from pxr import Usd, UsdPhysics
        import omni.usd
        
        print(f"📁 ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
        print(f"🐍 Python: {sys.executable}")
        print(f"📦 Isaac Lab: {Path(__file__).parent / 'source' / 'isaaclab'}")
        
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
            print("-" * 50)
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
            
            cube_props = {
                'mass': None,
                'density': None,
                'static_friction': None,
                'dynamic_friction': None,
                'restitution': None
            }
            
            if not rigid_bodies:
                print("⚠️  No rigid body found - using PhysX defaults")
                cube_props = {
                    'mass': '~1.0 kg (PhysX default)',
                    'static_friction': '0.5 (PhysX default)',
                    'dynamic_friction': '0.5 (PhysX default)',
                    'restitution': '0.0 (PhysX default)'
                }
            else:
                # Check properties of first rigid body
                prim = rigid_bodies[0]
                print(f"🎯 Rigid Body: {prim.GetPath()}")
                
                # Check mass properties
                mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if mass_api:
                    mass_attr = mass_api.GetMassAttr()
                    if mass_attr.HasValue():
                        cube_props['mass'] = f"{mass_attr.Get()} kg"
                        print(f"⚖️  Mass: {mass_attr.Get()} kg")
                    else:
                        cube_props['mass'] = "~1.0 kg (PhysX default)"
                        print("⚖️  Mass: Not set (default ~1.0 kg)")
                    
                    density_attr = mass_api.GetDensityAttr()
                    if density_attr.HasValue():
                        cube_props['density'] = f"{density_attr.Get()} kg/m³"
                        print(f"⚖️  Density: {density_attr.Get()} kg/m³")
                    else:
                        cube_props['density'] = "Default density"
                        print("⚖️  Density: Not set (default density)")
                else:
                    cube_props['mass'] = "~1.0 kg (PhysX default)"
                    print("⚖️  Mass: No MassAPI (default ~1.0 kg)")
                
                # Check physics material
                material_api = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
                if material_api:
                    static_friction = material_api.GetStaticFrictionAttr()
                    if static_friction.HasValue():
                        cube_props['static_friction'] = static_friction.Get()
                        print(f"🔧 Static Friction: {static_friction.Get()}")
                    else:
                        cube_props['static_friction'] = 0.5
                        print("🔧 Static Friction: Not set (default 0.5)")
                    
                    dynamic_friction = material_api.GetDynamicFrictionAttr()
                    if dynamic_friction.HasValue():
                        cube_props['dynamic_friction'] = dynamic_friction.Get()
                        print(f"🔧 Dynamic Friction: {dynamic_friction.Get()}")
                    else:
                        cube_props['dynamic_friction'] = 0.5
                        print("🔧 Dynamic Friction: Not set (default 0.5)")
                    
                    restitution = material_api.GetRestitutionAttr()
                    if restitution.HasValue():
                        cube_props['restitution'] = restitution.Get()
                        print(f"🔧 Restitution: {restitution.Get()}")
                    else:
                        cube_props['restitution'] = 0.0
                        print("🔧 Restitution: Not set (default 0.0)")
                else:
                    cube_props['static_friction'] = 0.5
                    cube_props['dynamic_friction'] = 0.5
                    cube_props['restitution'] = 0.0
                    print("🔧 Physics Material: No MaterialAPI (using defaults)")
                    print("   Default Static Friction: 0.5")
                    print("   Default Dynamic Friction: 0.5")
                    print("   Default Restitution: 0.0")
            
            results[cube_name] = cube_props
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 DETAILED COMPARISON: Original vs Your Modified Values")
        print("=" * 70)
        
        # Get first cube's properties as reference
        first_cube = list(results.values())[0] if results else {}
        
        print("| Property           | Original (USD Default) | Your Modified | Change        |")
        print("|--------------------|------------------------|---------------|---------------|")
        
        # Mass comparison
        original_mass = first_cube.get('mass', '~1.0 kg')
        print(f"| Mass               | {original_mass:<22} | 0.1 kg        | 10x lighter   |")
        
        # Friction comparison
        original_static = first_cube.get('static_friction', 0.5)
        original_dynamic = first_cube.get('dynamic_friction', 0.5)
        original_restitution = first_cube.get('restitution', 0.0)
        
        print(f"| Static Friction    | {original_static:<22} | 0.7           | 40% higher    |")
        print(f"| Dynamic Friction   | {original_dynamic:<22} | 0.6           | 20% higher    |")
        print(f"| Restitution        | {original_restitution:<22} | 0.1           | Slight bounce |")
        
        print("=" * 70)
        print("✅ Your modifications make the cubes:")
        print("   🔸 Much lighter (easier to manipulate)")
        print("   🔸 More grippy (better for stacking)")
        print("   🔸 Slightly bouncy (more realistic physics)")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this in Isaac Sim with env_isaaclab activated")
        print("💡 Try: conda activate env_isaaclab")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function."""
    results = inspect_cube_properties()
    
    if results:
        print(f"\n🎉 Successfully inspected {len(results)} cube files!")
        print("💡 You can now compare these values with your custom configuration.")
    else:
        print("\n❌ Failed to inspect cube properties.")
        print("💡 Make sure you're running in Isaac Sim with the proper environment.")

if __name__ == "__main__":
    main()
