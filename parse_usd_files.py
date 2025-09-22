#!/usr/bin/env python3
"""
Simple USD file parser to extract properties without requiring Isaac Sim.
This script parses USD files as text to find embedded properties.
"""

import os
import re
from pathlib import Path

def find_isaac_nucleus_dir():
    """Try to find ISAAC_NUCLEUS_DIR in common locations."""
    possible_paths = [
        "/isaac-sim/isaac-sim-2023.1.1/isaac_sim-2023.1.1/kit/isaac_sim/nucleus",
        "/isaac-sim/isaac-sim-2023.1.1/isaac_sim-2023.1.1/kit/isaac_sim/nucleus",
        "/isaac-sim/nucleus",
        "/opt/nvidia/isaac-sim/nucleus",
        "/home/chuanruo/isaac-sim/nucleus",
        "/home/chuanruo/IsaacSim/nucleus"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find from environment
    isaac_nucleus = os.environ.get('ISAAC_NUCLEUS_DIR')
    if isaac_nucleus and os.path.exists(isaac_nucleus):
        return isaac_nucleus
    
    return None

def parse_usd_file(usd_path):
    """Parse USD file to extract physics properties."""
    if not os.path.exists(usd_path):
        return None
    
    try:
        with open(usd_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        properties = {
            'mass': None,
            'density': None,
            'static_friction': None,
            'dynamic_friction': None,
            'restitution': None,
            'file_size': os.path.getsize(usd_path)
        }
        
        # Look for mass properties
        mass_match = re.search(r'mass.*?(\d+\.?\d*)', content, re.IGNORECASE)
        if mass_match:
            properties['mass'] = float(mass_match.group(1))
        
        # Look for density properties
        density_match = re.search(r'density.*?(\d+\.?\d*)', content, re.IGNORECASE)
        if density_match:
            properties['density'] = float(density_match.group(1))
        
        # Look for friction properties
        static_friction_match = re.search(r'staticFriction.*?(\d+\.?\d*)', content, re.IGNORECASE)
        if static_friction_match:
            properties['static_friction'] = float(static_friction_match.group(1))
        
        dynamic_friction_match = re.search(r'dynamicFriction.*?(\d+\.?\d*)', content, re.IGNORECASE)
        if dynamic_friction_match:
            properties['dynamic_friction'] = float(dynamic_friction_match.group(1))
        
        # Look for restitution
        restitution_match = re.search(r'restitution.*?(\d+\.?\d*)', content, re.IGNORECASE)
        if restitution_match:
            properties['restitution'] = float(restitution_match.group(1))
        
        # Check for physics material references
        has_physics_material = 'PhysicsMaterial' in content
        has_rigid_body_api = 'RigidBodyAPI' in content
        has_mass_api = 'MassAPI' in content
        
        properties['has_physics_material'] = has_physics_material
        properties['has_rigid_body_api'] = has_rigid_body_api
        properties['has_mass_api'] = has_mass_api
        
        return properties
        
    except Exception as e:
        print(f"Error parsing {usd_path}: {e}")
        return None

def main():
    """Main function to parse cube USD files."""
    print("🚀 USD File Properties Parser")
    print("=" * 50)
    
    # Try to find ISAAC_NUCLEUS_DIR
    nucleus_dir = find_isaac_nucleus_dir()
    if not nucleus_dir:
        print("❌ Could not find ISAAC_NUCLEUS_DIR")
        print("💡 Please run this in Isaac Sim environment or set ISAAC_NUCLEUS_DIR")
        return
    
    print(f"📁 Found nucleus directory: {nucleus_dir}")
    
    # Define cube files to check
    cube_files = [
        ("blue_block.usd", "Blue Cube (Cube 1)"),
        ("red_block.usd", "Red Cube (Cube 2)"), 
        ("green_block.usd", "Green Cube (Cube 3)")
    ]
    
    results = {}
    
    for usd_file, cube_name in cube_files:
        usd_path = os.path.join(nucleus_dir, "Props", "Blocks", usd_file)
        print(f"\n🔍 Parsing {cube_name}")
        print("-" * 40)
        print(f"📁 Path: {usd_path}")
        
        properties = parse_usd_file(usd_path)
        if not properties:
            print("❌ Failed to parse USD file")
            continue
        
        print(f"📊 File size: {properties['file_size']} bytes")
        
        # Display found properties
        if properties['mass'] is not None:
            print(f"⚖️  Mass: {properties['mass']} kg")
        else:
            print("⚖️  Mass: Not found (will use PhysX default ~1.0 kg)")
        
        if properties['density'] is not None:
            print(f"⚖️  Density: {properties['density']} kg/m³")
        else:
            print("⚖️  Density: Not found (will use PhysX default)")
        
        if properties['static_friction'] is not None:
            print(f"🔧 Static Friction: {properties['static_friction']}")
        else:
            print("🔧 Static Friction: Not found (will use PhysX default 0.5)")
        
        if properties['dynamic_friction'] is not None:
            print(f"🔧 Dynamic Friction: {properties['dynamic_friction']}")
        else:
            print("🔧 Dynamic Friction: Not found (will use PhysX default 0.5)")
        
        if properties['restitution'] is not None:
            print(f"🔧 Restitution: {properties['restitution']}")
        else:
            print("🔧 Restitution: Not found (will use PhysX default 0.0)")
        
        # Check for API presence
        print(f"🔧 Has PhysicsMaterial: {properties['has_physics_material']}")
        print(f"🤖 Has RigidBodyAPI: {properties['has_rigid_body_api']}")
        print(f"⚖️  Has MassAPI: {properties['has_mass_api']}")
        
        results[cube_name] = properties
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY - Original vs Your Modified Values:")
    print("=" * 50)
    
    # Get first cube's properties as reference
    first_cube = list(results.values())[0] if results else {}
    
    print("| Property           | Original (USD Default) | Your Modified | Change        |")
    print("|--------------------|------------------------|---------------|---------------|")
    
    # Mass comparison
    original_mass = first_cube.get('mass', '~1.0 kg (PhysX default)')
    print(f"| Mass               | {original_mass:<22} | 0.1 kg        | 10x lighter   |")
    
    # Friction comparison
    original_static = first_cube.get('static_friction', 0.5)
    original_dynamic = first_cube.get('dynamic_friction', 0.5)
    original_restitution = first_cube.get('restitution', 0.0)
    
    print(f"| Static Friction    | {original_static:<22} | 0.7           | 40% higher    |")
    print(f"| Dynamic Friction   | {original_dynamic:<22} | 0.6           | 20% higher    |")
    print(f"| Restitution        | {original_restitution:<22} | 0.1           | Slight bounce |")
    
    print("=" * 50)
    print("✅ Your modifications make the cubes:")
    print("   🔸 Much lighter (easier to manipulate)")
    print("   🔸 More grippy (better for stacking)")
    print("   🔸 Slightly bouncy (more realistic physics)")
    
    if results:
        print(f"\n🎉 Successfully parsed {len(results)} cube files!")
    else:
        print("\n❌ No cube files were successfully parsed.")

if __name__ == "__main__":
    main()
