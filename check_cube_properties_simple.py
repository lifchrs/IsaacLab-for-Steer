#!/usr/bin/env python3
"""
Simple script to check USD file properties without requiring Isaac Sim.
This version uses basic USD parsing to extract property information.
"""

import os
import sys
from pathlib import Path

def check_usd_file_simple(usd_path: str, cube_name: str):
    """Simple USD file inspection using text parsing."""
    print(f"\n🔍 Checking {cube_name}: {usd_path}")
    print("=" * 50)
    
    if not os.path.exists(usd_path):
        print(f"❌ USD file not found: {usd_path}")
        return
    
    try:
        with open(usd_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 File size: {os.path.getsize(usd_path)} bytes")
        
        # Look for mass-related properties
        mass_found = False
        if 'mass' in content.lower():
            print("⚖️  Mass properties found in USD file")
            mass_found = True
        
        # Look for friction-related properties  
        friction_found = False
        if 'friction' in content.lower():
            print("🔧 Friction properties found in USD file")
            friction_found = True
            
        # Look for physics material
        if 'PhysicsMaterial' in content:
            print("🔧 PhysicsMaterial found in USD file")
            friction_found = True
            
        # Look for rigid body API
        if 'RigidBodyAPI' in content:
            print("🤖 RigidBodyAPI found in USD file")
        
        # Look for collision shapes
        if 'CollisionAPI' in content:
            print("💥 CollisionAPI found in USD file")
            
        if not mass_found and not friction_found:
            print("⚠️  No explicit mass or friction properties found")
            print("   → Will use PhysX default values")
            
    except Exception as e:
        print(f"❌ Error reading USD file: {e}")

def main():
    """Main function."""
    print("🚀 Simple USD Properties Checker")
    print("=" * 50)
    
    # Try to find ISAAC_NUCLEUS_DIR
    isaac_nucleus_dir = os.environ.get('ISAAC_NUCLEUS_DIR')
    if not isaac_nucleus_dir:
        print("❌ ISAAC_NUCLEUS_DIR environment variable not set")
        print("💡 Try running this in Isaac Sim environment")
        return
    
    print(f"📁 ISAAC_NUCLEUS_DIR: {isaac_nucleus_dir}")
    
    # Check cube files
    cube_files = [
        ("blue_block.usd", "Blue Cube"),
        ("red_block.usd", "Red Cube"), 
        ("green_block.usd", "Green Cube")
    ]
    
    for usd_file, cube_name in cube_files:
        usd_path = os.path.join(isaac_nucleus_dir, "Props", "Blocks", usd_file)
        check_usd_file_simple(usd_path, cube_name)
    
    print("\n" + "=" * 50)
    print("📊 PhysX Default Values (when not specified):")
    print("   - Static Friction: 0.5")
    print("   - Dynamic Friction: 0.5")
    print("   - Restitution: 0.0")
    print("   - Mass: Calculated from default density")

if __name__ == "__main__":
    main()
