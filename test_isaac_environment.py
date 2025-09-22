#!/usr/bin/env python3
"""
Test script to check if Isaac Lab environment is properly set up.
Run this to verify you can access Isaac Lab modules and USD files.
"""

import os
import sys

def test_isaac_environment():
    """Test Isaac Lab environment setup."""
    print("🧪 Testing Isaac Lab Environment")
    print("=" * 40)
    
    # Test 1: Check Python path
    print("1️⃣  Python Path:")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version}")
    
    # Test 2: Check Isaac Lab import
    print("\n2️⃣  Isaac Lab Import:")
    try:
        import isaaclab
        print("   ✅ isaaclab imported successfully")
        print(f"   📁 Isaac Lab path: {isaaclab.__file__}")
    except ImportError as e:
        print(f"   ❌ Failed to import isaaclab: {e}")
        return False
    
    # Test 3: Check ISAAC_NUCLEUS_DIR
    print("\n3️⃣  ISAAC_NUCLEUS_DIR:")
    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        print(f"   ✅ ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
        
        if os.path.exists(ISAAC_NUCLEUS_DIR):
            print("   ✅ Directory exists")
        else:
            print("   ❌ Directory does not exist")
            return False
    except Exception as e:
        print(f"   ❌ Error getting ISAAC_NUCLEUS_DIR: {e}")
        return False
    
    # Test 4: Check USD files
    print("\n4️⃣  USD Files:")
    cube_files = ["blue_block.usd", "red_block.usd", "green_block.usd"]
    usd_dir = os.path.join(ISAAC_NUCLEUS_DIR, "Props", "Blocks")
    
    for cube_file in cube_files:
        usd_path = os.path.join(usd_dir, cube_file)
        if os.path.exists(usd_path):
            size = os.path.getsize(usd_path)
            print(f"   ✅ {cube_file}: {size} bytes")
        else:
            print(f"   ❌ {cube_file}: Not found")
    
    # Test 5: Check USD/PhysX imports
    print("\n5️⃣  USD/PhysX Imports:")
    try:
        from pxr import Usd, UsdPhysics
        print("   ✅ USD/PhysX modules imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import USD/PhysX: {e}")
        print("   💡 Make sure you're running in Isaac Sim environment")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Environment test completed!")
    print("💡 You can now run: python inspect_cube_defaults.py")
    return True

if __name__ == "__main__":
    test_isaac_environment()
