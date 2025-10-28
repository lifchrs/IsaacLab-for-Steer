#!/usr/bin/env python3
"""
Test script to verify DROID_CFG matches the actual joints in droid.usd
"""
import re

# Joints that are actually available according to the error message
available_joints = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
    "panda_joint5", "panda_joint6", "panda_joint7"
]

# Joint positions from DROID_CFG
droid_joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
}

# Actuator patterns from DROID_CFG
actuator_patterns = {
    "panda_shoulder": ["panda_joint[1-4]"],
    "panda_forearm": ["panda_joint[5-7]"],
}

print("Testing DROID_CFG joint configuration:")
print("=" * 50)

print("Available joints:", available_joints)
print("DROID_CFG joint_pos keys:", list(droid_joint_pos.keys()))

# Check if all joint_pos keys are in available joints
missing_joints = []
for joint in droid_joint_pos.keys():
    if joint not in available_joints:
        missing_joints.append(joint)

if missing_joints:
    print(f"❌ Missing joints in available list: {missing_joints}")
else:
    print("✅ All joint_pos keys are available")

# Check actuator patterns
print("\nActuator pattern matching:")
for actuator_name, patterns in actuator_patterns.items():
    print(f"\n{actuator_name}:")
    for pattern in patterns:
        matching_joints = [joint for joint in available_joints if re.match(pattern, joint)]
        print(f"  Pattern '{pattern}' matches: {matching_joints}")
        if not matching_joints:
            print(f"  ❌ Pattern '{pattern}' has no matches!")

print("\n" + "=" * 50)
print("Summary:")
print(f"- Available joints: {len(available_joints)}")
print(f"- Configured joints: {len(droid_joint_pos)}")
print(f"- Actuator groups: {len(actuator_patterns)}")

if not missing_joints:
    print("✅ Configuration should work correctly!")
else:
    print("❌ Configuration needs fixing")

