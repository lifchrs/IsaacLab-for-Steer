#!/usr/bin/env python3

"""
Simple script to print Droid robot link names.
This script creates the environment and prints the robot links.
"""

import gymnasium as gym
from isaaclab.utils import parse_env_cfg

def main():
    """Print Droid robot link names."""
    try:
        # Create environment configuration
        env_cfg = parse_env_cfg("Isaac-Stack-Droid-v0", device="cpu", num_envs=1)
        
        # Create environment
        env = gym.make("Isaac-Stack-Droid-v0", cfg=env_cfg)
        
        # Reset environment to initialize robot
        env.reset()
        
        # Get the robot articulation
        robot = env.unwrapped.scene["robot"]
        
        # Print robot link names
        print("=" * 60)
        print("DROID ROBOT LINK NAMES")
        print("=" * 60)
        print(f"Number of links: {robot.num_bodies}")
        print(f"Link names: {robot.body_names}")
        print()
        
        # Print with indices
        print("Links with indices:")
        for i, link_name in enumerate(robot.body_names):
            print(f"  {i:2d}: {link_name}")
        
        # Print joint names as well
        print(f"\nNumber of joints: {robot.num_joints}")
        print(f"Joint names: {robot.joint_names}")
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the correct environment name and Isaac Lab is properly set up.")

if __name__ == "__main__":
    main()
