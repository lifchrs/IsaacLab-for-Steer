#!/usr/bin/env python3

"""
Example script to test custom environments with different lighting and mass properties.

This script demonstrates how to:
1. Use the custom environment with modified configurations
2. Compare different environment variants
3. Test the environments with basic functionality

Usage:
    python scripts/imitation_learning/isaaclab_mimic/test_custom_environments.py
"""

import argparse
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv


def test_environment(env_name: str, num_envs: int = 1, num_steps: int = 100):
    """Test a specific environment with basic functionality."""
    print(f"\n=== Testing Environment: {env_name} ===")
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_name, render_mode="human")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Environment reset successful. Observation keys: {list(obs.keys())}")
    
    # Run simulation for a few steps
    for step in range(num_steps):
        # Random actions
        actions = torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 20 == 0:
            print(f"Step {step}: Rewards = {rewards.mean().item():.4f}")
    
    # Close environment
    env.close()
    print(f"Environment {env_name} test completed successfully!")


def main():
    """Main function to test custom environments."""
    parser = argparse.ArgumentParser(description="Test custom Isaac Lab Mimic environments")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--env_name", type=str, default=None, help="Specific environment to test")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    # Configure simulation
    sim_utils.launch_app(args.headless)
    
    # List of environments to test
    environments = [
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",  # Original
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Custom-Mimic-v0",  # Custom
    ]
    
    if args.env_name:
        environments = [args.env_name]
    
    # Test each environment
    for env_name in environments:
        try:
            test_environment(env_name, args.num_envs, args.num_steps)
        except Exception as e:
            print(f"Error testing {env_name}: {e}")
            continue
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()



