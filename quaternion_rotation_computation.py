#!/usr/bin/env python3
"""
Quaternion Rotation Computation Script

This script demonstrates how to apply rotations to quaternions using quaternion multiplication.
It shows the step-by-step computation for applying rotations around different axes.
"""

import numpy as np
import math


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.
    
    Args:
        q1: First quaternion as (w, x, y, z)
        q2: Second quaternion as (w, x, y, z)
    
    Returns:
        Result quaternion as (w, x, y, z)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return (w, x, y, z)


def rotation_quaternion(angle_degrees, axis):
    """
    Create a rotation quaternion for a given angle and axis.
    
    Args:
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z')
    
    Returns:
        Rotation quaternion as (w, x, y, z)
    """
    angle_rad = math.radians(angle_degrees)
    cos_half = math.cos(angle_rad / 2)
    sin_half = math.sin(angle_rad / 2)
    
    if axis == 'x':
        return (cos_half, sin_half, 0, 0)
    elif axis == 'y':
        return (cos_half, 0, sin_half, 0)
    elif axis == 'z':
        return (cos_half, 0, 0, sin_half)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


def apply_rotation_to_quaternion(original_quat, angle_degrees, axis):
    """
    Apply a rotation to an existing quaternion.
    
    Args:
        original_quat: Original quaternion as (w, x, y, z)
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z')
    
    Returns:
        New quaternion after rotation as (w, x, y, z)
    """
    rotation_quat = rotation_quaternion(angle_degrees, axis)
    return quaternion_multiply(original_quat, rotation_quat)


def print_quaternion(q, name="Quaternion"):
    """Print quaternion in a readable format."""
    w, x, y, z = q
    print(f"{name}: ({w:.5f}, {x:.5f}, {y:.5f}, {z:.5f})")


def main():
    """Main function demonstrating quaternion rotations."""
    print("Quaternion Rotation Computation")
    print("=" * 50)
    
    # Original quaternion from the camera configuration
    original_quat = (-0.38512, -0.59303, -0.59303, -0.38512)
    print_quaternion(original_quat, "Original quaternion")
    print()
    
    rotation_y_neg50 = rotation_quaternion(-45, 'x')
    final_quat = quaternion_multiply(original_quat, rotation_y_neg50)
    print_quaternion(final_quat, "Final quaternion")


if __name__ == "__main__":
    main()
