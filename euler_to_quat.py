"""Convert Euler angles (radians) to quaternion using numpy only."""

import numpy as np


def euler_xyz_to_quat(euler_xyz):
    """Euler angles (x, y, z) in radians to quaternion (x, y, z, w)."""
    x, y, z = euler_xyz[0], euler_xyz[1], euler_xyz[2]
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)

    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    qw = cx * cy * cz + sx * sy * sz
    return np.array([qx, qy, qz, qw])


def euler_zyx_to_quat(euler_zyx):
    """Euler angles (z, y, x) / yaw-pitch-roll in radians to quaternion (x, y, z, w)."""
    z, y, x = euler_zyx[0], euler_zyx[1], euler_zyx[2]
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)

    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    qw = cx * cy * cz - sx * sy * sz
    return np.array([qx, qy, qz, qw])


# Your Euler angles in radians
euler_rad = np.array([1.9118962653748173, 0.03641853569757658, 2.3314608097198026])

print("Euler (rad):", euler_rad)
print()

# Assuming order is (x, y, z) - intrinsic rotations about X, then Y, then Z
q_xyz = euler_xyz_to_quat(euler_rad)
print("Convention XYZ (intrinsic):")
print("  quaternion (x, y, z, w) =", q_xyz)
print("  (w, x, y, z) =", (q_xyz[3], q_xyz[0], q_xyz[1], q_xyz[2]))
print()

# ZYX (yaw, pitch, roll) - common in robotics
q_zyx = euler_zyx_to_quat(euler_rad)
print("Convention ZYX (yaw, pitch, roll):")
print("  quaternion (x, y, z, w) =", q_zyx)
print("  (w, x, y, z) =", (q_zyx[3], q_zyx[0], q_zyx[1], q_zyx[2]))
