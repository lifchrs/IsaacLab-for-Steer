"""Analytical IK solver for the Tianji CCS 7-DOF arm.

Wraps the Tianji SDK's ``libKine.so`` closed-form solver (up to 4 solutions).
Units: this module accepts meters/radians/quaternions (IsaacLab conventions)
and handles conversion to mm/degrees internally.

Usage::

    ik = TianjiAnalyticalIK()
    joints_rad = ik.solve_ik(
        arm="left",
        target_pos=torch.tensor([0.3, 0.1, 0.5]),   # meters, in arm base frame
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # wxyz
        ref_joints=torch.zeros(7),  # radians
    )
"""

import ctypes
import math
import os
from ctypes import Structure, c_bool, c_double, c_long
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
_SDK_DIR = Path(__file__).resolve().parents[4] / "assets" / "tianji" / "sdk"
_LIB_PATH = str(_SDK_DIR / "libKine.so")
_CFG_PATH = str(_SDK_DIR / "ccs_m3.MvKDCfg")

# ── ctypes structures (mirrored from Tianji SDK fx_kine.py) ────────────────
_FX_INT32L = c_long
_FX_DOUBLE = c_double
_FX_BOOL = c_bool


class _Vect7(Structure):
    _fields_ = [("data", _FX_DOUBLE * 7)]

    def to_list(self):
        return [self.data[i] for i in range(7)]


class _Matrix4(Structure):
    _fields_ = [("data", _FX_DOUBLE * 16)]


class _Matrix8(Structure):
    _fields_ = [("data", _FX_DOUBLE * 64)]


class _FX_InvKineSolvePara(Structure):
    _fields_ = [
        ("m_Input_IK_TargetTCP", _Matrix4),
        ("m_Input_IK_RefJoint", _Vect7),
        ("m_Input_IK_ZSPType", _FX_INT32L),
        ("m_Input_IK_ZSPPara", _FX_DOUBLE * 6),
        ("m_Input_ZSP_Angle", _FX_DOUBLE),
        ("m_DGR1", _FX_DOUBLE),
        ("m_DGR2", _FX_DOUBLE),
        ("m_DGR3", _FX_DOUBLE),
        ("m_Output_RetJoint", _Vect7),
        ("m_OutPut_AllJoint", _Matrix8),
        ("m_OutPut_Result_Num", _FX_INT32L),
        ("m_Output_IsOutRange", _FX_BOOL),
        ("m_Output_IsDeg", _FX_BOOL * 7),
        ("m_Output_JntExdTags", _FX_BOOL * 7),
        ("m_Output_JntExdABS", _FX_DOUBLE),
        ("m_Output_IsJntExd", _FX_BOOL),
        ("m_Output_RunLmtP", _Vect7),
        ("m_Output_RunLmtN", _Vect7),
    ]


# ── Helper: quaternion (wxyz) → 3x3 rotation matrix ───────────────────────
def _quat_to_rotmat(quat_wxyz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix (numpy)."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


class TianjiAnalyticalIK:
    """Analytical IK for the Tianji CCS 7-DOF arm using the official SDK."""

    def __init__(self):
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(f"libKine.so not found at {_LIB_PATH}")
        if not os.path.exists(_CFG_PATH):
            raise FileNotFoundError(f"Config not found at {_CFG_PATH}")

        self._lib = ctypes.CDLL(_LIB_PATH)
        self._init_arms()

    def _init_arms(self):
        """Load config and initialize kinematics for both arms."""
        # Load config → fills TYPE, DH, PNVA, BD arrays for both arms
        TYPE = (c_long * 2)()
        GRV = ((c_double * 3) * 2)()
        DH = (((c_double * 4) * 8) * 2)()
        PNVA = (((c_double * 4) * 7) * 2)()
        BD = (((c_double * 3) * 4) * 2)()
        Mass = ((c_double * 7) * 2)()
        MCP = (((c_double * 3) * 7) * 2)()
        I = (((c_double * 6) * 7) * 2)()

        self._lib.LOADMvCfg.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(c_long * 2),
            ctypes.POINTER((c_double * 3) * 2),
            ctypes.POINTER(((c_double * 4) * 8) * 2),
            ctypes.POINTER(((c_double * 4) * 7) * 2),
            ctypes.POINTER(((c_double * 3) * 4) * 2),
            ctypes.POINTER((c_double * 7) * 2),
            ctypes.POINTER(((c_double * 3) * 7) * 2),
            ctypes.POINTER(((c_double * 6) * 7) * 2),
        ]
        self._lib.LOADMvCfg.restype = c_bool

        ok = self._lib.LOADMvCfg(
            _CFG_PATH.encode("utf-8"),
            ctypes.byref(TYPE), ctypes.byref(GRV), ctypes.byref(DH),
            ctypes.byref(PNVA), ctypes.byref(BD), ctypes.byref(Mass),
            ctypes.byref(MCP), ctypes.byref(I),
        )
        if not ok:
            raise RuntimeError("Failed to load Tianji SDK config")

        # Initialize each arm (0=left, 1=right)
        for arm_idx in range(2):
            serial = c_long(arm_idx)
            robot_type = c_long(TYPE[arm_idx])

            # Extract DH[arm_idx][8][4]
            dh_arr = ((c_double * 4) * 8)()
            for i in range(8):
                for j in range(4):
                    dh_arr[i][j] = DH[arm_idx][i][j]

            # Extract PNVA[arm_idx][7][4]
            pnva_arr = ((c_double * 4) * 7)()
            for i in range(7):
                for j in range(4):
                    pnva_arr[i][j] = PNVA[arm_idx][i][j]

            # Extract BD[arm_idx][4][3]
            bd_arr = ((c_double * 3) * 4)()
            for i in range(4):
                for j in range(3):
                    bd_arr[i][j] = BD[arm_idx][i][j]

            self._lib.FX_Robot_Init_Type.argtypes = [c_long, c_long]
            self._lib.FX_Robot_Init_Type.restype = c_bool
            self._lib.FX_Robot_Init_Kine.argtypes = [c_long, (c_double * 4) * 8]
            self._lib.FX_Robot_Init_Kine.restype = c_bool
            self._lib.FX_Robot_Init_Lmt.argtypes = [c_long, (c_double * 4) * 7, (c_double * 3) * 4]
            self._lib.FX_Robot_Init_Lmt.restype = c_bool

            ok1 = self._lib.FX_Robot_Init_Type(serial, robot_type)
            ok2 = self._lib.FX_Robot_Init_Kine(serial, dh_arr)
            ok3 = self._lib.FX_Robot_Init_Lmt(serial, pnva_arr, bd_arr)

            if not (ok1 and ok2 and ok3):
                raise RuntimeError(f"Failed to init arm {arm_idx}")

        # Suppress SDK debug logging
        self._lib.FX_LOG_SWITCH.argtypes = [c_long]
        self._lib.FX_LOG_SWITCH(c_long(0))

    def set_tool_offset(self, arm: str, z_mm: float):
        """Set a pure-translation tool offset along the flange Z axis.

        This makes FK/IK operate at a point *z_mm* past the flange, matching
        the simulation's EE offset from link7.  For the Wuji palm center the
        value is approximately 90.5 mm (= 107 mm EE offset − 16.5 mm link7→flange).
        """
        serial = c_long(0 if arm == "left" else 1)
        tool = ((c_double * 4) * 4)()
        for i in range(4):
            for j in range(4):
                tool[i][j] = 1.0 if i == j else 0.0
        tool[2][3] = z_mm  # translation along Z

        self._lib.FX_Robot_Tool_Set.argtypes = [c_long, (c_double * 4) * 4]
        self._lib.FX_Robot_Tool_Set.restype = c_bool
        if not self._lib.FX_Robot_Tool_Set(serial, tool):
            raise RuntimeError("Failed to set tool offset")

    def fk(self, arm: str, joints_rad) -> tuple:
        """Forward kinematics: joint angles → (pos_mm, rotmat_3x3).

        Args:
            arm: "left" or "right"
            joints_rad: 7 joint angles in radians (list, numpy, or tensor)

        Returns:
            (pos_mm, rotmat): position in mm and 3x3 rotation matrix
        """
        serial = c_long(0 if arm == "left" else 1)
        joints_deg = [(float(j) * 180.0 / math.pi) for j in joints_rad]
        joints_c = (c_double * 7)(*joints_deg)

        pg = ((c_double * 4) * 4)()
        for i in range(4):
            for j in range(4):
                pg[i][j] = 1.0 if i == j else 0.0

        self._lib.FX_Robot_Kine_FK.argtypes = [
            c_long, ctypes.POINTER(c_double * 7), ctypes.POINTER((c_double * 4) * 4)
        ]
        self._lib.FX_Robot_Kine_FK.restype = c_bool

        ok = self._lib.FX_Robot_Kine_FK(serial, ctypes.byref(joints_c), ctypes.byref(pg))
        if not ok:
            raise RuntimeError("FK failed")

        mat = np.array([[pg[i][j] for j in range(4)] for i in range(4)])
        return mat[:3, 3], mat[:3, :3]

    def solve_ik(self, arm: str, target_pos_m, target_quat_wxyz, ref_joints_rad):
        """Analytical IK: target pose → 7 joint angles.

        Args:
            arm: "left" or "right"
            target_pos_m: (3,) position in meters, in the arm's base frame
            target_quat_wxyz: (4,) quaternion (w, x, y, z)
            ref_joints_rad: (7,) reference joint angles in radians

        Returns:
            numpy array of 7 joint angles in radians, or None if IK fails
        """
        serial = c_long(0 if arm == "left" else 1)

        # Build 4x4 target TCP matrix (mm, row-major flat)
        pos_mm = np.array([float(x) * 1000.0 for x in target_pos_m])
        quat = np.array([float(x) for x in target_quat_wxyz])
        R = _quat_to_rotmat(quat)

        tcp_flat = [0.0] * 16
        for i in range(3):
            for j in range(3):
                tcp_flat[i * 4 + j] = R[i, j]
            tcp_flat[i * 4 + 3] = pos_mm[i]
        tcp_flat[12], tcp_flat[13], tcp_flat[14], tcp_flat[15] = 0, 0, 0, 1

        # Reference joints in degrees
        ref_deg = [float(j) * 180.0 / math.pi for j in ref_joints_rad]

        # Fill structure
        sp = _FX_InvKineSolvePara()
        for i, v in enumerate(tcp_flat):
            sp.m_Input_IK_TargetTCP.data[i] = v
        for i, v in enumerate(ref_deg):
            sp.m_Input_IK_RefJoint.data[i] = v
        sp.m_Input_IK_ZSPType = 0  # minimize distance to reference

        # Call IK
        self._lib.FX_Robot_Kine_IK.argtypes = [c_long, ctypes.POINTER(_FX_InvKineSolvePara)]
        self._lib.FX_Robot_Kine_IK.restype = c_bool

        ok = self._lib.FX_Robot_Kine_IK(serial, ctypes.byref(sp))
        if not ok:
            return None

        # Convert result from degrees to radians
        result_deg = sp.m_Output_RetJoint.to_list()
        return np.array([d * math.pi / 180.0 for d in result_deg])
