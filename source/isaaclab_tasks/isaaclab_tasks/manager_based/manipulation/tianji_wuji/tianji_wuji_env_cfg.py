"""Tianji + Wuji unified positional control environment configuration.

52D action space:
    [0:6]   left arm IK delta   (dx, dy, dz, droll, dpitch, dyaw)
    [6:12]  right arm IK delta  (dx, dy, dz, droll, dpitch, dyaw)
    [12:32] left hand finger joints  (20 direct joint positions)
    [32:52] right hand finger joints (20 direct joint positions)

All actuator parameters (stiffness, damping, effort limits) come from the USD.
No overrides.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.tianji_wuji import TIANJI_WUJI_CFG

# EE offset from link7 toward the Wuji palm center
WUJI_EE_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(
    pos=[0.0, 0.0, 0.107], rot=[0.0, 0.0, 0.0, 1.0],
)


@configclass
class TianjiWujiSceneCfg(InteractiveSceneCfg):
    """Scene: Tianji+Wuji robot, ground, table, EE frame sensors, camera."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = TIANJI_WUJI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.6, 0.8, 0.8),
        ),
    )
    left_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/left_link7",
                name="left_ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ],
    )
    right_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/marvin_robot/right_link7",
                name="right_ee",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ],
    )
    overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.2, horizontal_aperture=5.376),
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.5, 0.8), rot=(0.426, 0.227, 0.435, 0.755), convention="opengl"),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """52D: 6 left IK + 6 right IK + 20 left hand + 20 right hand."""

    left_arm_action: ActionTermCfg = MISSING
    right_arm_action: ActionTermCfg = MISSING
    left_hand_action: ActionTermCfg = MISSING
    right_hand_action: ActionTermCfg = MISSING


@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TianjiWujiEnvCfg(ManagerBasedEnvCfg):
    """Unified positional control environment for Tianji arms + Wuji hands."""

    scene: TianjiWujiSceneCfg = TianjiWujiSceneCfg(
        num_envs=1, env_spacing=5.0, replicate_physics=False,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events = None
    commands = None

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = 1

        # No actuator overrides — all values from USD.

        # IK relative for arms (6D delta per step)
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_joint.*"],
            body_name="left_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
            ),
            scale=0.5,
            body_offset=WUJI_EE_OFFSET,
        )
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_joint.*"],
            body_name="right_link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
            ),
            scale=0.5,
            body_offset=WUJI_EE_OFFSET,
        )

        # Direct joint position for hands (20D each)
        self.actions.left_hand_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_finger.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.right_hand_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_finger.*"],
            scale=1.0,
            use_default_offset=True,
        )
