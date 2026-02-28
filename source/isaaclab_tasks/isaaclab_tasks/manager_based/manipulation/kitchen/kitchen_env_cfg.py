# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp

KITCHEN_SCENE_DIR = "/home/chuanruo/IsaacLab/assets/ArtVIP/Interactive_scene/kitchen"


##
# Scene definition
##
@configclass
class KitchenSceneCfg(InteractiveSceneCfg):
    """Configuration for the kitchen scene with the ArtVIP Interactive Kitchen.

    Each object is loaded from its own USD file so that there are no duplicate
    meshes.  Articulated objects use ``ArticulationCfg`` (joint control + state
    queries); static props use ``AssetBaseCfg``.

    NOTE: The ``init_state.pos`` / ``rot`` values below are **placeholders**.
    Open ``Interactive_kitchen.usd`` in Isaac Sim, read the world-space
    transform of each referenced sub-asset, and paste the values here.
    """

    # ── Static kitchen structure (walls, countertops, floor tiles) ──────
    kitchen_structure = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KitchenStructure",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "kitchen", "model_kitchen.usd"),
        ),
    )

    floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "07item", "floor", "model_floor.usd"),
        ),
    )

    # ── Articulated objects (interactive, with physics) ─────────────────
    # Access at runtime: env.scene["fridge"], env.scene["oven"], etc.

    fridge = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "fridge", "model_fridge.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "RevoluteJoint_fridge_left": 0.0,
                "RevoluteJoint_fridge_right": 0.0,
            },
        ),
        actuators={
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint_fridge_.*"],
                effort_limit_sim=87.0,
                stiffness=20.0,
                damping=2.5,
            ),
        },
    )

    oven = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Oven",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "oven", "model_oven.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "PrismaticJoint_oven_down": 0.0,
                "RevoluteJoint_oven_up": 0.0,
            },
        ),
        actuators={
            "oven_joints": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint_oven_.*", "RevoluteJoint_oven_.*"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
        },
    )

    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "door", "model_door.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "PrismaticJoint_door_left": 0.0,
                "PrismaticJoint_door_right": 0.0,
            },
        ),
        actuators={
            "door_joints": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint_door_.*"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
        },
    )

    access_control = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/AccessControl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "access_control", "access_control.usd"),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "RevoluteJoint_access_control": 0.0,
            },
        ),
        actuators={
            "access_control_joints": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint_access_control"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
        },
    )

    # ── Additional articulated objects (uncomment and adjust as needed) ──
    # cupboard = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Cupboard",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(KITCHEN_SCENE_DIR, "cupboard", "model_cupboard.usd"),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #     actuators={...},
    # )
    #
    # largecabinet = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/LargeCabinet",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(KITCHEN_SCENE_DIR, "largecabinet", "model_largecabinet.usd"),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #     actuators={...},
    # )
    #
    # ricecooker = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/RiceCooker",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(KITCHEN_SCENE_DIR, "ricecooker", "model_ricecooker.usd"),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #     actuators={...},
    # )
    #
    # juicer = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Juicer",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(KITCHEN_SCENE_DIR, "juicer", "model_juicer.usd"),
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #     actuators={...},
    # )

    # ── Static props (non-articulated items) ────────────────────────────
    chair = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Chair",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "chair", "model_chair.usd"),
        ),
    )

    extractor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Extractor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "extractor", "extractor.usd"),
        ),
    )

    bakingchamber = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BakingChamber",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "bakingchamber", "model_bakingchamber.usd"),
        ),
    )

    platerack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlateRack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "platerack", "platerack.usd"),
        ),
    )

    switch = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Switch",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(KITCHEN_SCENE_DIR, "switch", "switch.usd"),
        ),
    )

    # ── Environment infrastructure ──────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(2000.0, 2000.0, 0.02),
            visible=False,
            collision_props=sim_utils.schemas.CollisionPropertiesCfg(),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_action = ObsTerm(func=mdp.last_droid_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class KitchenEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kitchen environment."""

    scene: KitchenSceneCfg = KitchenSceneCfg(
        num_envs=4096, env_spacing=25, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 6
        self.episode_length_s = 30.0
        self.sim.dt = 1 / (6 * 15)
        self.sim.render_interval = self.decimation

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
