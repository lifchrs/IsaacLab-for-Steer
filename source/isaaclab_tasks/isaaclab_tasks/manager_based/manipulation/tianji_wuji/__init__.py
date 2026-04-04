import gymnasium as gym

from .tianji_wuji_env_cfg import TianjiWujiEnvCfg

gym.register(
    id="Isaac-Tianji-Wuji-v0",
    entry_point="isaaclab.envs:ManagerBasedEnv",
    kwargs={"env_cfg_entry_point": TianjiWujiEnvCfg},
    disable_env_checker=True,
)
