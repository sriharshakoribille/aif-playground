from gymnasium.envs.registration import register

from .frozen_lake_mine import FrozenLakeEnv_Mine

register(
    id="FrozenLake_Mine-v1",
    entry_point="aif_pg.envs.frozen_lake_mine:FrozenLakeEnv_Mine",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)