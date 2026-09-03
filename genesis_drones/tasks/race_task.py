from genesis_drones.adapters.rsl_rl_adapter import RslRlAdapter
from genesis_drones.envs.race_env import RaceEnv


class RaceTask(RslRlAdapter):
    """Thin PPO entry for racing. Logic lives in RslRlAdapter + GenesisTaskEnv."""

    def __init__(self, environment: RaceEnv, train_config: dict):
        super().__init__(environment.env, train_config)
