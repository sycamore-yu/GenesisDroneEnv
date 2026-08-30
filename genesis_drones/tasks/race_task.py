import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict

from genesis_drones.envs.race_env import RaceEnv


class RaceTask(VecEnv):
    def __init__(self, environment: RaceEnv, train_config: dict):
        self.environment = environment
        self.num_envs = environment.num_envs
        self.num_actions = environment.action_dim
        self.max_episode_length = environment.config.max_episode_steps
        self.episode_length_buf = environment.episode_length_buf
        self.device = environment.device
        self.cfg = train_config
        self.policy_observation, self.critic_observation = environment.reset()

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self.policy_observation, "critic": self.critic_observation},
            batch_size=self.num_envs,
        )

    def step(self, actions: torch.Tensor):
        policy, (_, _, reward), done, extras = self.environment.step(actions)
        self.policy_observation = policy
        self.critic_observation = extras["critic_observation_live"]
        extras_out = {
            "time_outs": extras["truncated"],
            "log": {
                "/gate_index": extras["gate_index"].to(dtype=reward.dtype),
                "/passed": extras["passed"].to(dtype=reward.dtype),
                "/completed": extras["completed"].to(dtype=reward.dtype),
                "/analytic_collision": extras["analytic_collision"].to(dtype=reward.dtype),
                "/gamma": torch.tensor(self.environment.config.gamma, device=self.device),
                "/lambda": torch.tensor(self.environment.config.td_lambda, device=self.device),
            },
        }
        return self.get_observations(), reward, done, extras_out

    def reset(self):
        self.policy_observation, self.critic_observation = self.environment.reset()
        return self.get_observations()
