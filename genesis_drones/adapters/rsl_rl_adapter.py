import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict

from genesis_drones.envs.genesis_task_env import GenesisTaskEnv, detached_torch_tensor


class RslRlAdapter(VecEnv):
    """Map GenesisTaskEnv EnvStep to rsl_rl VecEnv. No task/dynamics logic."""

    def __init__(self, environment: GenesisTaskEnv, train_config: dict):
        self.environment = environment
        self.num_envs = environment.num_envs
        self.num_actions = environment.action_dim
        self.max_episode_length = environment.task.config.max_episode_steps
        self.episode_length_buf = environment.episode_length_buf
        self.device = environment.device
        self.cfg = train_config
        policy, _state = environment.reset()
        self.policy_observation = detached_torch_tensor(policy)
        self.critic_observation = detached_torch_tensor(policy)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self.policy_observation, "critic": self.critic_observation},
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor):
        env_step = self.environment.step(actions)
        self.policy_observation = detached_torch_tensor(env_step.observation.policy)
        self.critic_observation = detached_torch_tensor(env_step.extras["critic_observation_live"])
        extras_out = {"time_outs": detached_torch_tensor(env_step.truncated)}
        if env_step.done.any() and "n_passed_gates" in env_step.extras:
            reset = env_step.done.bool()
            extras_out["episode"] = {
                "success_rate": detached_torch_tensor(env_step.extras["success"][reset].float()),
                "survive_rate": detached_torch_tensor(env_step.truncated[reset].float()),
                "l_episode": detached_torch_tensor(
                    (env_step.extras["episode_length"][reset] - 1).float() * self.environment.dt
                ),
                "n_passed_gates": detached_torch_tensor(env_step.extras["n_passed_gates"][reset].float()),
            }
        return (
            self.get_observations(),
            detached_torch_tensor(env_step.reward),
            detached_torch_tensor(env_step.done),
            extras_out,
        )

    def reset(self):
        policy, _state = self.environment.reset()
        self.policy_observation = detached_torch_tensor(policy)
        self.critic_observation = detached_torch_tensor(policy)
        return self.get_observations()
