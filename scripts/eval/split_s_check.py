"""Open-loop CTBR check that gate 4 to gate 5 is feasible. Not used by training."""

import argparse

import torch

import genesis as gs

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.tasks.racing_core import EvaluationInitialStates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    environment = RaceEnv(RaceEnvConfig(horizon=1, max_episode_steps=args.steps), 1, requires_grad=False)
    start = EvaluationInitialStates(
        position=torch.tensor([[-7.0, -5.0, 3.5]], device=gs.device),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device),
        linear_velocity=torch.zeros((1, 3), device=gs.device),
    )
    environment.reset(initial_states=start)
    environment.respawn_on_fail = False
    environment.gate_index.fill_(4)
    hover = float(environment.controller.hover_action)
    passed = False
    for _ in range(args.steps):
        height = float(environment._read_state().position[0, 2])
        if height > 1.5:
            action = torch.tensor([[hover - 0.02, 0.0, 0.0, 0.0]], device=gs.device)
        elif height > 1.15:
            action = torch.tensor([[hover, 0.0, 0.0, 0.0]], device=gs.device)
        else:
            action = torch.tensor([[hover + 0.05, 0.0, 0.08, 0.0]], device=gs.device)
        _, _, _, extras = environment.step(action)
        if bool(extras["passed"][0]) and int(extras["gate_index"][0]) >= 5:
            passed = True
            break
    print({"passed_gate_5": passed, "gate_index": int(environment.gate_index[0]), "steps": int(environment.episode_length_buf[0])})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
