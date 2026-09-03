from pathlib import Path

import torch

import genesis as gs

from genesis_drones.dynamics.base import DroneState
from genesis_drones.envs.contracts import EnvStep, ObservationBundle
from genesis_drones.envs.differentiable import finish_simulation_window


ASSETS_PATH = Path(__file__).resolve().parents[1] / "robots" / "assets"


def detached_torch_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    return value.as_subclass(torch.Tensor) if isinstance(value, gs.Tensor) else value


def _detach_cached_torch_tensor(value) -> None:
    candidates = [value]
    unwrap = getattr(value, "_unwrap", None)
    if callable(unwrap):
        candidates.append(unwrap())
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("_tc", "_T_tc"):
            tensor = getattr(candidate, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.grad_fn is not None:
                setattr(candidate, name, tensor.detach())


class GenesisTaskEnv:
    """Single Genesis scene shell: dynamics + task_core. No algorithms."""

    def __init__(
        self,
        task_core,
        dynamics_backend,
        num_envs: int,
        requires_grad: bool = True,
        show_viewer: bool = False,
        dt: float | None = None,
        horizon: int = 32,
        gate_entities_fn=None,
        enable_collision: bool = False,
        drone_initial_pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
        add_plane: bool = True,
        add_target_visual: bool = False,
    ):
        self.task = task_core
        self.plant = dynamics_backend
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = gs.device
        self.dynamics = dynamics_backend.name
        self.action_dim = dynamics_backend.action_dim
        self.dt = dt if dt is not None else getattr(getattr(task_core, "config", None), "dt", 0.01)
        self.horizon = horizon
        self._pending_waypoint_sequences = None
        self.target_visual = None
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=1,
                substeps_local=horizon if requires_grad else 1,
                requires_grad=requires_grad,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=enable_collision,
                enable_joint_limit=True,
                batch_links_info=dynamics_backend.batch_links_info,
            ),
            show_viewer=show_viewer,
        )
        if add_plane:
            self.scene.add_entity(gs.morphs.Plane())
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file=str(ASSETS_PATH / "drone_urdf" / "drone.urdf"),
                pos=drone_initial_pos,
                euler=(0.0, 0.0, 0.0),
                default_armature=2.6e-7,
            ),
        )
        if gate_entities_fn is not None and show_viewer:
            gate_entities_fn(self.scene)
        if add_target_visual:
            self.target_visual = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=str(ASSETS_PATH / "primitives" / "sphere.obj"),
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        self.scene.build(n_envs=num_envs)
        self.plant.attach(self.drone, self.scene)
        self.last_action = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=gs.tc_float)
        self.is_alive = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self._simulation_forces: list[torch.Tensor] = []
        self.respawn_on_fail = True

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.task.episode_length_buf

    def _read_state(self) -> DroneState:
        solver_state = self.scene.rigid_solver.get_state()
        link_idx = self.drone.base_link_idx
        dof_start = self.drone.dof_start
        quaternion = solver_state.links_quat[:, link_idx]
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
        return DroneState(
            solver_state.links_pos[:, link_idx],
            quaternion,
            solver_state.dofs_vel[:, dof_start : dof_start + 3],
            solver_state.dofs_vel[:, dof_start + 3 : dof_start + 6],
        )

    def hover_command(self, count: int | None = None) -> torch.Tensor:
        return self.plant.hover_command(self.num_envs if count is None else count)

    def release_simulation_graphs(self, physics_loss: torch.Tensor | None = None) -> None:
        if physics_loss is not None and physics_loss.grad_fn is not None:
            torch.autograd.backward(physics_loss)
        self._simulation_forces.clear()
        _detach_cached_torch_tensor(self.scene.rigid_solver.dyn_state.dofs.ctrl_force)
        self.last_action = detached_torch_tensor(self.last_action)
        self.plant.detach()
        if hasattr(self.task, "last_action"):
            self.task.last_action = detached_torch_tensor(self.task.last_action)

    def _apply_initial_pose(self, initial_states, environment_indices: torch.Tensor | None = None) -> None:
        position = initial_states.position.to(device=self.device, dtype=gs.tc_float)
        quaternion = initial_states.quaternion.to(device=self.device, dtype=gs.tc_float)
        linear_velocity = initial_states.linear_velocity.to(device=self.device, dtype=gs.tc_float)
        self.drone.set_pos(position, envs_idx=environment_indices, zero_velocity=True)
        self.drone.set_quat(quaternion, envs_idx=environment_indices, zero_velocity=True)
        self.drone.set_dofs_velocity(
            torch.cat((linear_velocity, torch.zeros_like(linear_velocity)), dim=-1),
            envs_idx=environment_indices,
        )

    def _set_initial_states(
        self,
        initial_states=None,
        environment_indices: torch.Tensor | None = None,
        seed: int | None = None,
        waypoint_sequences: torch.Tensor | None = None,
    ) -> None:
        indices = (
            torch.arange(self.num_envs, device=self.device)
            if environment_indices is None
            else environment_indices
        )
        waypoints = waypoint_sequences if waypoint_sequences is not None else self._pending_waypoint_sequences
        reset_kwargs = {"initial_states": initial_states, "seed": seed}
        if waypoints is not None or hasattr(self.task, "sample_target"):
            reset_kwargs["waypoint_sequences"] = waypoints
        try:
            initial_states = self.task.reset_task(indices, **reset_kwargs)
        except TypeError:
            # RacingCore has no waypoint_sequences.
            reset_kwargs.pop("waypoint_sequences", None)
            initial_states = self.task.reset_task(indices, **reset_kwargs)
        self._apply_initial_pose(initial_states, environment_indices)
        self.plant.reset(indices, seed=seed, environment_indices=environment_indices)
        if hasattr(self.task, "last_action"):
            self.last_action[indices] = self.task.last_action[indices]
        else:
            self.last_action[indices] = 0.0
        self.is_alive[indices] = True
        if hasattr(self.task, "is_alive"):
            self.task.is_alive[indices] = True

    def reset(self, initial_states=None, seed: int | None = 0, waypoint_sequences: torch.Tensor | None = None):
        self.release_simulation_graphs()
        self.scene.reset()
        self._pending_waypoint_sequences = waypoint_sequences
        self._set_initial_states(initial_states, seed=seed, waypoint_sequences=waypoint_sequences)
        self._pending_waypoint_sequences = None
        self._sync_target_visual()
        policy, state_obs = self.task.observe(self._read_state())
        return policy, state_obs

    def _sync_target_visual(self) -> None:
        if self.target_visual is None or not hasattr(self.task, "target_position"):
            return
        self.target_visual.set_pos(self.task.target_position, zero_velocity=True)

    def step(self, action: torch.Tensor) -> EnvStep:
        action = action.clamp(-1.0, 1.0)
        is_alive_before = self.is_alive.clone()
        state_before = self._read_state()
        control = self.plant.control(action, state_before, self.is_alive)
        generalized_force = control.generalized_force
        if generalized_force.requires_grad:
            if isinstance(generalized_force, gs.Tensor):
                generalized_force.scene = None
            else:
                generalized_force = gs.from_torch(generalized_force, detach=False, requires_grad=True)
            self._simulation_forces.append(generalized_force)
        self.drone.control_dofs_force(generalized_force)
        self.scene.step()
        state = self._read_state()
        self.plant.after_step(state, is_alive_before)
        result = self.task.evaluate(state_before, state, action, is_alive_before)
        done = result.terminated | result.truncated
        self.is_alive = is_alive_before & ~done
        if hasattr(self.task, "is_alive"):
            self.task.is_alive = self.is_alive
        if hasattr(self.task, "last_action"):
            self.last_action = self.task.last_action
        else:
            self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        self._sync_target_visual()

        policy_before, state_before_obs = self.task.observe(state)
        if self.respawn_on_fail and not self.requires_grad:
            reset_idx = done.nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
            policy, state_obs = self.task.observe(self._read_state())
        else:
            policy, state_obs = policy_before, state_before_obs

        critic = policy_before
        metrics = getattr(result, "metrics", {}) or {}
        extras = {
            "terminated": result.terminated.detach(),
            "truncated": result.truncated.detach(),
            "reset": done.detach(),
            "alive": self.is_alive.detach(),
            "critic_observation": critic,
            "critic_observation_live": policy,
            "state": state_before_obs,
            "state_live": state_obs,
            "actual_wrench": control.wrench,
            "motor_thrust": control.motor_thrust,
            **{key: value for key, value in metrics.items() if key != "loss_components"},
            "loss_components": metrics.get("loss_components", getattr(result, "loss_components", {})),
            "passed": getattr(result.events, "passed", None),
            "wrong_way": getattr(result.events, "wrong_way", None),
            "analytic_collision": getattr(result.events, "analytic_collision", None),
            "arrived": getattr(result.events, "arrived", None),
        }
        if extras["passed"] is not None:
            extras["passed"] = extras["passed"].detach()
            extras["wrong_way"] = extras["wrong_way"].detach()
            extras["analytic_collision"] = extras["analytic_collision"].detach()
        if extras["arrived"] is not None:
            extras["arrived"] = extras["arrived"].detach()
        return EnvStep(
            observation=ObservationBundle(policy=policy, critic=policy, state=state_obs),
            reward=result.reward,
            physics_loss=result.physics_loss,
            policy_loss=result.policy_loss,
            terminated=result.terminated,
            truncated=result.truncated,
            done=done.detach(),
            events=result.events,
            metrics=metrics,
            extras=extras,
        )

    def _reset_envs(self, environment_indices: torch.Tensor) -> None:
        count = int(environment_indices.numel())
        if count == 0:
            return
        seed = int(torch.randint(0, 2**31 - 1, (), device=self.device).item())
        self._set_initial_states(None, environment_indices, seed=seed)
        self._sync_target_visual()

    def detach_window(self):
        self.release_simulation_graphs()
        self.last_action = detached_torch_tensor(self.last_action)
        self.is_alive = detached_torch_tensor(self.is_alive)
        if hasattr(self.task, "detach_buffers"):
            self.task.detach_buffers(detached_torch_tensor)
        if hasattr(self.task, "target_gates"):
            self.task.target_gates = detached_torch_tensor(self.task.target_gates)
        if hasattr(self.task, "episode_length_buf"):
            self.task.episode_length_buf = detached_torch_tensor(self.task.episode_length_buf)
        if hasattr(self.task, "is_alive"):
            self.task.is_alive = self.is_alive
        if hasattr(self.task, "last_action"):
            self.last_action = self.task.last_action
        if self.respawn_on_fail:
            reset_idx = (~self.is_alive).nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
        policy, _ = self.task.observe(self._read_state())
        return detached_torch_tensor(policy), detached_torch_tensor(policy)

    def finish_window(self, physics_loss: torch.Tensor, simulation_actions: list[torch.Tensor]):
        action_gradients = finish_simulation_window(self, physics_loss, simulation_actions)
        policy, critic = self.detach_window()
        return policy, critic, action_gradients
