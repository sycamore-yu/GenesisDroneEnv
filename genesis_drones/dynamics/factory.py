import torch

from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.dynamics.full_quad import FullQuadBackend
from genesis_drones.dynamics.native_quad import NativeQuadBackend


BACKENDS = {
    "native_quad": NativeQuadBackend,
    "full_quad": FullQuadBackend,
}
QUAD_DYNAMICS = tuple(BACKENDS)


def make_dynamics_backend(
    name: str,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
    dt: float,
    controller_config: CtbrControllerConfig | None = None,
    native_config: NativeQuadConfig | None = None,
):
    if name not in BACKENDS:
        raise ValueError(f"unsupported dynamics: {name}; valid: {', '.join(BACKENDS)}")
    kwargs = {"num_envs": num_envs, "device": device, "dtype": dtype, "dt": dt}
    if name == "full_quad":
        kwargs["controller_config"] = controller_config or CtbrControllerConfig()
    else:
        kwargs["config"] = native_config or NativeQuadConfig()
    return BACKENDS[name](**kwargs)
