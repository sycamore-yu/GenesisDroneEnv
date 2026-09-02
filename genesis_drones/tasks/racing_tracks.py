import math
from pathlib import Path

from genesis_drones.tasks.racing_core import GateSpec, RaceTrackSpec


GATE_MESH = Path(__file__).resolve().parents[3] / "assets" / "gate" / "gate.glb"
GATE_MESH_OPENING_OFFSET_Z = 1.0668


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)


def _gate(position: tuple[float, float, float], yaw: float) -> GateSpec:
    return GateSpec(
        position=position,
        quaternion=_yaw_quaternion(yaw),
        opening_width=3.0,
        opening_height=3.0,
        outer_width=3.0,
        outer_height=3.0,
        depth=0.0,
    )


RACING_TRACK = RaceTrackSpec(
    name="diffaero_racing",
    gates=(
        _gate((1.5, -1.5, 1.5), math.pi / 2.0),
        _gate((0.0, 0.0, 1.5), math.pi),
        _gate((-1.5, 1.5, 1.5), math.pi / 2.0),
        _gate((0.0, 3.0, 1.5), 0.0),
        _gate((1.5, 1.5, 1.5), -math.pi / 2.0),
        _gate((0.0, 0.0, 1.5), -math.pi),
        _gate((-1.5, -1.5, 1.5), -math.pi / 2.0),
        _gate((0.0, -3.0, 1.5), 0.0),
    ),
    order=tuple(range(8)),
)


def add_track_gates(scene, track: RaceTrackSpec):
    import genesis as gs

    entities = []
    for gate in track.gates:
        yaw_degrees = math.degrees(2.0 * math.atan2(gate.quaternion[3], gate.quaternion[0]))
        x, y, z = gate.position
        entities.append(
            scene.add_entity(
                gs.morphs.Mesh(
                    file=str(GATE_MESH),
                    fixed=True,
                    convexify=False,
                    collision=False,
                    file_meshes_are_zup=True,
                    pos=(x, y, z - GATE_MESH_OPENING_OFFSET_Z),
                    euler=(0.0, 0.0, yaw_degrees),
                )
            )
        )
    return entities
