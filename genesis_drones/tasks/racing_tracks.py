import math
from pathlib import Path

from genesis_drones.tasks.racing_core import GateSpec, RaceTrackSpec

GATE_MESH = Path(__file__).resolve().parents[3] / "assets" / "gate" / "gate.glb"


_OPENING_WIDTH = 1.524
_OPENING_HEIGHT = 1.524
_OUTER_WIDTH = 2.1336
_OUTER_HEIGHT = 2.1336
_DEPTH = 0.1323


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)


def _gate(position: tuple[float, float, float], yaw: float) -> GateSpec:
    return GateSpec(
        position=position,
        quaternion=_yaw_quaternion(yaw),
        opening_width=_OPENING_WIDTH,
        opening_height=_OPENING_HEIGHT,
        outer_width=_OUTER_WIDTH,
        outer_height=_OUTER_HEIGHT,
        depth=_DEPTH,
    )


FIXED_SEVEN_GATE_TRACK = RaceTrackSpec(
    name="fixed_seven",
    gates=(
        _gate((0.0, 0.0, 2.0668), 0.0),
        _gate((10.0, 5.0, 1.0668), 0.0),
        _gate((10.0, -5.0, 1.0668), 5.0 * math.pi / 4.0),
        _gate((-5.0, -5.0, 3.5668), math.pi),
        _gate((-5.0, -5.0, 1.0668), 0.0),
        _gate((5.0, 0.0, 1.0668), math.pi / 2.0),
        _gate((0.0, 5.0, 1.0668), math.pi),
    ),
    order=tuple(range(7)),
)


def add_track_gates(scene, track: RaceTrackSpec, *, enable_contact: bool):
    import genesis as gs

    entities = []
    for gate in track.gates:
        yaw_degrees = math.degrees(2.0 * math.atan2(gate.quaternion[3], gate.quaternion[0]))
        entities.append(
            scene.add_entity(
                gs.morphs.Mesh(
                    file=str(GATE_MESH),
                    fixed=True,
                    convexify=enable_contact,
                    collision=enable_contact,
                    file_meshes_are_zup=True,
                    pos=gate.position,
                    euler=(0.0, 0.0, yaw_degrees),
                )
            )
        )
    return entities
