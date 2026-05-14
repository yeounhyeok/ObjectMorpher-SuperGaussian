from __future__ import annotations

from dataclasses import dataclass
import math

Matrix4 = list[list[float]]
Vector3 = tuple[float, float, float]


@dataclass(slots=True)
class OrbitTrajectorySpec:
    frames: int = 24
    radius: float = 2.0
    elevation_degrees: float = 15.0
    target: Vector3 = (0.0, 0.0, 0.0)


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vec
    denom = math.sqrt(x * x + y * y + z * z)
    if denom == 0:
        return vec
    return (x / denom, y / denom, z / denom)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def build_look_at_c2w(camera_position: Vector3, target: Vector3) -> Matrix4:
    forward = _normalize(tuple(cp - tp for cp, tp in zip(camera_position, target)))
    right = _normalize(_cross((0.0, 1.0, 0.0), forward))
    up = _normalize(_cross(forward, right))
    return [
        [right[0], up[0], forward[0], camera_position[0]],
        [right[1], up[1], forward[1], camera_position[1]],
        [right[2], up[2], forward[2], camera_position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_orbit_cameras(spec: OrbitTrajectorySpec) -> list[Matrix4]:
    tx, ty, tz = spec.target
    elevation = math.radians(spec.elevation_degrees)
    poses: list[Matrix4] = []
    for idx in range(spec.frames):
        azimuth = (2.0 * math.pi * idx) / spec.frames
        position = (
            spec.radius * math.cos(elevation) * math.sin(azimuth) + tx,
            -spec.radius * math.sin(elevation) + ty,
            spec.radius * math.cos(elevation) * math.cos(azimuth) + tz,
        )
        poses.append(build_look_at_c2w(position, spec.target))
    return poses
