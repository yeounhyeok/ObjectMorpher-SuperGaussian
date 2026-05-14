from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvConfig:
    objectmorpher: str = "objectmorpher"
    supergaussian: str = "super_gaussian_eccv24"
    realbasicvsr: str = "realbasicvsr"


DEFAULT_ENVS = EnvConfig()
