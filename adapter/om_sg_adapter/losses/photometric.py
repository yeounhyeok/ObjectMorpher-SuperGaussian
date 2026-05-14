from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PhotometricLossConfig:
    use_l1: bool = True
    use_lpips: bool = False


# Phase 1 intentionally leaves actual optimization out.
