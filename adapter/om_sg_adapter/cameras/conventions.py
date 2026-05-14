from __future__ import annotations

from copy import deepcopy
from typing import Sequence

Matrix4 = list[list[float]]


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> Matrix4:
    return [list(row) for row in deepcopy(matrix)]


def opencv_c2w_to_supergaussian_c2w(c2w: Sequence[Sequence[float]]) -> Matrix4:
    """Apply the OpenCV->OpenGL axis flip used by SuperGaussian transforms.json export."""
    converted = _copy_matrix(c2w)
    for row in range(3):
        converted[row][1] *= -1.0
        converted[row][2] *= -1.0
    return converted


def supergaussian_c2w_to_opencv_c2w(c2w: Sequence[Sequence[float]]) -> Matrix4:
    converted = _copy_matrix(c2w)
    for row in range(3):
        converted[row][1] *= -1.0
        converted[row][2] *= -1.0
    return converted
