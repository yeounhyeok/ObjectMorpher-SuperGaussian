from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(slots=True)
class UpsamplingCommand:
    prior: str
    input_dir: Path
    output_dir: Path


def build_upsampling_command(cmd: UpsamplingCommand) -> list[str]:
    """Scaffold only: actual SuperGaussian helper invocation is deferred."""
    return [
        "python",
        "-m",
        "adapter_placeholder_supergaussian_upsampling",
        f"--prior={cmd.prior}",
        f"--input={cmd.input_dir}",
        f"--output={cmd.output_dir}",
    ]


def run_upsampling(cmd: UpsamplingCommand, check: bool = False) -> subprocess.CompletedProcess[str] | None:
    if check:
        return subprocess.run(build_upsampling_command(cmd), check=False, text=True, capture_output=True)
    return None
