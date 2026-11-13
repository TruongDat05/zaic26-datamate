"""Locations of the synth repository, controller checkout, and runtime checkout."""
from __future__ import annotations

import os
from pathlib import Path

SYNTH_ROOT = Path(__file__).resolve().parents[2]


def controller_root() -> Path:
    return Path(os.environ.get("DATAMATE_CONTROLLER", SYNTH_ROOT.parent / "datamate_foxel")).resolve()


def upstream_root() -> Path:
    return Path(os.environ.get("DATAMATE_UPSTREAM", controller_root())).resolve()
