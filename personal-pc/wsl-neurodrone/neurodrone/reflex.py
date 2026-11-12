"""Picks the looming detector implementation from the config (snn.impl).

    "hybrid"   snn.LoomingSNN          array maths + 6 spiking decision neurons (the baseline)
    "spiking"  snn_spiking.SpikingLoomingSNN   every stage made of spiking neurons
    "classical" ttc_classical.ClassicalTTC     non-spiking event-based TTC from normal flow (comparison baseline)

Both take EventPackets and return the same ThreatReport, so the rest of the system (Brain,
Coordinator, closed-loop simulator, replay script) does not care which one runs.
"""
from __future__ import annotations

from typing import Optional, Union

from .config import SNNConfig
from .snn import LoomingSNN
from .snn_spiking import SpikingLoomingSNN
from .ttc_classical import ClassicalTTC

IMPLS = ("hybrid", "spiking", "classical")
Detector = Union[LoomingSNN, SpikingLoomingSNN, ClassicalTTC]


def make_snn(width: int, height: int, cfg: Optional[SNNConfig] = None) -> Detector:
    cfg = cfg or SNNConfig()
    if cfg.impl == "spiking":
        return SpikingLoomingSNN(width, height, cfg)
    if cfg.impl == "hybrid":
        return LoomingSNN(width, height, cfg)
    if cfg.impl == "classical":
        return ClassicalTTC(width, height, cfg)
    raise ValueError(f"snn.impl must be one of {IMPLS}, got {cfg.impl!r}")
