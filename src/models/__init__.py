"""Synopsis-specific static and temporal host-pair models."""

from .static_e_graphsage import StaticEGraphSAGE
from .temporal_attack_model import TemporalAttackPathModel

__all__ = ["StaticEGraphSAGE", "TemporalAttackPathModel"]
