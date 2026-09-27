"""Transparent risk ranking for predicted host-to-host attack edges."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAssessment:
    probability: float
    criticality: float
    risk_score: float
    risk_level: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "probability": self.probability,
            "criticality": self.criticality,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
        }


def score_risk(probability: float, criticality: float, max_criticality: float = 5.0) -> RiskAssessment:
    """Return 0–100 risk = attack probability × normalized asset criticality.

    Criticality is an explicit asset-inventory input on [0, max_criticality].
    No criticality value is inferred from red-team labels or host identifiers.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if max_criticality <= 0.0:
        raise ValueError("max_criticality must be positive")
    if not 0.0 <= criticality <= max_criticality:
        raise ValueError(f"criticality must be between 0 and {max_criticality}")
    score = 100.0 * probability * (criticality / max_criticality)
    level = "critical" if score >= 80 else "high" if score >= 60 else "medium" if score >= 30 else "low"
    return RiskAssessment(probability, criticality, score, level)
