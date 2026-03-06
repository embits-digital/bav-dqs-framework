from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class BoundaryDetectorCfg:
    threshold: float
    edge_window: int
    persistence: int

@dataclass
class DetectionResult:
    """Objeto de transferência de dados para os hits detectados."""
    step_left: Optional[int] = None
    step_right: Optional[int] = None
    first_hit_step: Optional[int] = None
    first_side: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Facilita a passagem para o Writer.save_run(attributes=...)"""
        return {
            "step_left": self.step_left,
            "step_right": self.step_right,
            "first_hit_step": self.first_hit_step,
            "first_side": self.first_side
        }