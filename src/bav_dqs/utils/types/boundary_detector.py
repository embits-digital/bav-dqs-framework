from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict

@dataclass(frozen=True)
class BoundaryDetectorCfg:
    """Setting the logical parameters of the edge detector.

    Attributes:
        threshold: Critical cumulative probability value to trigger the event.
        edge_window: Number of qubits (sites) monitored at each edge.
        edge_persistence: Requirement of consecutive steps above the threshold to validate the impact.
    """
    threshold: float
    edge_window: int
    edge_persistence: int

@dataclass
class DetectionResult:
    """Data Transfer Object (DTO) for capturing detected collision events.

   This class consolidates the temporal and spatial landmarks identified by the `BoundaryDetector`, 
   serving as a bridge between the detection logic and the final edge_persistence of the simulation metadata.

    Attributes:
        step_left: The time step index of the first confirmed detection on the left edge.
        step_right: The time step index of the first confirmed detection on the right edge.
        first_hit_step: The time step index of the initial global impact (minimum between edges).
        first_side: Identifier of the edge that triggered the first hit ('left', 'right', or 'both').

    Note:
        All fields are initialized as `None` and dynamically populated
        during the quantum evolution loop as detector edge_persistence is reached.
    """
    step_left: Optional[int] = None
    step_right: Optional[int] = None
    first_hit_step: Optional[int] = None
    first_side: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Converts detection results into an attribute dictionary.

        It facilitates direct integration with the `Writer.save_run(attributes=...)` method,
        ensuring that collision metadata is correctly serialized in HDF5 format.

        Returns:
            Dict[str, Any]: Dictionary containing the hit keys and their respective timing steps.
        """
        return {
            "step_left": self.step_left,
            "step_right": self.step_right,
            "first_hit_step": self.first_hit_step,
            "first_side": self.first_side
        }