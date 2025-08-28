from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

class OptimizationModel(ABC):
    """An abstract base class for optimizatio model."""
    @abstractmethod
    def solve(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        pass



