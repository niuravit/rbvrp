from enum import Enum

class DominanceLabel(Enum):
    UNDEFINED = "undefined"
    STRONGLY_DOMINANT = "strongly_dominant"
    SEMISTRONGLY_DOMINANT = "semistrongly_dominant"
    WEAKLY_DOMINANT = "weakly_dominant"