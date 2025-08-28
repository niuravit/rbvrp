from bisect import bisect_left
from typing import Dict, List, TypeVar, Generic
from solver.pricing.LabelAbstract import Label


# Define a TypeVar to represent the generic label type
T = TypeVar('T', bound=Label)

class StateStorage(Generic[T]):
    """
    A generic storage for labels in a label-setting algorithm.
    It stores labels in sorted lists, mapped by a node index 'i'.
    """
    def __init__(self):
        # Dictionary mapping a node index 'i' to a list of labels of type T
        self.S: Dict[int, List[T]] = {}

    def insert_label(self, label: T) -> None:
        """
        Inserts a new label into the appropriate position in S[i] while 
        maintaining sort order. Assumes the label type T supports the
        comparison operators required by bisect_left (e.g., __lt__).
        """
        if label.i not in self.S:
            self.S[label.i] = []
        
        # Find the insertion point using binary search
        # This assumes that the label object has an attribute 'i' and supports comparison
        idx = bisect_left(self.S[label.i], label)
        self.S[label.i].insert(idx, label)

    def remove_label(self, label: T) -> None:
        """
        Removes a specific label from the storage.
        """
        i = label.i
        if i in self.S:
            self.S[i].remove(label)

    def get_labels(self, i: int) -> List[T]:
        """
        Returns all labels for node i.
        """
        return self.S.get(i, [])

    def clear_node(self, i: int) -> None:
        """
        Clears all labels for node i.
        """
        if i in self.S:
            self.S[i].clear()