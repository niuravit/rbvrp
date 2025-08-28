from abc import ABC, abstractmethod
from solver.pricing.DominanceLabel import DominanceLabel

class Label(ABC):
    """
    An abstract base class for labels in a label-setting algorithm.
    All concrete label implementations must inherit from this class.
    """
    def __init__(self, i,acc_demand, acc_length, acc_duals, stops, m, reward, reached_flg, prevN, counter,
                  dominance_label: DominanceLabel = DominanceLabel.UNDEFINED):
        self.i = i
        self.acc_demand = acc_demand
        self.acc_length = acc_length
        self.acc_duals = acc_duals
        self.stops = stops
        self.m = m
        self.reward = reward
        self.reached_flg = reached_flg
        self.prevN = prevN
        self.counter = counter
        self.dominance_label = dominance_label
        self.force_extend = []

    @abstractmethod
    def __lt__(self, other) -> bool:
        """
        Defines the less-than comparison for sorting and dominance checks.
        Must return a boolean.
        """
        pass