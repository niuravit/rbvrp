from enum import Enum
from solver.pricing.LabelAbstract import Label
from solver.pricing.DominanceLabel import DominanceLabel


class LabelTWModel(Label):
    """Represents a state in the label-setting algorithm."""
    def __init__(self, i, acc_demand, acc_length, acc_duals, stops, m, reward, reached_flg, prevN, counter,
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
        # self.dominated = dominated # if true, this label is dominated and should not be extended
        self.dominance_label = dominance_label
        self.force_extend = []
        

    def __repr__(self):
        return (f"Label(i={self.i}, acc_d={self.acc_demand:.2f}, acc_l={self.acc_length:.2f}, "
                f"acc_duals={self.acc_duals:.2f}, dominance_label={self.dominance_label}), prevN={self.prevN}),"
                f"force_extend={self.force_extend}, "
                f"m={self.m:.2f}, rwd={self.reward:.2f}, stops={self.stops}, "
                f"reached={self.reached_flg}, counter={self.counter}, ")
    
    def __lt__(self, other):
        """
        Implements lexicographical ordering:
        - acc_demand (ascending)
        - acc_length (ascending)
        - acc_duals (descending)
        """
        if self.acc_demand != other.acc_demand:
            return self.acc_demand < other.acc_demand
        if self.acc_length != other.acc_length:
            return self.acc_length < other.acc_length
        # For acc_duals, we want descending order, so we reverse the comparison
        return self.acc_duals > other.acc_duals