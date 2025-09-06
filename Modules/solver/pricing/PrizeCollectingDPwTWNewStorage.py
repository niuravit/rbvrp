from ast import For
import pandas as pd
import numpy as np
import time
from operator import attrgetter
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import math
from solver.pricing.LabelSetting import LabelSetting
from solver.pricing.LabelTWModel import LabelTWModel
from solver.pricing.DominanceLabel import DominanceLabel
from solver.pricing.StateStorage import StateStorage
from collections import Counter
from queue import PriorityQueue

# Concrete implementation of the abstract class
# refactoring the previous PrizeCollectingDPwTWVer3
class PrizeCollectingDPwTWNewStorage(LabelSetting):
    def __init__(self, _n: int, _C: Dict, _Q: List, _dual: np.ndarray,
                 _s0: float, _veh_cap: float, _time_window: float, _wavg_factor: float,
                 _m_lim: int, _dom_ver: int, 
                 _time_limit: int = np.inf, _stop_lim: int = np.inf, _ch_dom: bool = True, 
                 _bch_cond: List = [], 
                 _forbid_link_dict: Dict[int, List[int]] = {},
                 _necess_link_dict: Dict[int, List[int]] = {},            
                 **kwargs):
        self.DEPOT = 0
        self.n = _n
        self.C = _C
        self.Q = _Q
        self.dual = _dual
        self.s0 = _s0
        self.veh_cap = _veh_cap
        self.time_window = _time_window
        self.wavg_factor = _wavg_factor
        self.m_lim = _m_lim
        self.domVer = _dom_ver
        self.time_limit = _time_limit
        self.stop_lim = _stop_lim
        self.ch_dom = _ch_dom
        self.bch_cond = _bch_cond
        self.forbid_link_dict = _forbid_link_dict
        self.necess_link_dict = _necess_link_dict
        self.storage = StateStorage()
        self.epsilon = 0 #1e-8
        self.no_profitable_route_iteration_max_tolerance = 120
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def solve(self) -> Tuple[Dict[int, List[LabelTWModel]], Tuple[int, int]]:        
        print('Solving time limit set to:', self.time_limit, 'secs.', "Dominance Checking:", self.ch_dom)
        if self.ch_dom:
            if self.domVer not in [2, 3, 4]:
                print("Wrong input of dominance version! _domVer should be in", [2, 3, 4])
                raise ValueError("Invalid dominance version.")
            else:
                print("Dominance Version:", self.domVer)
        
        print("bch-conds:", self.get('bch_cond', []))
        print("forbidden_link:", self.forbid_link_dict)
        print("necessary_link:", self.necess_link_dict)

        process_queue = PriorityQueue()
        
        _counter = 0; _rch_counter = 0
        # Insert the initial label for the depot
        initial_label = LabelTWModel(0, 0, 0, 0, 0, 0, False, False, 0, False)
        self.storage.insert_label(initial_label)

        # For initial label, use neutral priority scores
        process_queue.put((0, 0, 0, initial_label.counter, initial_label))

        tFlag = False
        _time = time.time()
        _neg_cost_counter = 0
        
        while not tFlag:           
            # for i in range(self.n + 1):
            #     current_labels_at_i = list(self.storage.get_labels(i))
            #     for current_label in current_labels_at_i:

            if not process_queue.empty():
                _, _, _, counter, current_label = process_queue.get()
                if True:

                    exit_now, _neg_cost_counter, _time =  self.early_terminate_if_profitable_states_exist(_time, self.storage, _neg_cost_counter)
                    if exit_now: return self.storage.S, (_counter, _rch_counter)
                    if (not current_label.reached_flg) and (not current_label.dominated):
                        if current_label.stops >= self.stop_lim:
                            current_label.reached_flg = True
                        else:
                            i = current_label.i
                            if (current_label.dominance_label == DominanceLabel.WEAKLY_DOMINANT):
                                reachTo = [
                                    x for x in current_label.force_extend if (x != i) 
                                    and (x not in self.forbid_link_dict[i]) 
                                    and (x != current_label.prevN)
                                    ]
                            else:
                                reachTo = [
                                    x for x in range(1, self.n + 1)
                                    if (x != i) and (x not in self.forbid_link_dict[i]) and (x != current_label.prevN)
                                ]
                            current_label.reached_flg = True
                            
                            for j in reachTo:
                                _rch_counter += 1
                                _d = current_label.acc_demand + self.Q[j]
                                _l = current_label.acc_length + self.C[(i, j)]
                                _p = current_label.stops + 1
                                
                                if (self.time_window - _l) > 0:
                                    m_ctc = np.ceil((_d * (_l + self.C[(j, 0)])) / self.veh_cap)
                                    m_tw = np.ceil((_l + self.C[(j, 0)]) / (self.time_window - _l))
                                    m = max(m_ctc, m_tw)
                                    
                                    _rwd = current_label.reward + self.dual[j - 1] + current_label.m - m
                                    _dual = current_label.acc_duals + self.dual[j - 1]
                                    
                                    if m <= self.m_lim:
                                        _counter += 1
                                        new_label = LabelTWModel(j, _d, _l, _dual, _p, m, _rwd, False, i, _counter, DominanceLabel.UNDEFINED)
                                        if self.ch_dom:
                                            is_dominated = self._check_dominance(new_label)
                                            if not is_dominated:
                                                # Negate priorities for max-heap behavior
                                                priorities = self.get_label_priority(new_label)
                                                process_queue.put(
                                                    (-priorities[0],      # Resource score (higher better)
                                                    -priorities[1],      # Reward efficiency (higher better)
                                                    -priorities[2],      # Path quality (higher better)
                                                    new_label.counter,   # Unique ID for tie-breaking
                                                    new_label)
                                                )
                                        else:
                                            self.storage.insert_label(new_label)
                                            
            _all_pc = all(all(label.reached_flg for label in self.storage.get_labels(i)) for i in range(self.n + 1))
            if _rch_counter%200==0:
                print(f"Reach counter: {_rch_counter}, States count:",self.log_state_storage_types())
            if _all_pc:
                tFlag = True
        return self.storage.S, (_counter, _rch_counter)
        # return S, (_counter, _rch_counter)
    
    def early_terminate_if_profitable_states_exist(self, cur_time: float, storage: StateStorage, neg_cost_counter: int) -> Tuple[bool, int, float]:
    # def early_terminate_if_profitable_states_exist(self, cur_time: float, S: List[List[LabelTWModel]], neg_cost_counter: int) -> Tuple[bool, int, float]:
        """
        Check if we should terminate early based on time limit and profitable states.
        
        Args:
            cur_time: Current timestamp when checking
            S: List of lists containing Label objects
            neg_cost_counter: Counter for consecutive negative cost iterations
            
        Returns:
            Tuple of (should_terminate: bool, updated_counter: int, updated_time: float)
        """
        elapsed_time = time.time() - cur_time
        if elapsed_time <= self.time_limit:
            return False, neg_cost_counter, cur_time
            
        print(f'Reach time limit!! {elapsed_time:.2f}/{self.time_limit}')
        
        # Check for profitable states (reward > epsilon)
        if any(label.reward > 0.000001 for sublist in storage.S.values() for label in sublist):
            return True, neg_cost_counter, cur_time
        # if any(label.reward > 0.000001 for sublist in S for label in sublist):
        #     return True, neg_cost_counter, cur_time
            
        # Increment counter and check for max attempts
        neg_cost_counter += 1
        if neg_cost_counter >= self.no_profitable_route_iteration_max_tolerance:
            return True, neg_cost_counter, cur_time
            
        # Reset timer and continue
        # print(f'{neg_cost_counter}: No positive reduced cost found! Reset time & continue searching...',
        #       f'states explored: {sum(len(x) for x in S)}')
        print(f'{neg_cost_counter}: No positive reduced cost found! Reset time & continue searching...',
              f'states explored: {sum(len(sublist) for sublist in storage.S.values())}')
        return False, neg_cost_counter, time.time()
    
    def log_state_storage_types(self):    
        all_labels = [label for sublist in self.storage.S.values() for label in sublist]
        counter = Counter(label.dominance_label for label in all_labels)
        reached_cnt = Counter(label.reached_flg for label in all_labels)
        weakly_converted_cnt = Counter(label.type_converted for label in all_labels if label.dominance_label == DominanceLabel.WEAKLY_DOMINANT)
        
        return {
            "total_states": len(all_labels),
            "strongly_dominant": counter[DominanceLabel.STRONGLY_DOMINANT],
            "semistrongly_dominant": counter[DominanceLabel.SEMISTRONGLY_DOMINANT],
            "weakly_dominant": counter[DominanceLabel.WEAKLY_DOMINANT],
            "converted_to_weakly": weakly_converted_cnt[True],
            "initiated_as_weakly": weakly_converted_cnt[False],
            "undefined": counter[DominanceLabel.UNDEFINED],
            "reached": reached_cnt[True],
            "not_reached": reached_cnt[False]
        }
    
    def convert_to_legacy_format(self, S: Dict[int, List[LabelTWModel]]) -> List[List[List[Any]]]:
        legacy_S = [[] for _ in range(self.n + 1)]
        for i, sublist in S.items():
            for label in sublist:
                legacy_label = [
                    label.i, label.acc_demand, label.acc_length, label.stops,
                    label.m, label.reward, label.reached_flg, label.prevN,
                    label.counter, label.dominance_label
                ]
                legacy_S[i].append(legacy_label)
        return legacy_S

    def _check_dominance(self, new_label: LabelTWModel) -> bool:
        """
        Check dominance and return whether the new label is dominated.
        Returns:
            bool: True if the new label is dominated, False otherwise
        """
        if self.domVer == 4:
            return self._filter_out_dominated_states(new_label)
        else:
            raise ValueError("Invalid dominance version.")

    def _check_dominance_ver4(self, state_a: LabelTWModel, state_b: LabelTWModel) -> bool:
        if ( (state_a.acc_demand <= state_b.acc_demand) and
            (state_a.acc_length <= state_b.acc_length) and
            (state_a.acc_duals >= state_b.acc_duals)):
            return True
        return False
    
    def _filter_out_dominated_states(self, new_label: LabelTWModel)-> bool:
        """
        Filter out dominated states and return whether the new label is dominated.
        Returns:
            bool: True if the new label is dominated, False otherwise
        """
        i = new_label.i
        is_dominated = False
        keep_as_weakly_dominant = False
        dominated_by_semi_strongly_prevN_collect = set()
        dominated_by_semi_strongly_labels = []
        dominated_old_labels = []

        # Get the labels at the current node from the storage
        current_labels = self.storage.get_labels(i)

        if len(current_labels) > 0: 
            # for w in range(len(current_labels) - 1, -1, -1):
            for w in range(len(current_labels)):
                current_label = current_labels[w]
                # we only need to check against labels that's not reached
                if current_label.reached_flg:
                    continue

                # Check for dominance of existing label by the new one
                if self._check_dominance_ver4(new_label, current_label):
                    dominated_old_labels.append(current_label)
                # Check for dominance of the new label by an existing one
                elif self._check_dominance_ver4(current_label, new_label):
                    is_dominated = True
                    if current_label.dominance_label in [DominanceLabel.STRONGLY_DOMINANT, DominanceLabel.WEAKLY_DOMINANT]:
                        continue
                    elif current_label.dominance_label == DominanceLabel.SEMISTRONGLY_DOMINANT:
                        if current_label.prevN == new_label.prevN:
                            continue
                        elif current_label.prevN != new_label.prevN:
                            if (new_label.acc_length + self.C[(new_label.i, current_label.prevN)] > self.time_window):
                                continue
                            else: 
                                dominated_by_semi_strongly_prevN_collect.add(current_label.prevN)
                                dominated_by_semi_strongly_labels.append(current_label)
                                keep_as_weakly_dominant = True
        
        if keep_as_weakly_dominant:
            if len(dominated_by_semi_strongly_prevN_collect) >= 2:
                keep_as_weakly_dominant = True
                return True
            else: 
                # add new label as weakly dominant
                if dominated_by_semi_strongly_labels[0].prevN!=0: # prevent reaching to 0
                    new_label.dominance_label = DominanceLabel.WEAKLY_DOMINANT
                    new_label.force_extend.add(dominated_by_semi_strongly_labels[0].prevN)
                    self.storage.insert_label(new_label)
                return False
        
        if not is_dominated:
            self.get_dominant_type(new_label)
            if new_label.dominance_label == DominanceLabel.SEMISTRONGLY_DOMINANT:
                for old_label in dominated_old_labels:
                    if (old_label.prevN != new_label.prevN) and (new_label.prevN != 0):
                        old_label.dominance_label = DominanceLabel.WEAKLY_DOMINANT
                        old_label.type_converted = True
                        old_label.force_extend.add(new_label.prevN)
                    else:
                        self.storage.remove_label(old_label)
                        # old_label.dominated = True
            elif new_label.dominance_label == DominanceLabel.STRONGLY_DOMINANT:
                for old_label in dominated_old_labels:
                    self.storage.remove_label(old_label)
                    # old_label.dominated = True
            else:
                raise Exception("New label must be either strongly or semi-strongly dominant if not dominated.")
            self.storage.insert_label(new_label)
            return False
        return is_dominated
            

    def get_dominant_type(self, label: LabelTWModel):
        # label is not feasible to perform 2-cycle
        if label.acc_length + self.C[(label.i, label.prevN)] > self.time_window:
            label.dominance_label = DominanceLabel.STRONGLY_DOMINANT
        else:
            label.dominance_label = DominanceLabel.SEMISTRONGLY_DOMINANT

    def get_label_priority(self, label: LabelTWModel) -> Tuple[float, float, float]:
        """
        Calculate priority for label processing based on research-backed criteria.
        Returns tuple of (resource_score, reward_efficiency, path_quality) where:
        - resource_score: Combined measure of remaining resources (higher is better)
        - reward_efficiency: Reward per unit of resource consumed (higher is better)
        - path_quality: Measure of path feasibility and efficiency (higher is better)
        
        Note: All values are normalized to [0,1] range for consistent comparison
        """
        # 1. Resource Consumption Score (Irnich and Desaulniers, 2005)
        remaining_time = (self.time_window - label.acc_length)/self.time_window
        resource_score = remaining_time # Bottleneck resource
        
        # 2. Reward Efficiency (Feillet et al., 2004)
        # Use accumulated duals instead of just reward for better column generation
        if label.stops > 0:
            reward_per_stop = label.acc_duals/label.stops
            reward_per_resource = label.acc_duals/(
                label.acc_length/self.time_window
            )
        else:
            reward_per_stop = 0
            reward_per_resource = 0
        
        # Normalize reward scores relative to best possible reward
        max_possible_dual = max(self.dual)
        reward_efficiency = reward_per_stop + reward_per_resource
                            #(reward_per_stop/max_possible_dual + 
                            #reward_per_resource/(2*max_possible_dual))/2
        
        # 3. Path Quality Score (Pugliese and Guerriero, 2013)
        # Consider average utilization of vehicle capacity
        resource_usage = label.acc_demand/(label.m*self.veh_cap)
        
        return (
            resource_score,      # Primary: ensure resource feasibility
            reward_efficiency,   # Secondary: maximize reward efficiency
            resource_usage        # Tertiary: prefer balanced, efficient paths
        )


    def get(self, key, default=None):
        return getattr(self, key, default)
    