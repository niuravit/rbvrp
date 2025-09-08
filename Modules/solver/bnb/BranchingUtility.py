from typing import List, Tuple


class BranchingUtility:
    def __init__(self):
        pass

    @staticmethod
    def parse_branching_conditions(n:int ,_bch_cond: List[Tuple[Tuple[str, str], int]]):
        """Parses branching conditions into dictionaries of forbidden and necessary links."""
        forbid_link_dict = {i: [] for i in range(n + 1)}
        necess_link_dict = {i: [] for i in range(n + 1)}

        if _bch_cond is None: 
            return forbid_link_dict, necess_link_dict
        
        forbid_link = [bh[0] for bh in _bch_cond if (bh[1] == 0)]
        necess_link = [bh[0] for bh in _bch_cond if (bh[1] == 1)]
        
        # Helper to parse node labels from strings like 'c_5' or 'O'
        def parse_node(node_str: str) -> int:
            if node_str == 'O':
                return 0
            return int(node_str.split("_")[-1])
            
        for arc in forbid_link:
            i = parse_node(arc[0]); j = parse_node(arc[1])
            forbid_link_dict[i].append(j)
            
        for arc in necess_link:
            i = parse_node(arc[0]); j = parse_node(arc[1])
            necess_link_dict[i].append(j)
            
            # Propagate necessary links to forbid others
            if i == 0: # Necessary link from depot: other links from depot are forbidden
                for k in range(n + 1):
                    # if k != j and k not in forbid_link_dict[i]:
                    #     forbid_link_dict[i].append(k)
                    if k != i and j not in forbid_link_dict[k]:
                        forbid_link_dict[k].append(j)
            elif j == 0: # Necessary link to depot: other links to depot are forbidden
                for k in range(n + 1):
                    # if k != i and j not in forbid_link_dict[k]:
                    #     forbid_link_dict[k].append(j)
                    if k != j and k not in forbid_link_dict[i]:
                        forbid_link_dict[i].append(k)
            else: # Necessary link between customers
                for k in range(n + 1):
                    if k != i and j not in forbid_link_dict[k]:
                        forbid_link_dict[k].append(j) # No other path to j
                    if k != j and k not in forbid_link_dict[i]:
                        forbid_link_dict[i].append(k) # No other path from i
                        
        return forbid_link_dict, necess_link_dict