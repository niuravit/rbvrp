import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class RouteCost:
    """
    A class to handle all route cost calculations and metrics.
    """
    def __init__(self, 
                 customer_demand: Dict[str, float],
                 distance_matrix: Dict[Tuple[str, str], float],
                 constant_dict: Dict[str, Any],
                 customer_index: List[int],
                 arcs_index: List[int],
                 depot: str = 'O'):
        """
        Initialize RouteCost calculator.
        
        Args:
            customer_demand: Dictionary mapping customer IDs to their demands
            distance_matrix: Dictionary mapping arc tuples to distances
            constant_dict: Dictionary containing constants like truck_speed, time_window, etc.
            customer_index: List of indices in route DataFrame corresponding to customers
            arcs_index: List of indices in route DataFrame corresponding to arcs
            depot: Depot node identifier (default 'O')
        """
        self.customer_demand = customer_demand
        self.distance_matrix = distance_matrix
        self.constant_dict = constant_dict
        self.customer_index = customer_index
        self.arcs_index = arcs_index
        self.depot = depot
        
        # Derived constants
        self.truck_speed = constant_dict.get('truck_speed', 1.0)
        self.tw_avg_factor = constant_dict.get('tw_avg_factor', 1.0)

    def calculate_route_metrics(self, route: pd.Series) -> Dict[str, Any]:
        """
        Calculate all metrics for a given route.
        
        Args:
            route: Pandas Series containing route information including:
                  - Customer visits (0/1 values at customer_index positions)
                  - Arc usage (0/1 values at arcs_index positions)
                  - 'lr': route length
                  - 'm': number of vehicles
        
        Returns:
            Dictionary containing all route metrics:
            - l_r: route length
            - m: number of vehicles
            - headway: time between consecutive vehicles
            - dem_waiting: dictionary of waiting times per customer
            - avg_waiting: average waiting time (demand weighted)
            - travel_dem_weighted: total travel time weighted by demand
            - average_total_dem_weighted: total cost including waiting
        """
        # Extract route components
        visiting_nodes = pd.Series(route.iloc[self.customer_index][route>=1].index)
        visiting_arcs = pd.Series(route.iloc[self.arcs_index][route>=1].index)
        
        # Basic route parameters
        lr = route['lr']
        m = route['m']
        headway = lr/m
        
        # Calculate demand and initial waiting times
        route_demand = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        avg_waiting = route_demand * lr/(2*m)
        
        # Initialize tracking variables
        visited_nodes = []
        next_nodes = ['STR']  # STR is start marker
        accumulated_distance = 0
        total_travel_cost = 0
        qr = route_demand  # Remaining demand
        
        # Initialize waiting times with headway
        demand_travel_times = dict(zip(
            visiting_nodes,
            [lr * self.tw_avg_factor/m] * len(visiting_nodes)
        ))
        
        # Process route node by node
        while next_nodes[0] != self.depot:
            if next_nodes[0] == 'STR':
                next_nodes.pop(0)
                current_node = self.depot
            else:
                current_node = next_nodes.pop(0)
                
            if current_node != self.depot:
                visited_nodes.append(current_node)
                
            # Find next arc and node
            outgoing_arcs = visiting_arcs[
                visiting_arcs.apply(lambda x: 
                    x[0] == current_node and x[1] not in visited_nodes
                )
            ].to_list()
            
            outgoing_arc = outgoing_arcs[0]
            next_node = outgoing_arc[1]
            next_nodes.append(next_node)
            
            # Calculate costs
            arc_time = self.distance_matrix[outgoing_arc]/self.truck_speed
            travel_cost = qr * arc_time
            accumulated_distance += arc_time
            total_travel_cost += travel_cost
            
            # Update remaining demand
            if next_node != self.depot:
                qr -= self.customer_demand[next_node]
                demand_travel_times[next_node] += accumulated_distance
        
        pkg_served = route_demand*lr
        pkg_served_per_vehicle = pkg_served/m
        utilization = pkg_served_per_vehicle/self.constant_dict['truck_capacity']

        return {
            "l_r": lr,
            "m": m,
            "headway": headway,
            "cus_demand": dict(zip(visiting_nodes, visiting_nodes.apply(lambda x: self.customer_demand[x]))),
            "dem_served": route_demand,
            "dem_waiting": demand_travel_times,
            "avg_waiting": avg_waiting,
            "travel_dem_weighted": total_travel_cost,
            "average_total_dem_weighted": total_travel_cost + avg_waiting,
            "pkgs_served": pkg_served,
            "pkgs_served_per_vehicle": pkg_served_per_vehicle,
            "utilization": utilization
        }
    
    def get_resource_utilization(self, route: pd.Series) -> Dict[str, float]:
        """
        Calculate resource utilization metrics for a route.
        
        Args:
            route: Route series with required information
            
        Returns:
            Dictionary with utilization metrics:
            - capacity_utilization: % of vehicle capacity used
            - time_window_utilization: % of time window used
        """
        visiting_nodes = pd.Series(route.iloc[self.customer_index][route>=1].index)
        total_demand = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        
        capacity_util = (total_demand * route['lr'])/(route['m'] * self.constant_dict.get('vehicle_capacity', 1.0))
        time_util = route['lr']/self.constant_dict.get('time_window', 1.0)
        
        return {
            "capacity_utilization": capacity_util,
            "time_window_utilization": time_util
        }
    
    def validate_route(self, route: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate route feasibility.
        
        Args:
            route: Route series with required information
            
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []
        visiting_nodes = pd.Series(route.iloc[self.customer_index][route>=1].index)
        visiting_arcs = pd.Series(route.iloc[self.arcs_index][route>=1].index)
        
        # Check connectivity
        if not self._is_route_connected(visiting_arcs):
            violations.append("Route is not connected")
            
        # Check demand feasibility
        total_demand = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        if total_demand * route['lr'] > route['m'] * self.constant_dict.get('vehicle_capacity', float('inf')):
            violations.append("Vehicle capacity exceeded")
            
        # Check time window feasibility
        if route['lr'] > self.constant_dict.get('time_window', float('inf')):
            violations.append("Time window exceeded")
            
        return len(violations) == 0, violations
    
    def _is_route_connected(self, arcs: pd.Series) -> bool:
        """Helper method to check route connectivity."""
        if len(arcs) == 0:
            return False
            
        # Convert arcs to adjacency list
        adj_list = {}
        for arc in arcs:
            if arc[0] not in adj_list:
                adj_list[arc[0]] = []
            adj_list[arc[0]].append(arc[1])
            
        # Check if path exists from depot to all nodes and back
        visited = set()
        def dfs(node):
            visited.add(node)
            if node in adj_list:
                for next_node in adj_list[node]:
                    if next_node not in visited:
                        dfs(next_node)
                        
        dfs(self.depot)
        return len(visited) == len(set([arc[0] for arc in arcs] + [arc[1] for arc in arcs]))