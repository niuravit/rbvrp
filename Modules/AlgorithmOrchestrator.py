from data_model.Instance import *
from data_model.ExperimentConfig import *
from solver.MinimumFleetSizeWithTimeWindowModel import MinimumFleetSizeWithTimeWindowModel
import pandas as pd
import os
import json

class AlgorithmOrchestrator:
    def __init__(self, 
                instance_config:Instance, 
                experiment_config: ExperimentConfig,
                edge_plot_config: Dict[str, Any],
                working_root_dir: str):
        self.instance_config = instance_config
        self.experiment_config = experiment_config
        self.edge_plot_config = edge_plot_config
        self.working_root_dir = working_root_dir

        col=[  # Instance Info
            'n','total_dem','npr','mpr','init_cols',
            # Solution Info
            'termCond', 'upb', 'lwb', 'gap', 'nodesExp', 'wallTime','bnpTimeLim',
            'remSpace']

    def solve_instance(self):
        # Implement the logic to run the algorithm with the provided configurations
        inst_log = dict()
        sol_log = dict()
        # _nodeTce['marker']['size'] = _nSize
        inst_log['n'] = len(self.instance_config.node_position)-1
        inst_log['total_dem'] = self.instance_config.customer_demand.sum()
        inst_log['npr'] = self.experiment_config.to_dict()['max_nodes_proute_DP']
        inst_log['mpr'] = self.experiment_config.to_dict()['max_vehicles_proute_DP']

        if (self.experiment_config.model == "MinimumFleetSizeWithTimeWindowModel"):
            self.model = MinimumFleetSizeWithTimeWindowModel(
                instance=self.instance_config,
                experiment_config=self.experiment_config,
                vis_config=self.edge_plot_config,
                solution_directory=self.resolve_solution_directory(self.working_root_dir))
        else:
            raise ValueError(f"Unsupported model type: {self.experiment_config.model}")

        # Solve the instance using the selected model
        problem, results, inst_log = self.model.solve()

        # Record the experiment log
        self.record_solving_results(problem, results, inst_log)    
        
    def record_solving_results(self, problem, results, inst_log):
        """
        Records the solving logs from the problem and results.
        :param problem: The problem instance containing the solution.
        :param results: The results from the solving process.
        :param inst_log: The instance log to be updated.
        """
        result_file = f"{self.experiment_config.experiment_id}_result"
        result_dir = self.resolve_result_directory(self.working_root_dir)
        # Save the DataFrame to a JSON file
        with open(f'{result_dir}/{result_file}.json', 'w') as json_file:
            # Use json.dump() to write the dictionary to the file
            json.dump(inst_log, json_file, indent=4)
        print(f"Results saved to {result_file}/{result_file}.json")


    def resolve_solution_directory(self, root_dir = "Results"):
        # sol_dir = f"/InstancesForExperiment/{self.experiment_config.dp_mode}/Solutions/TYPE{self.instance_config.instance_type}/{self.instance_config.no_demand_node}N/"
        sol_dir = f"/NewResults/{self.experiment_config.experiment_id}/"
        sol_dir = root_dir + sol_dir
        # check if directory exists, if not create it
        if not os.path.exists(sol_dir):
            os.makedirs(sol_dir)
            print(f"Created solution directory: {sol_dir}")
        else:
            print(f"Solution directory already exists: {sol_dir}")  
        return sol_dir
    
    def resolve_result_directory(self, root_dir = "Results"):
        re_dir = f"/NewResults/{self.experiment_config.experiment_id}/"
        re_dir = root_dir + re_dir
        # check if directory exists, if not create it
        if not os.path.exists(re_dir):
            os.makedirs(re_dir)
            print(f"Created result directory: {re_dir}")
        else:
            print(f"Result directory already exists: {re_dir}")  
        return re_dir

    def visualize_results(self):
        # Implement the logic to visualize results based on edge_plot_config
        pass