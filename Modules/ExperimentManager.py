import os
from data_model.Instance import *
from data_model.ExperimentConfig import ExperimentConfig
from AlgorithmOrchestrator import AlgorithmOrchestrator
from typing import List, Any, Dict
from datetime import datetime
import json

class ExperimentManager:
    def __init__(self,
                    experiment_configs: List[Dict[str, Any]],
                    instance_config_dict: Dict[str, Any], 
                    edge_plot_config: Dict[str, Any], 
                    working_root_dir: str
                    ):
        self.experiment_configs_dict = experiment_configs
        self.instance_config_dict = instance_config_dict
        self.edge_plot_config = edge_plot_config
        self.experiment_log = []
        if not os.path.exists(working_root_dir):
            raise FileNotFoundError(f"Working directory {working_root_dir} does not exist.")
        self.working_root_dir = working_root_dir
    
        self.rand_inst = None  # Placeholder for the instance import class, to be initialized later

    def run_experiment(self):
        # Logic to run the experiment based on the configurations
        print(f"Instance Config: {self.instance_config_dict}")
        print(f"Edge Plot Config: {self.edge_plot_config}")
        for i, experiment_config_value in enumerate(self.experiment_configs_dict):
            print("Running experiment with the following configurations:")
            print(f"Experiment: {i} - Config: {experiment_config_value}")
            experiment_config_value = self.resolve_experiment_config(experiment_config_value)
            
            # read instance data
            instance = self.import_instance(self.instance_config_dict)

            # generate experiment unique identifier
            time_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            experiment_id = f"exp_{instance.get_instance_config_join_name()}_{experiment_config_value.model}_{time_id}"
            experiment_config_value.update_experiment_id(experiment_id)
            print(f"Experiment Unique ID: {experiment_id}")

            # save experiment uid for reference
            self.append_experiment_log(
                experiment_id=experiment_id,
                experiment_config=experiment_config_value,
                instance_config=self.instance_config_dict,
                edge_plot_config=self.edge_plot_config.copy(),
                status='started'
            )

            # call the algorithm orchestrator to solve the instance
            algorithm_orchestrator = AlgorithmOrchestrator(
                instance_config=instance,
                experiment_config=experiment_config_value,
                edge_plot_config=self.edge_plot_config,
                working_root_dir=self.working_root_dir
            )
            algorithm_orchestrator.solve_instance()

            # When experiment is done, update the log
            self.update_experiment_log(
                experiment_id=experiment_id,
                status='completed'
            )
            print(f"Experiment {experiment_id} completed successfully.")
            # clean up algorithm orchestrator to free resources
            del algorithm_orchestrator


    def import_instance(self, instance_config):
        inst_dir = self.resolve_instance_dir(instance_config)
        inst_file = self.resolve_instance_file(instance_config,inst_dir)
        print(f"Importing instance from file: {inst_file}")
        return Instance.import_instance(inst_dir, inst_file, instance_config)
    
    def resolve_experiment_config(self, experiment_config):
        return ExperimentConfig.from_dict(experiment_config)

    def resolve_instance_dir(self, instance_config):
        # Logic to resolve the instance directory based on the instance configuration
        inst_type = instance_config.get('instance_type')
        inst_dem_no = instance_config.get('no_demand_node')
        inst_dir = f'{self.working_root_dir}/InstancesForExperiment/L2_norm/TYPE{inst_type}/{inst_dem_no}N/'
        if not os.path.exists(inst_dir):
            raise FileNotFoundError(f"Instance directory {inst_dir} does not exist.")
        # If the directory exists, return it
        inst_dir = os.path.abspath(inst_dir)
        print(f"Resolved instance directory: {inst_dir}")
        return inst_dir
    
    def resolve_instance_file(self, instance_config, inst_dir):
        inst_type = instance_config.get('instance_type')
        inst_dem_no = instance_config.get('no_demand_node')
        inst_id = instance_config.get('instance_id') 
        inst_distance_metric = instance_config.get('distance_metric')
        inst_file = f"/InstanceType{inst_type}_{inst_dem_no}n_{inst_id}_{inst_distance_metric}.pickle"
        if not os.path.exists(inst_dir + inst_file):
            raise FileNotFoundError(f"Instance file {inst_file} does not exist.")
        # If the file exists, return it
        print(f"Resolved instance file: {inst_file}")
        return inst_file


    def append_experiment_log(self, experiment_id, experiment_config, instance_config, edge_plot_config, status):
        # Logic to append the experiment log with the new experiment details
        self.experiment_log.append({
            'experiment_id': experiment_id,
            'experiment_config': experiment_config,
            'instance_config': instance_config,
            'edge_plot_config': edge_plot_config,
            'status': status
        })
        print(f"Experiment {experiment_id} with status: {status}")


    def update_experiment_log(self, experiment_id, status):
        # Logic to update the experiment log with the status
        for log in self.experiment_log:
            if log['experiment_id'] == experiment_id:
                log['status'] = status
                print(f"Updated experiment {experiment_id} status to {status}")
                return
        print(f"Experiment ID {experiment_id} not found in log.")

    def save_experiment_log(self, file_path):
        # Logic to save the experiment log to a file
        with open(file_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=4)
        print(f"Experiment log saved to {file_path}")
        return file_path





