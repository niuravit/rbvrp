import pickle as pk
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import numpy as np

@dataclass(frozen=True)
class ExperimentConfig:
    """
    An immutable class to hold experiment configuration settings.
    """
    dom_rule: str
    truck_capacity: int
    fixed_setup_time: int
    truck_speed: float
    max_vehicles: int
    max_nodes_proute_DP: int
    max_vehicles_proute_DP: int
    init_max_nodes_proute: int
    dp_mode: str
    dp_time_limit: float
    time_window: int
    tw_avg_factor: float
    bnp_node_limit: int
    bnp_time_limit: int
    model: str
    experiment_id: str = field(default_factory=lambda: "undefined_experiment_id")

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Creates an ExperimentConfig instance from a dictionary.
        """
        if 'bnp_node_limit' not in config_dict:
            config_dict['bnp_node_limit'] = None  # default value means no limit
        else:
            if (config_dict['bnp_node_limit'] == np.inf):
                config_dict['bnp_node_limit'] = None # default value means no limit
        return cls(**config_dict)

    def get_experiment_name(self) -> str:
        """
        Generates a unique experiment name by joining field names and their values.
        """
        config_dict = asdict(self)
        
        # Create a list of "name-value" strings for each field
        name_value_pairs = [f"{key}-{value}" for key, value in config_dict.items()]
        
        # Join the "name-value" pairs with an underscore
        return "_".join(name_value_pairs)

    def update_experiment_id(self, new_id: str):
        """
        Updates the experiment ID.
        """
        object.__setattr__(self, 'experiment_id', new_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Instance dataclass to a dictionary.
        
        Returns:
            A dictionary representation of the instance.
        """
        return asdict(self)
    

def read_experiment_configs_json( file_path: str):
    """
    Reads the instance configuration from a JSON file.
    :param file_path: Path to the JSON file containing instance configuration.
    :return: A dictionary containing the instance configuration.
    """    
    import json
    with open(file_path, 'r') as f:
        return json.load(f)