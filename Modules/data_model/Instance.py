import pickle as pk
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import pandas as pd


@dataclass(frozen=True)
class Instance:
    no_demand_node: int
    instance_type: str
    instance_id: int
    distance_metric: str
    distance_matrix: Dict[str, Any]
    node_trace: Dict[str, Any]
    customer_demand: pd.Series
    node_position: Dict[str, Any]

    @classmethod
    def import_instance(cls, _dir: str, _file_name: str, instance_config: Dict[str, Any]):
        print(f"Importing instance: {_file_name}")
        with open(f"{_dir}{_file_name}", 'rb') as f1:
            r_instance = pk.load(f1)
        
        # Instantiate the immutable object using a class method
        return cls(
            no_demand_node=instance_config['no_demand_node'],
            instance_type=instance_config['instance_type'],
            distance_metric=instance_config['distance_metric'],
            instance_id=instance_config['instance_id'],
            distance_matrix=r_instance['distance_matrix'],
            node_trace=r_instance['node_trace'],
            customer_demand=r_instance['customer_demand_df'],
            node_position=r_instance['node_position'],
        )
    def get_instance_config_join_name(self) -> str:
        """
        Generates a unique instance config joined name by joining field names and their values.
        """
        config_dict = asdict(self)
        keys = ['no_demand_node', 'instance_type', 'instance_id', 'distance_metric']
        key_accronym = {"no_demand_node": "ndn", "instance_type": "itype", "instance_id": "id", "distance_metric": "dm"}
        
        # Create a list of "name-value" strings for each field
        name_value_pairs = [f"{key_accronym[key]}-{config_dict[key]}" for key in keys]
        
        # Join the "name-value" pairs with an underscore
        return "_".join(name_value_pairs)

    def to_dict_for_logging(self) -> Dict[str, Any]:
        """
        Converts the Instance dataclass to a dictionary.
        
        Returns:
            A dictionary representation of the instance.
        """
        config_dict = dict()
        config_dict['no_demand_node'] = self.no_demand_node
        config_dict['instance_type'] = self.instance_type
        config_dict['instance_id'] = self.instance_id   
        config_dict['distance_metric'] = self.distance_metric
        config_dict['total_demand'] = self.customer_demand.sum()
        return config_dict


def read_instance_config_json( file_path: str):
    """
    Reads the instance configuration from a JSON file.
    :param file_path: Path to the JSON file containing instance configuration.
    :return: A dictionary containing the instance configuration.
    """    
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
        