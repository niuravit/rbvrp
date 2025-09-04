# MODULE_DIR = '/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/Modules'
MODULE_DIR = '../Modules'
# GUROBI_LICENSE_DIR = '/Users/ravitpichayavet/gurobi.lic'
# PACE GUROBI = '/usr/local/pace-apps/manual/packages/gurobi/11.0.1/license/gurobi.lic'
MAIN_DIR = '../ComputationalExperiment/'

ARG_COLOR = '#104375'
DEPOT_COLOR = '#D0F6FF'
NODE_COLOR = '#484848'

import os
print(os.getcwd())

import sys
sys.path.insert(0,MODULE_DIR)
sys.path.insert(0,"./Modules")
import utility as util
from gurobipy import *
import argparse

from copy import deepcopy
from math import ceil

from data_model.Instance import *
from data_model.ExperimentConfig import *
from ExperimentManager import ExperimentManager

DATA_ROOT = "/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/pybnb_workspace"

# Step 1: Create the parser
parser = argparse.ArgumentParser(
    description="A tool to run optimization experiments with configurable settings.",
    formatter_class=argparse.RawTextHelpFormatter
)
# Step 2: Define config directory
parser.add_argument(
    "--config-dir",
    type=str,
    default=".",
    help="Path to the configuration directory. Defaults to the current directory."
)

parser.add_argument(
    "--working-dir",
    type=str,
    default=".",
    help="Path to the working directory. Defaults to the current directory."
)

# Step 3: Define all the necessary arguments
parser.add_argument(
    "--instance-config",
    type=str,
    default="instance_config.json",
    help="Path to the instance configuration JSON file. Defaults to 'instance_config.json'."
)

parser.add_argument(
    "--experiment-config",
    type=str,
    default="experiment_config.json",
    help="Path to the experiment configuration JSON file. Defaults to 'experiment_config.json'."
)

parser.add_argument(
    "--vis-config",
    type=str,
    default="vis_config.json",
    help="Path to the visualization configuration JSON file. Defaults to 'vis_config.json'."
)

parser.add_argument(
    "--gurobi-license",
    type=str,
    default="LOCAL",
    help="Path to the Gurobi license file. Defaults to 'gurobi.lic'."
)
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Error parsing arguments: {e}")
    sys.exit(1)

if (args.gurobi_license == "LOCAL"):
    os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
elif (args.gurobi_license == "PACE"):
    os.environ['GRB_LICENSE_FILE'] = '/usr/local/pace-apps/manual/packages/gurobi/11.0.1/license/gurobi.lic'


# Step 4: Use the parsed arguments in your script's logic
print(f"Starting experiment with:")
print(f"-- Config directory: {args.config_dir}")
print(f"-- Working directory: {args.working_dir}")
print(f"-- Instance config: {args.instance_config}")
print(f"-- Experiment config: {args.experiment_config}")
print(f"-- Visualization config: {args.vis_config}")
print(f"-- Gurobi license: {args.gurobi_license}")

instance_config = util.read_json(
    file_path=f'{args.config_dir}/instance_configs/{args.instance_config}')
experiment_configs = util.read_json(
    file_path=f'{args.config_dir}/experiment_configs/{args.experiment_config}')
edge_plot_config = util.read_json(
    file_path=f'{args.config_dir}/visualization_configs/{args.vis_config}')
exp_manager = ExperimentManager(
    experiment_configs=experiment_configs,
    instance_config_dict=instance_config,
    edge_plot_config=edge_plot_config,
    working_root_dir=args.working_dir)

exp_manager.run_experiment()