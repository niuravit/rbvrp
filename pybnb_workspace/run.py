MODULE_DIR = '/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/Modules'
GUROBI_LICENSE_DIR = '/Users/ravitpichayavet/gurobi.lic'
MAIN_DIR = '../ComputationalExperiment/'

ARG_COLOR = '#104375'
DEPOT_COLOR = '#D0F6FF'
NODE_COLOR = '#484848'

import sys
sys.path.insert(0,MODULE_DIR)
import importlib
from datetime import datetime

import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
import model as md
import bnp as bnp

import numpy as np
from gurobipy import *
import os
os.environ['GRB_LICENSE_FILE'] = GUROBI_LICENSE_DIR


from itertools import combinations,permutations 
import pandas as pd
import itertools
import plotly.graph_objects as go
import networkx as nx
import plotly.offline as py 
import pickle as pk
import nltk
import time
import copy
import argparse

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import distance
import logging

import pybnb
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

# Step 2: Define all the necessary arguments
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


try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Error parsing arguments: {e}")
    sys.exit(1)

# Step 4: Use the parsed arguments in your script's logic
print(f"Starting experiment with:")
print(f"-- Instance config: {args.instance_config}")
print(f"-- Experiment config: {args.experiment_config}")
print(f"-- Visualization config: {args.vis_config}")

instance_config = util.read_json(
    file_path=DATA_ROOT+f'/instance_configs/{args.instance_config}')
experiment_configs = util.read_json(
    file_path=DATA_ROOT+f'/experiment_configs/{args.experiment_config}')
edge_plot_config = util.read_json(
    file_path=DATA_ROOT+f'/visualization_configs/{args.vis_config}')
exp_manager = ExperimentManager(
    experiment_configs=experiment_configs,
    instance_config_dict=instance_config,
    edge_plot_config=edge_plot_config,
    working_root_dir="/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/ComputationalExperiment")

exp_manager.run_experiment()