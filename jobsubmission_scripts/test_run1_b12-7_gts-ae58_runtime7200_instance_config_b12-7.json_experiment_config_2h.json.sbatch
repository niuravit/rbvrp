#!/bin/bash
#SBATCH -J test_run1_b12-7_gts-ae58_runtime7200_instance_config_b12-7.json_experiment_config_2h.json
#SBATCH --account=gts-ae58
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rpichayavet3@gatech.edu
#SBATCH -N 1 --ntasks-per-node=24 
#SBATCH --mem-per-cpu=4G     
#SBATCH -olog_files/%j_test_run1_b12-7_gts-ae58_runtime7200_instance_config_b12-7.json_experiment_config_2h.json
#SBATCH -qinferno
#SBATCH -t0-02:30:00
cd ./../pybnb_workspace
module load gurobi/11.0.1
conda init
conda activate rbvrpenv

python run.py --instance-config instance_config_b12-7.json --experiment-config experiment_config_2h.json --vis-config vis_config1.json
 --working-dir ./.. --config-dir ./
