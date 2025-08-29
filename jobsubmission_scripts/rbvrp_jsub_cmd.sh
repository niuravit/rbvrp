#!/bin/bash

# job scripts dir arg 1
JSCRIPTDIR=$1

# job list file from cmd arg 2
jListFileName=$2

# working dir
WKDIR=$3

# the file with the run information, e.g., instance file name and other parameters
INPUT="${WKDIR}/${JSCRIPTDIR}/${jListFileName}"
echo $INPUT

# operating mode
MODE="imp"

# buffer time
buffer_time=30

header=true
while IFS=, read -r job_name account jqueue mail node_nbr core_per_node_nbr mem_per_core runtime_limit instance_config experiment_config vis_config ; do
    # Skip the header line
    if [ "$header" = true ]; then
        header=false
        continue
    fi
    
    # Skip empty lines
    if [[ -z "$job_name" ]]; then
        continue
    fi
# Prepare job parameters
job_log_id=${job_name}_${account}_runtime${runtime_limit}_inst${instance_config}_exp${experiment_config}
# Calculate days, hours, minutes, and seconds
days=$((runtime_limit / 86400))
hours=$(((runtime_limit % 86400) / 3600))
minutes=$(( (runtime_limit % 3600) / 60 )+ $buffer_time)
seconds=$((runtime_limit % 60))
# Format the time limit for SLURM
slurmfmt_time_limit="${days}-$(printf "%02d" "$hours"):$(printf "%02d" ${minutes}):$(printf "%02d" "$seconds")"
echo "jname:${job_log_id}, sl_j_tl:${slurmfmt_time_limit}"
echo  "#!/bin/bash
#SBATCH -J "${job_log_id}"
#SBATCH --account="${account}"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="${mail}"
#SBATCH -N "${node_nbr}" --ntasks-per-node="${core_per_node_nbr}" 
#SBATCH --mem-per-cpu="${mem_per_core}"G     
#SBATCH -olog_files/%j_"${job_log_id}"
#SBATCH -q"${jqueue}"
#SBATCH -t${slurmfmt_time_limit}
cd $WKDIR
module load gurobi/11.0.1
conda activate rbvrpenv

python run.py --instance-config $instance_config --experiment-config $experiment_config --vis-config $vis_config" > ${WKDIR}/${JSCRIPTDIR}/${job_log_id}.sbatch

# Submit the job
sbatch ${WKDIR}/${JSCRIPTDIR}/${job_log_id}.sbatch

# Optionally remove the temporary script file
# rm ${WKDIR}/${JSCRIPTDIR}/${job_log_id}.sbatch

done < "$INPUT"


