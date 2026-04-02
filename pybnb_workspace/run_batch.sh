#!/bin/bash

# Loop through instance IDs 1-10
for id in {1..10}; do
    echo "Running instance $id..."
    # Replace the instance ID in the config filename
    instance_config="instance_config_b24-${id}.json"
    python run.py --instance-config "$instance_config" --experiment-config experiment_config_2h.json --vis-config vis_config1.json
    
    # Add a small delay between runs (optional)
    sleep 2
done
