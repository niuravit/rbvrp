import json
import os

# Configuration parameters
types = {'a': 2, 'b': 1}  # mapping type name to instance_type number
nodes = [12, 24, 36]
instance_ids = range(1, 11)  # 1 to 10

# Ensure the directory exists
os.makedirs('instance_configs', exist_ok=True)

# Generate config files
for type_name, type_num in types.items():
    for n in nodes:
        for inst_id in instance_ids:
            # Create config data
            config = {
                "no_demand_node": n,
                "instance_type": type_num,
                "distance_metric": "L2norm",
                "instance_id": inst_id
            }
            
            # Generate filename
            filename = f"instance_config_{type_name}{n}-{inst_id}.json"
            filepath = os.path.join('instance_configs', filename)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Created {filename}")

print("\nGenerated all instance config files successfully!")
