# rbvrp: Rate-based Vehicle Routing Problem

> **Paper:** *Rate-based Vehicle Routing Problem for Delivery in Densely Populated Urban Areas*
> Ravit Pichayavet, Alexander M. Stroh, Alejandro Toriello, Alan L. Erera —
> H. Milton Stewart School of Industrial and Systems Engineering, Georgia Institute of Technology

An exact Branch-and-Price (BnP) solver for the **Rate-based Vehicle Routing Problem (rbVRP)**, motivated by high-velocity last-mile delivery operations in megacities (e.g., local hub → micro-hubs or parcel lockers).

## Problem Overview

The setting is a single-depot delivery network where demand arrives at each node as a **continuous fluid at a constant rate** $\lambda_i$. Each vehicle is assigned to a single **fixed cyclical route** that it repeats continuously. Assigning $m_r$ vehicles to route $r$ (of total travel time $l_r$) yields a **headway** of $l_r / m_r$, and the delivery capacity constraint requires:

$$\frac{Q}{l_r / m_r} \geq \lambda_r \quad \forall r$$

The **delivery lead time** of a package is the sum of (1) its waiting time at the depot before pickup and (2) its traveling time en route to its destination. The model minimizes the demand-weighted average delivery lead time subject to a maximum lead time guarantee $t_w$.

The solver handles a **two-phase optimization**:

- **Phase I (Model 1):** Minimize total fleet size $M^*$ subject to a maximum lead time guarantee $t_w$
- **Phase II (Model 2):** Given $M^*$, minimize the demand-weighted average delivery lead time per package

**Solution method:** Branch-and-Bound with Column Generation (Branch-and-Price) at each node.
- **Master Problem (RMP):** Set-covering LP/IP solved with [Gurobi](https://www.gurobi.com/)
- **Pricing Subproblem:** Resource-Constrained Shortest Path Problem (RCSPP) solved via Label-Setting Dynamic Programming with model-specific dominance rules
- **BnB Framework:** [pybnb](https://github.com/ghackebeil/pybnb)

### Key Contributions

1. **Novel rate-based VRP** that jointly optimizes fleet size and route planning in a single framework for high-velocity last-mile delivery, extending prior rate-based shuttle network design to the local-hub-to-delivery-point tier.
2. **Efficient column generation** with new state domination rules derived specifically for the rate-based objective, enabling close-to-optimal solutions by solving only the root node of the B&P tree.
3. **Analytical properties:** proof that (under strict triangle inequality) a finite fleet size drives optimal routes to direct deliveries; and that, given fixed routes, additional vehicles can be assigned greedily to minimize demand-weighted lead time.

---

## Repository Structure

```
rbvrp/
├── pybnb_workspace/            # Main experiment runner and configuration files
│   ├── run.py                  # Entrypoint — run experiments from here
│   ├── experiment_configs/     # JSON configs for solver hyperparameters
│   ├── instance_configs/       # JSON configs specifying which instance to load
│   ├── visualization_configs/  # JSON configs for plot styling
│   ├── generate_instance_configs.py  # Script to batch-generate instance configs
│   ├── solve_lp.py             # Standalone LP relaxation solver (for debugging)
│   └── analyze_results.ipynb  # Jupyter notebook for post-run result analysis
├── Modules/                    # Core Python library
│   ├── ExperimentManager.py    # Orchestrates experiment runs and result logging
│   ├── AlgorithmOrchestrator.py# Selects and runs the appropriate model
│   ├── initialize_path.py      # Initial feasible route generator (warm start)
│   ├── random_instance.py      # Instance generator and loader
│   ├── utility.py              # I/O helpers (JSON, pickle)
│   ├── visualize_sol.py        # Solution plotting with Plotly
│   ├── data_model/             # Dataclasses: Instance, ExperimentConfig
│   └── solver/                 # BnP solver implementation
│       ├── OptimizationModel.py        # Abstract base class
│       ├── MinimumFleetSizeWithTimeWindowModel.py   # Phase I model
│       ├── MinimumAverageTimeWithTimeWindowModel.py # Phase II model
│       ├── bnb/                # Branch-and-Price pybnb problems + branching logic
│       ├── model/              # Gurobi RMP models and route cost calculations
│       └── pricing/            # Label-setting DP pricing subproblem
├── Instances/                  # Demo instances (2–10 customers, pickle format)
├── jobsubmission_scripts/      # SLURM job scripts for HPC (Georgia Tech PACE cluster)
└── rbvrpenv_minimal.yml        # Conda environment specification
```

---

## Requirements

- Python 3.12
- [Gurobi](https://www.gurobi.com/) ≥ 12.0 with a valid license
- [pybnb](https://github.com/ghackebeil/pybnb) == 0.62
- numpy ≥ 1.26, pandas ≥ 2.0, scipy ≥ 1.11, networkx ≥ 3.0, plotly ≥ 5.13

### Setting Up the Environment

```bash
conda env create -f rbvrpenv_minimal.yml
conda activate rbvrpenv
```

---

## Running an Experiment

All experiments are launched from `pybnb_workspace/run.py`.

### Basic Usage

```bash
cd pybnb_workspace
python run.py \
    --instance-config instance_config.json \
    --experiment-config experiment_config_2h.json \
    --vis-config vis_config.json \
    --gurobi-license LOCAL
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--instance-config` | `instance_config.json` | Instance config JSON (in `instance_configs/`) |
| `--experiment-config` | `experiment_config.json` | Experiment config JSON (in `experiment_configs/`) |
| `--vis-config` | `vis_config.json` | Visualization config JSON (in `visualization_configs/`) |
| `--config-dir` | `.` | Directory containing the config subdirectories |
| `--working-dir` | `.` | Working directory root |
| `--gurobi-license` | `LOCAL` | `LOCAL` (uses `~/gurobi.lic`) or `PACE` (GT PACE cluster path) |

### Experiment Config Options

Key fields in an experiment config JSON:

```json
{
  "model": "MinimumFleetSizeWithTimeWindowModel",
  "truck_capacity": 20,
  "truck_speed": 20,
  "time_window": 2,
  "bnp_time_limit": 7200,
  "bnp_node_limit": "Infinity",
  "dp_time_limit": 60,
  "dom_rule": 3
}
```

Available experiment configs:

| File | Description |
|------|-------------|
| `experiment_config_2h.json` | Phase I, 2-hour BnP time limit |
| `experiment_config_5h.json` | Phase I, 5-hour BnP time limit |
| `experiment_config_2h_model2.json` | Phase II, 2-hour BnP time limit |
| `experiment_config_5h_model2.json` | Phase II, 5-hour BnP time limit |
| `experiment_config1.json` | Short test run |

---

## Running Multiple Jobs on HPC (SLURM / GT PACE)

Use the script in `jobsubmission_scripts/rbvrp_jsub_cmd.sh` to submit batches of jobs to the SLURM scheduler.

### Usage

```bash
bash jobsubmission_scripts/rbvrp_jsub_cmd.sh <JSCRIPTDIR> <JOB_LIST_CSV> <WORKING_DIR>
```

| Argument | Description |
|----------|-------------|
| `JSCRIPTDIR` | Subdirectory (relative to `WORKING_DIR`) where `.sbatch` files will be written |
| `JOB_LIST_CSV` | CSV file listing job parameters (one row per job) |
| `WORKING_DIR` | Absolute path to the repository root |

### Job List CSV Format

```
job_name,account,jqueue,mail,node_nbr,core_per_node_nbr,mem_per_core,runtime_limit,instance_config,experiment_config,vis_config
my_job,gt-account,inferno,user@gatech.edu,1,4,8,7200,instance_config_a12_1.json,experiment_config_2h.json,vis_config.json
```

- `runtime_limit` — in seconds; a 30-second buffer is automatically added
- Each job runs `python run.py` inside `pybnb_workspace/` with the `rbvrpenv` conda environment
- Submitted job IDs are recorded in `jobsubmission_scripts/sbatch_submission_records/submitted_batch_<timestamp>.csv`

### Example

```bash
bash jobsubmission_scripts/rbvrp_jsub_cmd.sh \
    jobsubmission_scripts \
    job_list.csv \
    /path/to/CTC_CVRP
```

---

## Data Flow

```
run.py
  └─> ExperimentManager.run_experiment()
        ├─ Resolves instance path → Instances/
        ├─ Loads Instance (.pickle)
        ├─ Loads ExperimentConfig (from JSON)
        └─> AlgorithmOrchestrator.solve_instance()
              ├─ Phase I: MinimumFleetSizeWithTimeWindowModel
              │    ├─ InitialRouteGenerator  (warm-start columns)
              │    ├─ MinimumFleetSizeWithTimeWindowBnP (pybnb.Problem)
              │    │    ├─ bound()  → Column Generation (Gurobi LP + Pricing DP)
              │    │    └─ objective() → Gurobi IP (integer RMP)
              │    └─ pybnb.solver.solve()
              └─ Phase II: MinimumAverageTimeWithTimeWindowModel (if model2)
                   ├─ Solve Phase I at root → get M*
                   └─ MinimumAverageTimeWithTimeWindowBnP (pybnb.Problem)
```

---

## Instance Format

Instances are stored as `.pickle` files with the following dictionary structure:

```python
{
    "distance_matrix":    dict,       # {(node_i, node_j): distance}
    "node_trace":         object,     # Plotly visualization trace
    "customer_demand_df": pd.Series,  # demand per customer node
    "node_position":      dict        # {node_id: (x, y)} coordinates
}
```

Demo instances (2–10 customers) are in `Instances/`. Benchmark instances for experiments follow the naming convention:

```
InstanceType{type}_{n}n_{id}_L2norm.pickle
```

---

## Generating Instance Configs

To generate batch instance config files for experiment types `a`/`b`, sizes 12/24/36, and IDs 1–10:

```bash
cd pybnb_workspace
python generate_instance_configs.py
```

This writes 60 JSON files into `instance_configs/`.
