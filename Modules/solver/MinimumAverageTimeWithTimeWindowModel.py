from data_model.Instance import Instance
from solver.OptimizationModel import OptimizationModel
from data_model.ExperimentConfig import ExperimentConfig
# from Modules import initialize_path, bnp, pybnb, vis_sol
from initialize_path import InitialRouteGenerator
from visualize_sol import create_nodes,plot_network
from solver.bnb.MinimumFleetSizeWithTimeWindowBnP import *
from solver.bnb.MinimumAverageTimeWithTimeWindowBnP import *
import pybnb
import numpy as np

class MinimumAverageTimeWithTimeWindowModel(OptimizationModel):
    """
    A class to represent the minimum average time spent model for the cTCCVRP.
    Inherits from OptimizationModel.
    """

    def __init__(self, instance: Instance, 
                 experiment_config: ExperimentConfig, 
                 vis_config: dict,
                 solution_directory: str):
        """
        Initializes the MinimumAverageTimeSpentModel with instance configuration and constant dictionary.
        
        :param instance_config: Configuration for the instance.
        :param constant_dict: Dictionary containing constants for the model.
        """
        self.model_type = 'MinimumAverageTimeSpentModel'
        self.instance = instance
        self.experiment_config = experiment_config
        self.vis_config = vis_config
        self.solution_directory = solution_directory    

        self.resolve_instance_vis_configs()   

    def solve(self):
        """
        Solves the minimum average time spent model.
        :return: The solution to the model.
        This model requires total fleet size value M and max time window tw. 
        tw is predefined, while M is solved with minimum fleet with time window model.
        We will only use the root node solution of the minimum fleet with time window model.
        Warm starting the model with the root node columns is optional parameter to config.
        """

        # solve fleet size model with time window to get the total fleet size M: only solve at root node
        fleetsize_problem, fleetsize_results, fleet_size_stat_log = self.solve_fleetsize_model()

        # solve min average time model with two options:
        # 1) Use columns from root node solution to warm start the model
        return self.solve_min_average_time_model(fleetsize_problem, fleet_size_stat_log)

        # 2) Use initial feasible solution generated (might be infeasible still..)

    def solve_min_average_time_model(self, warm_start_prob_inst, fleet_size_stat_log):
        """
        Solves the min average time model.
        """
        constant_dict = self.experiment_config.to_dict()
        constant_dict['total_fleet_size'] = fleet_size_stat_log['root_node']['IPObj']
        # TODO: Implement the solution for the min average time model
        # Call Branch and Price solver & solve!
        min_avg_time_problem = MinimumAverageTimeWithTimeWindowBnP(
                            self.instance.distance_matrix,
                            warm_start_prob_inst.initializer,  
                            constant_dict,
                            self.solution_directory,
                            _chDom = True, 
                            _dom_rule = 4)
        min_avg_time_problem.load_rmp_initial_model(warm_start_prob_inst.initializer, 
                                                    warm_start_prob_inst.root_node[3], 
                                                    warm_start_prob_inst.root_node[4])
        print(f"\n======================================================")
        print(f"Finished initializing the problem with {len(min_avg_time_problem.rmp_initializer_model.model.getVars())} variables and {len(min_avg_time_problem.rmp_initializer_model.model.getConstrs())} constraints.")
        bnb_log_filename=self.solution_directory+'phase2_bnb.log'
        print(f"Starting to solve the problem. BnB log will be saved to {bnb_log_filename}")
        # Attach logger to the problem
        min_avg_time_problem.custom_logger = CustomLogger(bnb_log_filename)
        # Solve with original pybnb solver
        results = pybnb.solver.solve(min_avg_time_problem, 
                                   log_filename=bnb_log_filename,
                                   comm=None, absolute_gap=1e-6,
                                   comparison_tolerance=1e-7,
                                   time_limit=self.experiment_config.bnp_time_limit,
                                   log_interval_seconds=0.1,  # Log as frequently as possible
                                   log_new_incumbent=False)  # Don't wait for new incumbents to log
        # Close the custom logger
        min_avg_time_problem.custom_logger.close()
        solution_stat_log = self.log_solving_stats(min_avg_time_problem, results, sol_phase=2)
        # adding some initial stat from fleet_size_stat_log
        self.update_warm_start_stat_log(fleet_size_stat_log, solution_stat_log)
        return min_avg_time_problem, results, solution_stat_log

    def update_warm_start_stat_log(self, fleet_size_stat_log, min_avg_time_stat_log):
        min_avg_time_stat_log['warm_start_sol'] = fleet_size_stat_log

    def solve_fleetsize_model(self):
        """
        Solves the fleet size model.
        :return: The solution to the fleet size model.
        """
        customer_node_num = len(self.instance.node_position)-1
        labeling_dict = create_nodes(0,customer_node_num)
        docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()
        constant_dict = self.experiment_config.to_dict()
        # Generate initial feasible set R+ for root node
        initializer = InitialRouteGenerator(1,labeling_dict,
                                              self.instance.customer_demand,
                                              constant_dict,
                                              self.instance.distance_matrix)
        row_labels = ['lr','m']+depot+customers+arcs
        
        init_route = initializer.generateInitialRouteWithFleetSize(row_labels, 
                                                                    n=len(labeling_dict['customers']),
                                                                    tw=constant_dict.get('time_window', np.inf),
                                                                    tw_factor=constant_dict.get("tw_factor", 1),
                                                                    init_max_npr=constant_dict.get('init_max_nodes_proute', np.inf),
                                                                    init_max_mpr=np.inf)
        # Call Branch and Price solver & solve!
        # TODO: Need to limit node = 1 as we only need root node solution.
        problem = MinimumFleetSizeWithTimeWindowBnP(
                            self.instance.distance_matrix,
                            initializer, 
                            init_route, 
                            constant_dict,
                            self.solution_directory,

                            _chDom = True)
        print(f"Finished initializing the problem with {len(problem.rmp_initializer_model.model.getVars())} variables and {len(problem.rmp_initializer_model.model.getConstrs())} constraints.")
        bnb_log_filename=self.solution_directory+'phase1_bnb.log'
        print(f"Starting to solve the problem. BnB log will be saved to {bnb_log_filename}")
        # Attach logger to the problem
        problem.custom_logger = CustomLogger(bnb_log_filename)
        # Solve with original pybnb solver
        results = pybnb.solver.solve(problem, 
                                   log_filename=bnb_log_filename,
                                   comm=None, absolute_gap=1e-6,
                                   node_limit=1, # We only solve root node for warm starting
                                   time_limit=self.experiment_config.bnp_time_limit,
                                   log_interval_seconds=0.1,  # Log as frequently as possible
                                   log_new_incumbent=False)  # Don't wait for new incumbents to log
        # Close the custom logger
        problem.custom_logger.close()
        return problem, results, self.log_solving_stats(problem, results, sol_phase=1)
    

    
    def resolve_instance_vis_configs(self):
        if "node_trace" in self.vis_config:
            if "marker_size" in self.vis_config["node_trace"]:
                self.instance.node_trace['marker_size'] = self.vis_config["node_trace"]["marker_size"]

    def get_solution_remaining_space_and_plot_routes(self, bnb_problem, ip_init_routes_df, ip_model, sol_name, sol_phase, plot_solution ):
        dummy_model = bnb_problem.rmp_initializer_model
        dummy_model.init_routes_df = ip_init_routes_df
        dummy_model.model = ip_model

        if plot_solution:
            twBnP_sol = dummy_model.getRouteSolution(ip_model.getVars(),
                                             self.vis_config, self.instance.node_trace,
                                              self.instance.customer_demand)
            if sol_phase == 1:
                plt_title = f"SOL_{sol_name}_{self.instance.instance_id}_{str(round(ip_model.ObjVal,2))}M_{self.instance.distance_metric}"
            elif sol_phase == 2:
                plt_title = f"SOL_{sol_name}_{self.instance.instance_id}_{str(self.get_obj_in_min_per_pkg(ip_model.ObjVal))}min-per-pkg_{self.instance.distance_metric}"
            else: raise Exception("sol_phase must be 1 or 2")
            plot_network(twBnP_sol,self.instance.node_trace,_display_cus_dem=True,
                _cus_dem=self.instance.customer_demand,_title=plt_title,
                _save_to_file=self.solution_directory+f'SOL_{sol_name}_{self.instance.instance_id}.png',
                _display_plot=False,_display_info_table=True,_show_all_info=False)

        return dummy_model.calculateAverageRemainingSpace(ip_model.getVars())
    
    def get_optimal_route_cost(self, bnb_problem, ip_init_routes_df, ip_model):
        dummy_model = bnb_problem.rmp_initializer_model
        dummy_model.init_routes_df = ip_init_routes_df
        dummy_model.model = ip_model
        optimal_route_names = [v.VarName for v in dummy_model.model.getVars() if v.x > 0.98]
        route_cost = {}
        for r_name in optimal_route_names:
            sample_r = ip_init_routes_df.set_index('labels').loc[:][r_name]
            cost_dict = dummy_model.route_cost_calculator.calculate_route_metrics(sample_r)
            route_cost[r_name] = cost_dict
        return route_cost
    
    def get_solution_stat(self, bnb_problem, bnb_node, sol_name, sol_phase, plot_solution = True):
        lp_obj,ip_obj,route_pats,ip_model,ip_init_df,node_count,wall_time = bnb_node
        # node_model = self.solve_final_set_partitioning(bnb_problem, bnb_node)

        sol_stat = {}
        # need to get lp obj and ip obj
        sol_stat['relaxObj'] = np.round(lp_obj, 6)
        sol_stat['IPObj'] = np.round(ip_obj,6)
        if (sol_phase == 2):
            sol_stat['relaxObj_in_min_per_pkg'] = self.get_obj_in_min_per_pkg(lp_obj)
            sol_stat['IPObj_in_min_per_pkg'] = self.get_obj_in_min_per_pkg(ip_obj)
        sol_stat['node'] = node_count
        sol_stat['gap'] = (ip_obj-lp_obj)/lp_obj
        sol_stat['wallTime'] = wall_time 
        # timing 
        sol_stat['remSpace'] = self.get_solution_remaining_space_and_plot_routes(bnb_problem,ip_init_df,ip_model, sol_name, sol_phase, plot_solution)
        sol_stat['optimalRoutes'] = self.get_optimal_route_cost(bnb_problem, ip_init_df, ip_model)
        sol_stat['total_fleet_size'] = ip_model.getConstrByName("fleet_size").RHS if ip_model.getConstrByName("fleet_size") is not None else None
        return sol_stat

    def get_initial_node(self, problem):
        init_rmp_model = problem.rmp_initializer_model
        init_rmp_model.solveRelaxedBoundedModel()
        init_rmp_model.solveModel()

        df_lab = init_rmp_model.init_routes_df.set_index("labels")
        scc_cols = df_lab.loc[df_lab.index.isin(init_rmp_model.arcs),:]
        init_route_pattern = {r_name: {k:v for k,v in r_dict.items() if v>0} for r_name, r_dict in scc_cols.to_dict().items()} 

        return init_rmp_model.relaxedBoundedModel.ObjVal,init_rmp_model.model.ObjVal,init_route_pattern,init_rmp_model.model,init_rmp_model.init_routes_df,0

    def log_solving_stats(self, problem, results, sol_phase, plot_solution=True):
        """
        Logs the solving statistics from the problem and results.
        :param problem: The problem instance containing the solution.
        :param results: The results from the solving process.
        :return: A dictionary containing the logged statistics.
        """
        inst_log = {}
        inst_log['initial_node'] = self.get_solution_stat(problem, problem.init_node, f"{problem.name}_{self.experiment_config.init_max_nodes_proute}stopsInit" if sol_phase==1 else f"{problem.name}_Phase1Init", sol_phase, plot_solution)
        inst_log['root_node'] = self.get_solution_stat(problem, problem.root_node, f"{problem.name}_RootNodeHeu",sol_phase,  plot_solution)
        inst_log['best_node'] = self.get_solution_stat(problem, problem.best_node, f"{problem.name}_BestNodeBnP",sol_phase,  plot_solution)

        inst_log.update(self.instance.to_dict_for_logging())
        inst_log.update(self.experiment_config.to_dict())
        # bnb tree stat
        lp_obj,ip_obj,route_pats,rmp_model,rmp_init_df,node_count,solve_time = problem.best_node
        inst_log['bnb'] = {}
        bnb_log = inst_log['bnb']
        bnb_log['termCond'] = results.termination_condition
        bnb_log['upb'] = np.round(ip_obj,4)
        bnb_log['lwb'] = np.round(results.bound,4)
        bnb_log['sol_phase'] = sol_phase
        if sol_phase==2:
            bnb_log['upb_in_min_per_pkg'] = self.get_obj_in_min_per_pkg(ip_obj)
            bnb_log['lwb_in_min_per_pkg'] = self.get_obj_in_min_per_pkg(results.bound)
        bnb_log['gap'] = results.absolute_gap
        bnb_log['nodesExp'] = results.nodes
        bnb_log['wallTime'] = results.wall_time
        bnb_log['bnpTimeLim'] = self.experiment_config.bnp_time_limit
        return inst_log
    def get_obj_in_min_per_pkg(self, dem_weighted_obj):
        return np.round(dem_weighted_obj * 60 / (self.instance.customer_demand.sum()),2)

# Create a custom logger to capture branching decisions
class CustomLogger:
    def __init__(self, log_filename):
        self.log_file = open(log_filename, 'a')
        
    def log_branch(self, branch_conditions):
        for cond in branch_conditions:
            self.log_file.write(f"  {cond}\n")
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()