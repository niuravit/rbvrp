from data_model.Instance import Instance
from solver.OptimizationModel import OptimizationModel
from data_model.ExperimentConfig import ExperimentConfig
# from Modules import initialize_path, bnp, pybnb, vis_sol
from initialize_path import InitialRouteGenerator
from visualize_sol import create_nodes,plot_network
from solver.bnb.MinimumFleetSizeWithTimeWindowBnP import *
import pybnb

class MinimumFleetSizeWithTimeWindowModel(OptimizationModel):
    """
    A class to represent the minimum fleet size model for the cTCCVRP.
    Inherits from OptimizationModel.
    """

    def __init__(self, instance: Instance, 
                 experiment_config: ExperimentConfig, 
                 vis_config: dict,
                 solution_directory: str):
        """
        Initializes the MinimumFleetSizeModel with instance configuration and constant dictionary.
        
        :param instance_config: Configuration for the instance.
        :param constant_dict: Dictionary containing constants for the model.
        """
        self.model_type = 'MinimumFleetSizeWithTimeWindowModel'
        self.instance = instance
        self.experiment_config = experiment_config
        self.vis_config = vis_config
        self.solution_directory = solution_directory    

        self.resolve_instance_vis_configs()   
    
    def solve(self):
        """
        Solves the minimum fleet size model.
        :return: The solution to the model.
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
        init_route = initializer.generateInitDFV4wTimeWindow(row_labels, constant_dict)
            
        # Call Branch and Price solver & solve!
        problem = MinimumFleetSizeWithTimeWindowBnP(
                            self.instance.distance_matrix,
                            initializer, 
                            init_route, 
                            constant_dict,
                            self.solution_directory,
                            _chDom = True)
        print(f"Finished initializing the problem with {len(problem.rmp_initializer_model.model.getVars())} variables and {len(problem.rmp_initializer_model.model.getConstrs())} constraints.")
        bnb_log_filename=self.solution_directory+'bnb.log'
        print(f"Starting to solve the problem. BnB log will be saved to {bnb_log_filename}")
        # Attach logger to the problem
        problem.custom_logger = CustomLogger(bnb_log_filename)
        # Solve with original pybnb solver
        results = pybnb.solver.solve(problem, 
                                   log_filename=bnb_log_filename,
                                   comm=None, absolute_gap=1e-6,
                                   time_limit=self.experiment_config.bnp_time_limit,
                                   log_interval_seconds=0.1,  # Log as frequently as possible
                                   log_new_incumbent=False)  # Don't wait for new incumbents to log
        # Close the custom logger
        problem.custom_logger.close()
        return problem, results, self.log_solving_stats(problem, results)
    
    def resolve_instance_vis_configs(self):
        if "node_trace" in self.vis_config:
            if "marker_size" in self.vis_config["node_trace"]:
                self.instance.node_trace['marker_size'] = self.vis_config["node_trace"]["marker_size"]

    def get_solution_remaining_space_and_plot_routes(self, bnb_problem, ip_init_routes_df, ip_model, sol_name, plot_solution ):
        dummy_model = bnb_problem.rmp_initializer_model
        dummy_model.init_routes_df = ip_init_routes_df
        dummy_model.model = ip_model

        if plot_solution:
            twBnP_sol = dummy_model.getRouteSolution(ip_model.getVars(),
                                             self.vis_config, self.instance.node_trace,
                                              self.instance.customer_demand)
            
            plt_title = f"SOL_{sol_name}_{self.instance.instance_id}_{str(round(ip_model.ObjVal,2))}_{self.instance.distance_metric}"
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
    
    def get_solution_stat(self, bnb_problem, bnb_node, sol_name, plot_solution = True):
        lp_obj,ip_obj,route_pats,ip_model,ip_init_df,node_count,wall_time = bnb_node
        # node_model = self.solve_final_set_partitioning(bnb_problem, bnb_node)

        sol_stat = {}
        # need to get lp obj and ip obj
        sol_stat['relaxObj'] = np.round(lp_obj, 6)
        sol_stat['IPObj'] = np.round(ip_obj,2)
        sol_stat['node'] = node_count
        sol_stat['gap'] = (ip_obj-lp_obj)/lp_obj
        sol_stat['wallTime'] = wall_time
        # timing 
        sol_stat['remSpace'] = self.get_solution_remaining_space_and_plot_routes(bnb_problem,ip_init_df,ip_model, sol_name,  plot_solution)
        sol_stat['optimalRoutes'] = self.get_optimal_route_cost(bnb_problem, ip_init_df, ip_model)
        return sol_stat

    def get_initial_node(self, problem):
        init_rmp_model = problem.rmp_initializer_model
        init_rmp_model.solveRelaxedBoundedModel()
        init_rmp_model.solveModel()

        df_lab = init_rmp_model.init_routes_df.set_index("labels")
        scc_cols = df_lab.loc[df_lab.index.isin(init_rmp_model.arcs),:]
        init_route_pattern = {r_name: {k:v for k,v in r_dict.items() if v>0} for r_name, r_dict in scc_cols.to_dict().items()} 

        return init_rmp_model.relaxedBoundedModel.ObjVal,init_rmp_model.model.ObjVal,init_route_pattern,init_rmp_model.model,init_rmp_model.init_routes_df,0

    def log_solving_stats(self, problem, results, plot_solution=True):
        """
        Logs the solving statistics from the problem and results.
        :param problem: The problem instance containing the solution.
        :param results: The results from the solving process.
        :return: A dictionary containing the logged statistics.
        """
        inst_log = {}
        inst_log['initial_node'] = self.get_solution_stat(problem, problem.init_node, f"{self.experiment_config.init_max_nodes_proute}stopsInit", plot_solution)
        inst_log['root_node'] = self.get_solution_stat(problem, problem.root_node, "RootNodeHeu", plot_solution)
        inst_log['best_node'] = self.get_solution_stat(problem, problem.best_node, "BestNodeBnP", plot_solution)

        inst_log.update(self.instance.to_dict_for_logging())
        inst_log.update(self.experiment_config.to_dict())
        # bnb tree stat
        lp_obj,ip_obj,route_pats,rmp_model,rmp_init_df,node_count,solve_time = problem.best_node
        inst_log['bnb'] = {}
        bnb_log = inst_log['bnb']
        bnb_log['termCond'] = results.termination_condition
        bnb_log['upb'] = np.round(ip_obj,2)
        bnb_log['lwb'] = np.round(results.bound,6)
        bnb_log['gap'] = results.absolute_gap
        bnb_log['nodesExp'] = results.nodes
        bnb_log['wallTime'] = results.wall_time
        bnb_log['bnpTimeLim'] = self.experiment_config.bnp_time_limit
        return inst_log

#         vis_sol.plot_network(instSol['bnpSol'],_nodeTce,_display_cus_dem=True,
#                 _cus_dem=_cusDem,_title=_plt_title,
#                 _save_to_file=_sol_dir+'SOL_IPBnp_{0}.png'.format(inst),
#                 _display_plot=False,_display_info_table=True,_show_all_info=False)
# #         try:
# #             _plt_title = 'SOL_{0}stops_{1}_{2}'.format(_init_mstops,inst,str(round(instLog['%s-stopsBnp'%_init_mstops],2)))
# #             vis_sol.plot_network(instSol['bnpSol'],_nodeTce,_display_cus_dem=True,
# #                                  _cus_dem=_cusDem,_title=_plt_title,_save_to_file=_sol_dir+'SOL_{0}stopsBnp_{1}.png'.format(_init_mstops,inst),
# #                                  _display_plot=False,_display_info_table=True,_show_all_info=False)
# #         except:
# #             print('Error occurred while trying to save img')
        pass

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