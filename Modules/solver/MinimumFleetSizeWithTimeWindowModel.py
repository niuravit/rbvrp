from data_model.Instance import Instance
from solver.OptimizationModel import OptimizationModel
from data_model.ExperimentConfig import ExperimentConfig
# from Modules import initialize_path, bnp, pybnb, vis_sol
from initialize_path import InitialRouteGenerator
from visualize_sol import create_nodes,plot_network
from bnp import MinimumFleetSizeWithTimeWindowBnP

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
                            _chDom = True)
        print(f"Finished initializing the problem with {len(problem.rmp_initializer_model.model.getVars())} variables and {len(problem.rmp_initializer_model.model.getConstrs())} constraints.")
        bnb_log_filename=self.solution_directory+'bnb.log'
        print(f"Starting to solve the problem. BnB log will be saved to {bnb_log_filename}")
        results = pybnb.solver.solve(problem, 
                                     log_filename=bnb_log_filename,
                                     comm=None, absolute_gap=1e-6,
                                    time_limit=self.experiment_config.bnp_time_limit)
        obj,route_pats,rmp_model,rmp_init_df,node_count = problem.best_node
        
        final_mip = problem.rmp_initializer_model
        final_mip.init_routes_df = rmp_init_df
        final_mip.model = rmp_model.copy()
        final_mip.shortCuttingColumns()
        final_mip.model.update()
        final_mip.solveModel()        
        
        # instSol['bnpSol'] = twBnP_sol
        return problem, results, self.log_solving_stats(problem, results, final_mip)
        

    def log_solving_stats(self, problem, results, final_mip, plot_solution=True):
        """
        Logs the solving statistics from the problem and results.
        :param problem: The problem instance containing the solution.
        :param results: The results from the solving process.
        :return: A dictionary containing the logged statistics.
        """
        inst_log = {}
        obj,route_pats,rmp_model,rmp_init_df,node_count = problem.best_node
        inst_log.update(self.instance.to_dict_for_logging())
        inst_log.update(self.experiment_config.to_dict())
        inst_log['termCond'] = results.termination_condition
        inst_log['upb'] = obj
        inst_log['lwb'] = results.bound
        inst_log['gap'] = results.absolute_gap
        inst_log['nodesExp'] = results.nodes
        inst_log['wallTime'] = results.wall_time
        inst_log['bnpTimeLim'] = self.experiment_config.bnp_time_limit
        inst_log['remSpace'] = final_mip.calculateAverageRemainingSpace(final_mip.model.getVars())

        if plot_solution:
            twBnP_sol = final_mip.getRouteSolution(final_mip.model.getVars(),
                                             self.vis_config, self.instance.node_trace,
                                              self.instance.customer_demand)
            
            plt_title = f"SOL_IPBnp_{self.instance.instance_id}_{str(round(obj,2))}_{self.instance.distance_metric}"
            plot_network(twBnP_sol,self.instance.node_trace,_display_cus_dem=True,
                _cus_dem=self.instance.customer_demand,_title=plt_title,
                _save_to_file=self.solution_directory+f'SOL_IPBnp_{self.instance.instance_id}.png',
                _display_plot=False,_display_info_table=True,_show_all_info=False)
            
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