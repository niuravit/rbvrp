import sys
sys.path.insert(0,'/Users/ravitpichayavet/Documents/GaTechOR/GraduateResearch/CTC_CVRP/Modules')
import Modules.ignored_files.visualize_sol as vis_sol
import Modules.ignored_files.initialize_path as init_path
import Modules.ignored_files.random_instance as rand_inst
import Modules.ignored_files.utility as util
# import branch_and_price as bnp
import pandas as pd
import time
import numpy as np
from gurobipy import *
# from gurobipy import Column, Constr, Var
from operator import itemgetter
from solver.pricing.PrizeCollectingDPwTW import PrizeCollectingDPwTW
from solver.pricing.PrizeCollectingDPwTWNewStorage import PrizeCollectingDPwTWNewStorage
import os
# os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
epsilon = 1e-5
from typing import Dict, List, Any, Tuple, Optional
from solver.model.RouteCost import RouteCost
from solver.bnb.BranchingUtility import BranchingUtility


class timeWindowModel:
    def __init__(self, _init_route, _initializer,
                 _distance_matrix,constant_dict,
                 extra_constr=None, _model_name = "timeWindowModel", _mode=None,_relax_route=False):
        
        self.init_route = _init_route.copy()
        self.route_coeff = _init_route['PathCoeff'].values
        self.init_routes_df = _initializer.init_routes_df.copy()
        self.relax_route_flag = _relax_route
        
        self.depot = _initializer.depot
        self.depot_s = _initializer.depot_s
        self.depot_t = _initializer.depot_t
        self.all_depot = _initializer.all_depot
        self.customers = _initializer.customers
        self.n = len(self.customers)
        self.nodes = _initializer.nodes
        self.arcs = _initializer.arcs
        self.route_cost = []
        
        self.coeff_series = _initializer.init_routes_df['labels']
        self.depot_index = self.coeff_series[self.coeff_series.isin(self.all_depot)].index.values
        self.depot_s_index = self.coeff_series[self.coeff_series.isin(self.depot_s)].index.values
        self.depot_t_index = self.coeff_series[self.coeff_series.isin(self.depot_t)].index.values
        self.customer_index = self.coeff_series[self.coeff_series.isin(self.customers)].index.values
        self.nodes_index = self.coeff_series[self.coeff_series.isin(self.nodes)].index.values
        self.arcs_index = self.coeff_series[self.coeff_series.isin(self.arcs)].index.values
        self.veh_no_index = self.coeff_series.loc[self.coeff_series=='m'].index.values
        self.lr_index = self.coeff_series.loc[self.coeff_series=='lr'].index.values
    
        self.constant_dict = constant_dict.copy()
        self.vehicle_capacity = self.constant_dict['truck_capacity']
        self.fixed_setup_time = self.constant_dict['fixed_setup_time']
        self.truck_speed = self.constant_dict['truck_speed']
        self.distance_matrix = _distance_matrix
        self.customer_demand = _initializer.customer_demand
        self.max_vehicles = self.constant_dict['max_vehicles']
        self.max_nodes_proute_DP = self.constant_dict['max_nodes_proute_DP']
        
        self.route_index = pd.Series(self.init_route.index).index.values
        self.model = Model(_model_name)
        if _mode is None: self.mode='multiObjective'
        else: self.mode=_mode
        
        self.cost_matrix = dict()
        for k,v in self.distance_matrix.items():
            if k[0].split('_')[-1] == "O":
                i = 0
            else: 
                i = int(k[0].split('_')[-1])
            if k[1].split('_')[-1] == "O":
                j = 0
            else:
                j = int(k[1].split('_')[-1])
            nk = (i,j)
            self.cost_matrix[nk] = v/self.truck_speed
        
        self.DPRouteDict=dict()
        self.forbid_link_dict = dict()
        self.necess_link_dict = dict()
        self.route_cost_calculator = RouteCost(
            customer_demand=self.customer_demand,
            distance_matrix=self.distance_matrix,
            constant_dict=self.constant_dict,
            customer_index=self.customer_index,
            arcs_index=self.arcs_index
        )
        self.branching_utility = BranchingUtility()
            
    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
#         self.generateCostOfRoutes()
        self.generateObjective()
        self.model.update()
        
    def generateVariables(self):
        self.route = self.model.addVars(self.route_index, lb=0,
                                       vtype=GRB.BINARY, name='route')
        print('Finish generating variables!')
        
    def generateConstraints(self):  
        if self.relax_route_flag:
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) >= int(1) \
                                 for i in self.customer_index )
        else: 
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) == int(1) \
                                 for i in self.customer_index )
        
        self.model.addConstrs( const1,name='customer_coverage' )
        print('Finish generating constrains!')

    def convertToSetPartitioning(self):
        """
        Converts the model's covering constraints (>=1) to set partitioning constraints (==1).
        This method should be called before solving if set partitioning formulation is desired.
        """
        model = self.model
        # Iterate through all constraints in the model
        for constr in model.getConstrs():
            # Check if this is a covering constraint for customers
            # Typically these constraints have names starting with 'customer_' or similar
            if constr.getAttr('Sense') == '>=' and constr.getAttr('RHS') == 1.0:
                # Convert to equality constraint
                constr.setAttr('Sense', '=')
                
        # Update the model
        model.update()
        print("Model converted to set partitioning formulation.")
        
#     def generateCostOfRoutes(self):
#         t1=time.time()
#         self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.calculateCostOfRoute(col),axis=0)
#         print('Finish generating cost vector!....Elapsed-time:',time.time()-t1)
#     def calculateCostOfRoute(self, route):
#         visiting_nodes = pd.Series(route.iloc[self.customer_index][route>=1].index)
#         visiting_arcs = pd.Series(route.iloc[self.arcs_index][route>=1].index)
#         next_node = ['STR']
#         route_cost = 0
#         qr = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
#         avg_waiting = qr*route['lr']/(2*route['m'])
#         visited_node = []
#         # this is initialized with headway
#         demand_travel_time_dict = dict(zip(visiting_nodes,[route['lr']*self.constant_dict['tw_avg_factor']/route['m']]*len(visiting_nodes)))
#         acc_distance = 0
#         while next_node[0]!=self.depot[0]:
#             if next_node[0] == 'STR': next_node.pop(0);selecting_node = self.depot[0] #only for the first arc
#             else: selecting_node = next_node.pop(0)
# #             print(selecting_node)
# #             print(visited_node)
#             outgoing_arc_list = visiting_arcs[visiting_arcs.apply(lambda x: ((x[0]==selecting_node) and (x[1] not in visited_node) ))].to_list()
#             if (selecting_node != self.depot[0]): visited_node.append(selecting_node)
# #             print("outgoing",outgoing_arc_list)
#             outgoing_arc = outgoing_arc_list[0]
#             node_j = outgoing_arc[1]
#             next_node.append(node_j)
#             qj = self.customer_demand[node_j]
#             traveling_time_carrying_pkg = qr*(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
#             acc_distance +=(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
#             route_cost+=traveling_time_carrying_pkg
#             qr = qr-qj
#             if node_j!=self.depot[0]:
#                 demand_travel_time_dict[node_j] += acc_distance
# #         print(avg_waiting,route_cost)
# #         route_cost += avg_waiting
# #         print(route_cost)
#         cost_dict = {
#             "l_r":route['lr'],
#             "m":route['m'],
#             'headway':route['lr']/route['m'],
#             'dem_waiting': demand_travel_time_dict, # headway + travel time
#             'avg_waiting': avg_waiting, # avg waiting time demand weighted
#             'travel_dem_weighted': route_cost, # travel time demand weighted
#             'average_total_dem_weighted': route_cost + avg_waiting
#         }
#         return cost_dict
        
    def generateObjective(self):
        # Minimize the total vehicles
        self.model.setObjective( quicksum(self.route[rt]*((self.route_coeff[rt][self.veh_no_index[0]])) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        print('Finish generating objective!')
        
    def solveModel(self, timeLimit = None,GAP=None):
        if timeLimit is not None: self.model.setParam('TImeLimit', timeLimit)
        if GAP is not None: self.model.setParam('MIPGap',GAP)
        self.model.setParam('SolutionNumber',2)
        self.model.setParam(GRB.Param.PoolSearchMode, 2)
        self.model.optimize()
        
    ##RELAXATION
    def solveRelaxedModel(self):
        #Relax integer variables to continous variables
        self.relaxedModel = self.model.relax()
        var_ss = pd.Series(self.relaxedModel.getVars())
        var_ss.apply(lambda x: x.setAttr('ub',GRB.INFINITY))
        self.relaxedModel.optimize()
        
    def solveRelaxedBoundedModel(self):
        #Relax integer variables to continous variables <=1
        self.relaxedBoundedModel = self.model.relax()
        var_ss = pd.Series(self.relaxedBoundedModel.getVars())
        var_ss.apply(lambda x: x.setAttr('ub',1))
        self.relaxedBoundedModel.optimize()
        
    def getRelaxSolution(self):
        a = pd.Series(self.relaxedModel.getAttr('X'))
        return a[a>0]

    def getDuals(self):
        return self.relaxedModel.getAttr('Pi',self.relaxedModel.getConstrs())
    
    ##COLUMNS GENERATION
    def addColumn(self,_col_object,_col_cost,_name):
        self.model.addVar(lb=0,vtype=GRB.BINARY,column=_col_object, obj= _col_cost, name=_name)
        self.model.update()
    
    def generateColumns(self,_filtered_df,_duals, ):
         # Create a dictionary to hold the new columns
        new_columns_data = {}
        for index, row in _filtered_df.iterrows():
            _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list()
            newColumn = Column(_col, self.model.getConstrs())
            _name = row.colDF.columns[-1]
            self.addColumn(newColumn,row.routeCost,_name)
            # self.init_routes_df[_name] = row.colDF.iloc[:,-1]
            column_data = row.colDF.iloc[:,-1]
            new_columns_data[_name] = column_data

        # This is a single, vectorized operation that avoids fragmentation.
        new_columns_df = pd.DataFrame(new_columns_data)
        # Concatenate the new columns to the existing DataFrame in one go.
        self.init_routes_df = pd.concat([self.init_routes_df, new_columns_df], axis=1)

    def shortCuttingColumns(self,_var_keywords = 'DP', forbidden_arcs = None):
        _cols = self.init_routes_df.set_index('labels')
        DP_cols = _cols.loc[:,_cols.columns.str.contains(_var_keywords)].copy()
        count = 0 
        sc_col_count = 0
        t1 = time.time()
        for r_name, col in DP_cols.items():
            if ((count/len(DP_cols.columns))*100 % 10)==0:
                print('shortcutting columns:',count,'/',len(DP_cols.columns))
            if (col[(col>=2)&(col.index.isin(self.nodes))].size >=1):
#                 print(r_name)
                node_seq = self.DPRouteDict[r_name]
                sct_seq = [self.depot[0]]; visited_seq = [self.depot[0]]
                for n in node_seq:
                    if n not in visited_seq: 
                        sct_seq.append(n)
                        visited_seq.append(n)
                sct_seq.append(self.depot[0])  
                arc_route = [(sct_seq[i],sct_seq[i+1]) for i in range(len(sct_seq)-1)]
#                 print(node_seq,sct_seq)
                qr = sum([self.customer_demand[c] for c in sct_seq])
                lr = self.calculateLr(arc_route)
                m_opt_recal = self.calculate_opt_route_fleet_size(lr,qr,visited_seq[-1])
                _route_coef = np.array(sct_seq+arc_route,dtype=object)
                sc_col = pd.Series(index = _cols.index,data=0.0,name=r_name)
                sc_col.loc[_route_coef]+=1
                sc_col.loc['m'] = m_opt_recal
                sc_col.loc['lr'] = lr
                #Update DF: Use sc_col for updating init_routes_df
                r_var = self.model.getVarByName(r_name)
                self.update_variable_coefficients(sc_col, r_var, m_opt_recal, forbidden_arcs)
                self.model.update()
                self.init_routes_df.loc[:,r_name] = sc_col.values
                sc_col_count+=1
            count+=1
        self.shortcutCols = sc_col_count
        self.shortcutColsPc = len(DP_cols.columns)
        self.shortcutColsTe = time.time()-t1
    
    def calculate_opt_route_fleet_size(self, lr, qr, last_customer):
        travel_length_to_last_customer = lr - self.cost_matrix[(int(last_customer.split("_")[-1]), 0)]
        m_ctc = np.ceil((qr * (lr )) / self.vehicle_capacity)
        m_tw = np.ceil((lr ) / (self.constant_dict['time_window'] - travel_length_to_last_customer))
        return max(m_ctc, m_tw)

    def update_variable_coefficients(self, sc_col, r_var: Var, m_opt_recal: int, _forbidden_arcs = None):
        """
        Updates the coefficients of a variable in the model using a Gurobi Column object.

        Args:
            sc_col: A pandas Series or similar object containing the new coefficients.
            r_var: The Gurobi variable whose coefficients you want to change.
        """
        # Update the objective coefficient of the existing variable
        var_obj = 1e10 if self.col_contains_forbidden_arc(sc_col,self.convert_forbidden_links(_forbidden_arcs)) else m_opt_recal
        r_var.obj = var_obj

        # 1. Get the list of all constraints in the model
        all_constrs = self.model.getConstrs()

        # 2. Prepare the list of tuples for bulk update
        # The format is (constraint, variable, new_coefficient)
        updates = []
        for const_idx,constr in enumerate(all_constrs):
            new_coeff = sc_col.loc["c_%s"%(const_idx+1)]
            self.model.chgCoeff(constr,r_var,new_coeff)
            # up_list.append(["c_%s"%(const_idx+1),new_coeff])
        

    def col_contains_forbidden_arc(self, sc_col,_forbidden_arcs) -> bool:
        """
        Checks if the column (route) contains any forbidden arcs.

        Args:
            sc_col: A pandas Series representing the column data.
            _forbidden_arcs: A list of forbidden arcs, e.g., [('O', 'c_1'), ('c_2', 'c_3')].

        Returns:
            True if a forbidden arc is found, otherwise False.
        """
        if not _forbidden_arcs:
            return False
        
        # Iterate through the forbidden arcs
        for arc in _forbidden_arcs:
            # Check if the arc is present in the column's index.
            # This assumes your sc_col index contains the arc tuples as labels.
            if arc in sc_col.index:
                # Check if the coefficient for that arc is non-zero (i.e., it's used in the route)
                if sc_col[arc] > 0:
                    return True
                    
        return False

    def convert_forbidden_links(self,forbidden_link_dict: Dict[int, List[int]]) -> List[Tuple[str, str]]:
        """
        Converts a forbidden links dictionary to a list of arc tuples.
        
        Args:
            forbidden_link_dict: A dictionary where keys are from-nodes (0 for depot)
                                and values are a list of to-nodes.
        
        Returns:
            A list of arc tuples, e.g., [('O', 'c_11'), ('O', 'c_1')].
        """
        if forbidden_link_dict is None:
            return None
        forbidden_arcs = []
        for from_node, to_nodes in forbidden_link_dict.items():
            from_label = 'O' if from_node == 0 else f'c_{from_node}'
            for to_node in to_nodes:
                to_label = 'O' if to_node == 0 else f'c_{to_node}'
                forbidden_arcs.append((from_label, to_label))
        return forbidden_arcs
                        
    def calculateLr(self, route_arcs):
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
#         print(route_arcs,lr)
        return lr

    def runColumnsGeneration(self,_m_collections,
                             _pricing_status=False,
                               _check_dominance=True,
                               _dominance_rule=None,
                               _DP_ver=None,
                               _update_m_ub=False, _time_limit=None,_filtering_mode=None,_heu_add_m=False, _bch_cond=None,_node_count_lab=None):
        outer_dict = dict(zip(['Duals','Inner','ttTime','ttStates'],[[],[],[],[]]))
        inner_dict = dict(zip(['m','route','reward','#states','time'],[None,None,None,None,None]))
        if _DP_ver not in ['ITER_M','SIMUL_M']:
            print("Invalid DP Mode")
        else:
            print('.Running Col. Gen. with DP mode: ', _DP_ver )
            _m_ub_dp = self.constant_dict['max_vehicles_proute_DP']
            # if _m_ub_dp == np.inf:
            #     self.solveModel()
            #     print('Bound upper bound mprDp with initial IP sol:', self.model.ObjVal)
            #     _m_ub_dp = np.ceil(self.model.ObjVal)
            if (_DP_ver == "ITER_M"):
                print("Dominance Checking:",_check_dominance,', rule:',_dominance_rule)
                self.solveRelaxedModel()
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers)
                opt_cond_vect = pd.Series(index = _m_collections,data=False)
                out_loop_counter = 0
                iter_log = dict()
                self.colgenLogs = dict()
                iter_log['es_time'] = 0; iter_log['duals'] = duals_vect[0]; iter_log['cols_gen'] = 0;
                iter_log['cols_add'] = 0; iter_log['max_stops'] = 0;
                self.colgenLogs[out_loop_counter]=iter_log
                self.feasibleStatesExplored = 0
                _outerLogList = []
                t1 = time.time()
                while opt_cond_vect.sum()<len(_m_collections):
                    iter_log = dict(); proc_list = [];
                    iter_log['es_time'] = time.time()
                    iter_log['cols_gen'] = 0; iter_log['cols_add'] = 0; iter_log['max_stops'] = 0
                    _outerLog = outer_dict.copy()
                    _innerLogList = []
                    for _m_veh in _m_collections:
                        _innerLog = inner_dict.copy()
                        if _pricing_status:
                            print('.Running Col. Gen. for m_r:', _m_veh,'| Max nodes visited: %s'%self.max_nodes_proute_DP, '| Out-loop-%s'%out_loop_counter)
                        n = len(self.customers)
                        Q = [0]+list(self.customer_demand.loc[self.customers].values)
    #                     M = 3 #max m per route from collection of m
                        print("\n DUALS:",self.getDuals())
                        s0 = self.fixed_setup_time
                        _inner_t = time.time()
                        S,_st_counter = prizeCollectingDPwTWVer1(n,self.cost_matrix,Q,_m_veh,self.getDuals(),s0,
                          _veh_cap=self.vehicle_capacity,_time_window=self.constant_dict['time_window'],
                          _wavg_factor=self.constant_dict['tw_avg_factor'],_chDom=_check_dominance,
                                                             _stopLim=self.max_nodes_proute_DP)
                        self.feasibleStatesExplored +=_st_counter[0]
                        _inner_t = time.time()-_inner_t
    #                     print(S)
                        P,bestState = pathReconstructionTWVer1(S,Q,self.cost_matrix)
    #                     print(S) #return P,bestState,S
                        reward = bestState[4]
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
                            dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]]) #count repeat visits
                            route_cost = _m_veh #or we can use m from DP?
                            print('m from duals:',-reward+dual_r,'m',_m_veh) #m from DP
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index, columns=['labels'])
                            prefix = str(_m_veh)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = _m_veh
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
                            print(bestState,'M:',_m_veh,'RouteName:',name)
                            print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
                            print('Route:',P,'RouteCost:',route_cost,'Reward:',reward)
                            self.DPRouteDict[name] = prx_route
                            # nCol
                            for idx in col_coeff:
                                nCol.loc[nCol.labels==idx,name] +=1
                            adColDf = pd.DataFrame(columns=['routeCost','colDF'])
                            adColDf.loc[name,['routeCost']] =[route_cost]
                            adColDf.loc[name,['colDF']] = [nCol]
                            #Add columns
    #                         return nCol,adColDf
                            self.generateColumns(adColDf, duals_vect)
                            iter_log['cols_add'] +=1
                            _innerLog['m'] = _m_veh
                            _innerLog['route'] = P
                            _innerLog['reward'] = reward
                            _innerLog['#states'] = _st_counter
                            _innerLog['time'] = _inner_t
                            _innerLogList=_innerLogList+[_innerLog]
                        else: 
                            opt_cond_vect[_m_veh] = True
                            _innerLog['m'] = _m_veh
                            _innerLog['#states'] = _st_counter
                            _innerLog['time'] = _inner_t
                            _innerLogList=_innerLogList+[_innerLog]
                            continue
                        tt_states = sum([len(l) for l in S])
                        iter_log['cols_gen'] += tt_states 
                    #Resolve relax model
                    self.solveRelaxedModel()
                    duals_vect = pd.DataFrame(self.getDuals(), index = self.customers)
                    out_loop_counter+=1
                    iter_log['es_time'] = time.time()-iter_log['es_time']
                    iter_log['duals'] = duals_vect[0]
                    self.colgenLogs[out_loop_counter]=iter_log
                    ######COMPARISON##########
                    _outerLog['ttTime'] = sum([nn['time'] for nn in _innerLogList])
                    _outerLog['ttStates'] = np.sum([nn['#states'] for nn in _innerLogList],axis=0)
                    _outerLog['Duals'] = self.getDuals()
                    _outerLog['Inner'] = _innerLogList
                    _outerLogList = _outerLogList+[_outerLog]
                self.colGenTe = time.time()-t1
                self.colGenCompLog = _outerLogList
                print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
            elif (_DP_ver == "SIMUL_M"):
                self.solveRelaxedModel()
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers)
                opt_cond = False; out_loop_counter = 0; iter_log = dict();self.colgenLogs = dict()
                iter_log['es_time'] = 0; iter_log['duals'] = duals_vect[0]; iter_log['cols_gen'] = 0;
                iter_log['cols_add'] = 0; iter_log['max_stops'] = 0;
                self.colgenLogs[out_loop_counter]=iter_log; self.feasibleStatesExplored = 0
                t1 = time.time(); _outerLogList = []
                while not(opt_cond):
                    iter_log = dict(); proc_list = [];
                    _outerLog = outer_dict.copy(); _innerLog = inner_dict.copy()
                    iter_log['es_time'] = time.time(); iter_log['cols_gen'] = 0; 
                    iter_log['cols_add'] = 0; iter_log['max_stops'] = 0
                    if _pricing_status:
                        print('.Start Running DP','| Max nodes visited: %s'%self.max_nodes_proute_DP,'| Max vehicles per route: %s'%_m_ub_dp,'| Out-loop-%s'%out_loop_counter)
                    n = len(self.customers)
                    Q = [0]+list(self.customer_demand.loc[self.customers].values)
                    s0 = self.fixed_setup_time
#                     if _m_ub_dp > self.relaxedModel.ObjVal: 
#                         print("Bound max veh per dp by relaxObj:", self.relaxedModel.ObjValub_dp)
#                         _m_ub_dp = np.ceil(self.relaxedModel.ObjVal)
                    print("\n DUALS:",np.round(self.getDuals(), 4),"mMAX:",_m_ub_dp)
                    _inner_t = time.time()
                    
                    #  S,_st_counter = prizeCollectingDPwTWVer2(
                    #             n,self.cost_matrix,Q,self.getDuals(),s0,
                    #             _veh_cap=self.vehicle_capacity,
                    #             _time_window=self.constant_dict['time_window'],
                    #             _wavg_factor=self.constant_dict['tw_avg_factor'],
                    #             _mLim=_m_ub_dp,_chDom=_check_dominance,
                    #             _stopLim=self.max_nodes_proute_DP,
                    #             _time_limit=_time_limit,_heu_add_m=_heu_add_m,
                    #             _domVer=_dominance_rule)
                    self.forbid_link_dict, self.necess_link_dict = self.branching_utility.parse_branching_conditions(self.n, _bch_cond)
                    # solver = PrizeCollectingDPwTW(
                    solver = PrizeCollectingDPwTWNewStorage(
                        _n=n, 
                        _C=self.cost_matrix, 
                        _Q=Q,
                        _dual=self.getDuals(), 
                        _s0=s0,
                        _veh_cap=self.vehicle_capacity,
                        _time_window=self.constant_dict['time_window'],
                        _wavg_factor=self.constant_dict['tw_avg_factor'],
                        _m_lim=_m_ub_dp,
                        _dom_ver=_dominance_rule,
                        _ch_dom=_check_dominance,
                        _time_limit=_time_limit,
                        _stop_lim=self.constant_dict['max_nodes_proute_DP'],
                        _heu_add_m=_heu_add_m,
                        _bch_cond=_bch_cond,
                        _forbid_link_dict=self.forbid_link_dict,
                        _necess_link_dict=self.necess_link_dict
                    )
                    # Call the solve method on the instance
                    S,_st_counter = solver.solve()
                    S = solver.convert_to_legacy_format(S)
                    
                    self.feasibleStatesExplored +=_st_counter[0];
                    print('States explored in {0}-iters:{1}'.format(out_loop_counter,_st_counter))
                    print()
                    _inner_t = time.time()-_inner_t
                    PList,bestStateList = pathReconstructionTWVer2(S,Q,self.cost_matrix,
                                                     _filtering_mode,_m_ub_dp,
                                                     _bch_cond=_bch_cond)
                    rwdList = [((b[5]>0.000001) and (b[0]>0)) for b in bestStateList]
                    print('RWDList:',rwdList) #,'BestStateList:',bestStateList)
                    _innerLogList=[]
#                     print(bestStateList)
                    if not(any(rwdList)):
                            opt_cond = True
                            continue
                    for idx in range(len(bestStateList)):
                        _innerLog = inner_dict.copy()
                        P = PList[idx];bestState = bestStateList[idx]
                        reward = bestState[5]
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
                            dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]]) #count repeat visits
                            route_cost = int(round(-reward+dual_r)) #or we can use m from DP?
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index,
                                                columns=['labels'])
                            if _node_count_lab is not None: prefix = f"BnP{_node_count_lab}-{idx}-{out_loop_counter}"
                            else: prefix = str(idx)+str(out_loop_counter)
                            name = f'sDP_C{prefix}-{bestState[7]}'
                            nCol[name] = 0.0
                            nCol.loc[nCol.labels=='m',name] = route_cost
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
#                             print(bestState,'M:',route_cost,'RouteName:',name)
#                             print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
                            print('RouteName:',name,
                                  'Route:',P,
                                #   'col_coeff:',col_coeff,
                                  'RouteCost/DP_m:',f"{route_cost}/{bestState[4]}",
                                  'Reward:',np.round(reward,6))
                            self.DPRouteDict[name] = prx_route
                            # nCol
                            for idx in col_coeff:
                                nCol.loc[nCol.labels==idx,name] +=1
                            adColDf = pd.DataFrame(columns=['routeCost','colDF'])
                            adColDf.loc[name,['routeCost']] =[route_cost]
                            adColDf.loc[name,['colDF']] = [nCol]
                            #Add columns
    #                         return nCol,adColDf
                            self.generateColumns(adColDf, duals_vect)
                            iter_log['cols_add'] +=1
                            _innerLog['m'] = bestState[4]
                            _innerLog['route'] = P
                            _innerLog['reward'] = reward
                            _innerLog['#states'] = None
                            _innerLog['time'] = None
                            _innerLogList=_innerLogList+[_innerLog]
                        else:
                            _innerLog['m'] = bestState[4]
                            _innerLog['route'] = None
                            _innerLog['reward'] = None
                            _innerLog['#states'] = None
                            _innerLog['time'] = None
                            _innerLogList=_innerLogList+[_innerLog]
                    # shortcutting columns, guarantee that each column cannot visit any node more than once
                    # self.shortCuttingColumns()

                    tt_states = sum([len(l) for l in S])
                    iter_log['cols_gen'] += tt_states 
                    #Resolve relax model
                    self.solveRelaxedModel()
                    duals_vect = pd.DataFrame(self.getDuals(), index = self.customers)
                    ######COMPARISON##########
                    _outerLog['ttTime'] = _inner_t
                    _outerLog['ttStates'] = _st_counter
                    _outerLog['Duals'] = self.getDuals()
                    _outerLog['Inner'] = _innerLogList
                    _outerLogList = _outerLogList+[_outerLog]
                    if _update_m_ub:
                        #Solve IP to update upper bound for m
#                         self.shortCuttingColumns()
                        self.solveModel()
                        print("Updating mpr ub from {0} to {1}".format(_m_ub_dp,round(self.model.ObjVal)))
                        _m_ub_dp = round(self.model.ObjVal)
                    out_loop_counter+=1
                    iter_log['es_time'] = time.time()-iter_log['es_time']
                    iter_log['duals'] = duals_vect[0]
                    self.colgenLogs[out_loop_counter]=iter_log
                self.colGenTe = time.time()-t1
                self.colGenCompLog = _outerLogList
                print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
    
    def getWaiting4EachDemand(self, _col):
        _lr = _col.loc[['lr']]
        _visited_nodes = _col[_col.index.isin(self.customers)][_col>=1].index.to_list()
        _qr = sum([self.customer_demand[c] for c in _visited_nodes])
        

    def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
        reformatted_arcs=[]
        ref_df = self.init_routes_df.set_index('labels')
        COLORLIST = ["#FE2712","#347C98","#FC600A",
                     "#66B032","#0247FE","#B2D732",
                    "#FB9902","#4424D6","#8601AF",
                    "#FCCC1A","#C21460","#FEFE33"]
        content_array = ['arcs_list','config','route_info','info_topics','column_width','column_format']
        route_info_topic_array = ['tw_avg_factor','lr','demand_waiting','avg_waiting_per_pkg','pkgs_per_veh','utilization']
        column_width = [2.5,2.5,3.5,3,3,3.2]
        column_format = ['.2f','.2f',None,'.2f','.2f','.2f']
        for j in range(len(_route_name_list)):
            idx = _route_name_list[j]
            col_idx = j%12
#             print(idx)
            curr_route_config = _route_config.copy()
            curr_route_config['line_color'] = COLORLIST[col_idx]
            sample_r = _colums_df.loc[:][idx]
            curr_route_config['name'] = idx+"-"+str(round(_colums_df.loc['m'][idx]))+"m"
            sample_arcs = sample_r[sample_r.index.isin(self.arcs)][sample_r==1]
            sample_nodes = sample_r[sample_r.index.isin(self.customers)][sample_r>=1]
            #Route INFO:
            _qr = sum([self.customer_demand[c] for c in sample_nodes.index.to_list()])
            _mr = round(_colums_df.loc['m'][idx])
            cost_dict = self.route_cost_calculator.calculate_route_metrics(sample_r)
            _avg_CTC_cost = cost_dict['average_total_dem_weighted']
            ############
            _tw_avg = self.constant_dict['tw_avg_factor']
            _lr = sample_r.loc['lr']
            _dem_waiting = cost_dict['dem_waiting']
            _avg_waiting_per_pkg = _avg_CTC_cost/_qr
            _pkgs_per_vehicle = cost_dict['pkgs_served_per_vehicle']
            _util = cost_dict['utilization']*100
            # _pkgs = _lr*(_qr)
            # _util = (self.vehicle_capacity*_mr-self.getRemainingSpace(idx))*100/(self.vehicle_capacity*_mr)
            ###########
            route_info_value = [_tw_avg,_lr,_dem_waiting,_avg_waiting_per_pkg,_pkgs_per_vehicle,_util]
            route_info_dict = dict(zip(route_info_topic_array,route_info_value))
            route_plot_dict = dict(zip(content_array,
                           [sample_arcs.index.to_list(),curr_route_config,route_info_dict,
                            route_info_topic_array,column_width,column_format]))
            reformatted_arcs += [route_plot_dict]
        return reformatted_arcs
    
    def getRouteSolution(self,_model_vars,_edge_plot_config,_node_trace,_cus_dem):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes = sol_vec.loc[sol_vec['value']>=0.98]
        ref_df = self.init_routes_df.set_index('labels')
#         print(ref_df.loc[['m','lr']][optimal_routes.index])
        formatted_routes_list =  self.getRoute4Plot(optimal_routes.index.to_list(),
                                                                ref_df,_edge_plot_config)
        self.EliminateMultipleVisits(formatted_routes_list)
        return formatted_routes_list
    
    
    def EliminateMultipleVisits(self, _route_obj_list):
        # Find multiple visits
        cusList = self.customers
        cusDem = self.customer_demand
        vehicleSpeed = self.truck_speed
        vehicleCapacity = self.vehicle_capacity
        timeWindow = self.constant_dict['time_window']
        visitsRecord = dict(zip(cusList,[[]]*len(cusList)));

        for idx_sol in range(len(_route_obj_list)):
            for (n1,n2) in _route_obj_list[idx_sol]['arcs_list']:
                if( n1 != 'O') and (idx_sol not in visitsRecord[n1]): visitsRecord[n1] = visitsRecord[n1]+[idx_sol]
                if (n2 != 'O') and (idx_sol not in visitsRecord[n2]): visitsRecord[n2] = visitsRecord[n2]+[idx_sol]

        multipleVisitsList = [c for c in cusList if len(visitsRecord[c])>1]
        print('multipleVisitsList',multipleVisitsList )
        print('visitsRecord',visitsRecord )
        # Iterate to correct multiple visits
        disMatrix = self.distance_matrix
        for bCus in multipleVisitsList:
            multipleVisitsRouteIdList = visitsRecord[bCus]
            # Find incident arcs on all routes
            costSavingRecords = dict()
            for r_id in multipleVisitsRouteIdList:
                routeArcs = _route_obj_list[r_id]['arcs_list']
                incidentArcs = [a for a in routeArcs if (bCus in a)]
                inArc = [a for a in incidentArcs if (a[1]==bCus)][0]
                outArc = [a for a in incidentArcs if (a[0]==bCus)][0]
                replacingArc = (inArc[0],outArc[1])
                savingCost = disMatrix[incidentArcs[0]] + disMatrix[incidentArcs[1]] - disMatrix[replacingArc]
                print("incidentArcs %s"%(incidentArcs))
                print("inArc %s, outArc %s"%(inArc,outArc))
                print('cost inArc %f'%disMatrix[incidentArcs[0]],
                      'cost outArc %f'%disMatrix[incidentArcs[1]])
                print("replacingArc",(replacingArc))
#                 print('cost replacingArc %f'%disMatrix[replacingArc],)
                print("savingCost",(savingCost))
                costSavingRecords[r_id] = savingCost
            # Pick max saving route to be shortcut
#             shortCutRouteId = max(costSavingRecords, key =costSavingRecords.get)
            shortCutRouteIdList = sorted(costSavingRecords.items(),
                                         key=itemgetter(1),reverse=True)[:-1]
            print(costSavingRecords,shortCutRouteIdList)

            # Shortcutting process
            for (shortCutRouteId,costSaving) in shortCutRouteIdList:
                # Construct node&arc sequence
                endFlag = False
                routeArcs = _route_obj_list[shortCutRouteId]['arcs_list']
                incidentArcs = [a for a in routeArcs if (bCus in a)]
                inArc = [a for a in incidentArcs if (a[1]==bCus)][0]
                outArc = [a for a in incidentArcs if (a[0]==bCus)][0]
                replacingArc = (inArc[0],outArc[1])
                routeArcs.remove(inArc);routeArcs.remove(outArc);routeArcs.append(replacingArc);
                
                print("\n routeArcs",routeArcs)
                
                i = 'O'; j='';
                nodeSeq = [i]; arcSeq = [];
                lr = 0; qr = 0;
                demandWaitingDict = dict([('O',0)])
                returnDuration = 0;
                
                while(not(endFlag)):
                    for idx in range(len(routeArcs)-1,-1,-1):
                        arc = routeArcs[idx]
                        if (i==arc[0]): 
                            j = arc[1]
                            nodeSeq.append(j); arcSeq.append(arc);
                            qr += cusDem[j]
                            lr += disMatrix[arc]/vehicleSpeed
                            demandWaitingDict[j] = demandWaitingDict[i]+(disMatrix[arc]/vehicleSpeed)
                            i=j; routeArcs.pop(idx)
                        if (j=='O'):
                            demandWaitingDict[j] = 0;
                            endFlag = True
                            returnDuration = disMatrix[arc]/vehicleSpeed
                            lastStop = arc[0];
                            break
                mDemFeas = np.ceil((qr*lr)/(vehicleCapacity))
                mTwFeas = np.ceil((lr)/(timeWindow-(demandWaitingDict[nodeSeq[-2]])))
                optM = max(mDemFeas,mTwFeas)
                avgWaitingAtDepotPerDem = (lr)/(self.constant_dict['tw_avg_factor']*optM)
                avgTravelingTimePerDem = sum([cusDem[n]*demandWaitingDict[n] for n in demandWaitingDict.keys()])/qr
                avgTimeSpentPerDem = avgWaitingAtDepotPerDem+avgTravelingTimePerDem
                for n in nodeSeq: 
                    if n!='O':demandWaitingDict[n] = demandWaitingDict[n]+avgWaitingAtDepotPerDem
                        
                print("\ndemandWaitingDict",demandWaitingDict)   
                remainingSpace = (optM*vehicleCapacity) - (lr*qr)
                utilization = (vehicleCapacity*optM-remainingSpace)*100/(vehicleCapacity*optM)
                print(arcSeq)
                print(nodeSeq)
    #             print("lr", lr)
    #             print("qr", qr)
    #             print("demandWaitingDict", demandWaitingDict)
                print('optM',optM)
    #             print('avgWaitingAtDepotPerDem',avgWaitingAtDepotPerDem)
    #             print('avgTravelingTimePerDem',avgTravelingTimePerDem)
    #             print('avgTimeSpentPerDem',avgTimeSpentPerDem)
    #             print('utilization',utilization)

                # Replace with new route info
                _route_obj_list[shortCutRouteId]['arcs_list'] = arcSeq
                _route_obj_list[shortCutRouteId]['route_info']['lr'] = lr
                _route_obj_list[shortCutRouteId]['route_info']['demand_waiting'] = demandWaitingDict
                _route_obj_list[shortCutRouteId]['route_info']['avg_waiting_per_pkg'] = avgTimeSpentPerDem
                _route_obj_list[shortCutRouteId]['route_info']['pkgs'] = qr*lr
                _route_obj_list[shortCutRouteId]['route_info']['utilization'] = utilization
     
    def plotCurrentSolution(self,_model_vars,_edge_plot_config,_node_trace,_title,_cus_dem):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes = sol_vec.loc[sol_vec['value']>=0.98]
        ref_df = self.init_routes_df.set_index('labels')
        print(ref_df.loc[['m','lr']][optimal_routes.index])
        formatted_routes_list =  self.getRoute4Plot(optimal_routes.index.to_list(),
                                                                ref_df,_edge_plot_config)
        vis_sol.plot_network(formatted_routes_list,_node_trace,_title,_cus_dem)


def pathReconstructionTWVer2(_S, _Q, _C, _filtering_mode=None, _maxM=None, _bch_cond=None):
    if _filtering_mode is None: _filtering_mode = "BestRwdPerI"
    if _filtering_mode not in ["BestRwdPerI","BestRwdPerM"]: print("Incorrect filtering mode!")
    print("Reconstructing Path....")
    print(' Filtering Mode:',_filtering_mode)
    _route_list=[];_bestSt_list=[]
    _dummy_s = [0,0,0,0,0,-np.inf,True,None,None]
    ####### Data from braching cond #####
    _abandon_n_dict = dict();
    forbid_link=[]; necess_link=[]; skip_ending_n=[];
    if _bch_cond is not None: 
        for bh in _bch_cond:
            if (bh[1]==0): 
                forbid_link+=[bh[0]] # 0-branch
                if bh[0][1]=="O":
                    skip_ending_n+=[int(bh[0][0].split("_")[-1])]
            if (bh[1]==1): 
                necess_link+=[bh[0]] # 1-branch
        forbid_link = [bh[0] for bh in _bch_cond if (bh[1]==0)] 
        necess_link = [bh[0] for bh in _bch_cond if (bh[1]==1)] 
    print("skip_ending_n",skip_ending_n)
#     print("bch-conds:",_bch_cond)
#     print("forbidden_link:",forbid_link)
#     print("necessary_link:",necess_link)
    ####################################
    
    if _filtering_mode == "BestRwdPerM":
        #HOW TO FILTER ONLY BEST REWARD FOR EACH M
        _temp_S = [_S[i][:-1] for i in range(len(_S))]
        _maxMPerI=[]
        for i in range(len(_S)):
            if len(_temp_S[i])==0:
                _maxMPerI.append(0)
            else:
                _maxMPerI.append(np.max(np.array(_temp_S[i])[:,4],axis=0))
#         print(_maxMPerI,)
        if _maxM is None: _maxM = int(np.max(_maxMPerI))
        _maxMdict = dict(zip([i for i in range(1,_maxM+1)],[[]]*_maxM))
#         print(_maxMdict,)
        _bestStList = [_S[i][0] for i in range(len(_S))]
        _route_list=[];_bestSt_list=[]
        for _idx in range(1,len(_S)):
#             for _m in range(1,int(_maxMPerI[_idx]+1)):
            for _m in range(1,int(_maxM+1)):
                f_list = list(filter(lambda x:(x[4]==(_m)), _S[_idx]))
#                 print(_m,f_list,_maxMdict[_m])
                if (len(f_list)==0): 
                    if len(_maxMdict[_m])==0: 
                        _maxMdict[_m] = [-1,0,0,0,_m,-np.inf,True,None,None]
                    continue
                else:
                    _bestR = sorted(f_list, key=itemgetter(5),reverse=True)[0]
                    if len(_maxMdict[_m])==0:
                        _maxMdict[_m] = _bestR
                    elif _maxMdict[_m][5] <_bestR[5] :
                        _maxMdict[_m] = _bestR
        print(_maxMdict)
        #Reconstruction
        for _idx in range(1,_maxM+1):
            _route = [0]
            throw_away_flag = False
            lSt = _maxMdict[_idx]
            if ((len(lSt)==0)): continue
            if ((lSt[0]==0)): continue
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route; _bestSt = lSt
            while lP!=0:
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00000001)), _S[prevN]))
                if len(f_list)==0:
                    print("Unmatched State:",lSt)
                    throw_away_flag=True;break
                else:
                    if len(f_list)>1:
                        print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(2),reverse=False)
                        f_list = _temp[:1]
                    lSt = f_list[0]
                    lI = lSt[0]
                    lD = lSt[1]
                    lL = lSt[2]
                    lP = lSt[3]
                    lM = lSt[4]
                    prevN = lSt[7]
                    _route = [lI]+_route
#             print(_idx,_route,_bestSt)
            if (throw_away_flag):
                _route =[] ;_bestSt = [-1,0,0,0,_idx,-np.inf,True,None,None]
            _route_list.append(_route)
            _bestSt_list.append(_bestSt)
            
    elif _filtering_mode == "BestRwdPerI":
        _loop_cond = [True]*len(_S)
        _cter_idx = 1 # node idx
        _order_idx = 0 # First rank of highest reward
        while (any(_loop_cond)) and (_cter_idx<len(_S)):
#             print(_loop_cond,_cter_idx,_order_idx)
            _S[_cter_idx] = sorted(_S[_cter_idx], key=lambda x: (-x[5]))
            
            _route = [0]
            _route_arcs = []
            throw_away_flag = False
            lSt = _S[_cter_idx][_order_idx]
            if ((lSt[0]==0) or (lSt[7] is None)): 
                print("No improvement for state last visit at:",lSt)
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0
                continue
#                 if _order_idx < len(_S[_cter_idx])-1: 
#                     _order_idx+=1
#                 else:
           
            if (lSt[0] in skip_ending_n):
                print("Skip all state ending at:",lSt[0])
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0;
                continue
                
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route
            _route_arcs = [(0,lI),(lI,0)]
            _bestSt = lSt
#             print("Starting new route, i/bestSt: ", lI, _bestSt)
            while lP!=0:
                # this filtering tracing back for exactly matching states
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and
                                      (np.abs(x[1]-(lD-_Q[lI]))<0.00001) and
                                        (x[4]<=lM) and 
                                        ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)),
                                        _S[prevN]))
#                 print('f_list:',f_list)
                if len(f_list)==0:
#                     _order_idx+=1
                    throw_away_flag=True;
                    break
                else:
                    if len(f_list)>1:
#                         print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(5),reverse=True)
                        f_list = _temp[:1]
                    lSt = f_list[0]; lI = lSt[0]; lD = lSt[1]
                    lL = lSt[2]; lP = lSt[3]; lM = lSt[4]
                    prevN = lSt[7]; _route = [lI]+_route
                    _route_arcs = [(0,lI),(lI,_route_arcs[0][1])]+_route_arcs[1:]
#                     print(_route)
#                     print(lSt)
#                 print("Route:",_route,throw_away_flag)
               
            # check branching conditions
    #         if not(throw_away_flag):
    #             for tup in necess_link:
    #                 tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
    #                 if tup[0]==0: # there is (i,j) in route where i!=0
    #                     if (tup not in _route_arcs) and (tup[1] in _route): 
    #                         throw_away_flag = True; break
    #                 elif tup[1]==0: # there is (i,j) in route where j!=0
    #                     if (tup not in _route_arcs) and (tup[0] in _route): 
    #                         throw_away_flag = True; break
    #                 else:
    #                     if (tup not in _route_arcs):
    #                         throw_away_flag = True; break
    # #              
    #                 if throw_away_flag:  print("Skipped:",_route,"No arc:",tup,"State:",)#_S[_cter_idx][_order_idx])
    #                     if _order_idx < len(_S[_cter_idx])-1:
    #                         _order_idx+=1
    #                     else:
    #                         _loop_cond[_cter_idx] = False
    #                         _cter_idx+=1
    #                         _order_idx=0
                    
            
            # filtering out 0-branch as DP cannot forbid ending at i
            if not(throw_away_flag):
                for tup in forbid_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if (tup in _route_arcs): 
                        print("Skipped:",_route,"Forbidden arc:",tup,"State:",)#_S[_cter_idx][_order_idx])
                        throw_away_flag = True
    #                     if _order_idx < len(_S[_cter_idx])-1:
    #                         _order_idx+=1
    #                     else:
    #                         _loop_cond[_cter_idx] = False
    #                         _cter_idx+=1
    #                         _order_idx=0
                        break

            if not(throw_away_flag):
                _route_list.append(_route)
                _bestSt_list.append(_bestSt)
                _loop_cond[_cter_idx] = False
                _cter_idx+=1
                _order_idx=0
            else:
                if _order_idx < len(_S[_cter_idx])-1:
                    _order_idx+=1
                else:
                    _loop_cond[_cter_idx] = False
                    _cter_idx+=1
                    _order_idx=0
                
#                 print("Added route:",_route)
#         print([(_route_list[i],_bestSt_list[i][5]) for i in range(len(_route_list))])#,_bestSt_list)
        return _route_list,_bestSt_list
        
    elif _filtering_mode == "TopKRwdPerI":
        k = 10
        _loop_cond = [True]*len(_S)
        _cter_idx = 1 # node idx
        _order_idx = 0 # First rank of highest reward
        routes_found_at_node = 0  # Track number of valid routes found for current node

        while (any(_loop_cond)) and (_cter_idx<len(_S)) and (len(_S[_cter_idx])>0):
            if _cter_idx >= len(_S) or _order_idx >= len(_S[_cter_idx]):
                _loop_cond[_cter_idx] = False
                _cter_idx+=1
                _order_idx=0
                routes_found_at_node = 0  # Reset counter for new node
                continue

            if _order_idx == 0:  # Only sort when starting a new node
                _S[_cter_idx] = sorted(_S[_cter_idx], key=lambda x: (-x[5]))

            _route = [0]
            _route_arcs = []
            throw_away_flag = False

            # route construction
            lSt = _S[_cter_idx][_order_idx]
            if ((lSt[0]==0) or (lSt[7] is None)): 
                print("No improvement for state last visit at:",lSt)
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0
                continue
           
            if (lSt[0] in skip_ending_n):
                print("Skip all state ending at:",lSt[0])
                _loop_cond[_cter_idx] = False
                _cter_idx+=1; _order_idx=0;
                continue
                
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route
            _route_arcs = [(0,lI),(lI,0)]
            _bestSt = lSt
#             print("Starting new route, i/bestSt: ", lI, _bestSt)
            while lP!=0:
                # this filtering tracing back for exactly matching states
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and
                                      (np.abs(x[1]-(lD-_Q[lI]))<0.00001) and
                                        (x[4]<=lM) and 
                                        ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)),
                                        _S[prevN]))
#                 print('f_list:',f_list)
                if len(f_list)==0:
#                     _order_idx+=1
                    throw_away_flag=True;
                    break
                else:
                    if len(f_list)>1:
#                         print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(5),reverse=True)
                        f_list = _temp[:1]
                    lSt = f_list[0]; lI = lSt[0]; lD = lSt[1]
                    lL = lSt[2]; lP = lSt[3]; lM = lSt[4]
                    prevN = lSt[7]; _route = [lI]+_route
                    _route_arcs = [(0,lI),(lI,_route_arcs[0][1])]+_route_arcs[1:]
                   
            
            # filtering out 0-branch as DP cannot forbid ending at i
            if not(throw_away_flag):
                for tup in forbid_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if (tup in _route_arcs): 
                        print("Skipped:",_route,"Forbidden arc:",tup,"State:",)#_S[_cter_idx][_order_idx])
                        throw_away_flag = True
                        break

            if not(throw_away_flag):
                _route_list.append(_route)
                _bestSt_list.append(_bestSt)
                routes_found_at_node += 1
                
                # Check if we've found k valid routes for this node
                if routes_found_at_node >= k:
                    # Move to next node
                    _loop_cond[_cter_idx] = False
                    _cter_idx += 1
                    _order_idx = 0
                    routes_found_at_node = 0  # Reset for next node
                else:
                    _order_idx += 1
            else:
                _order_idx += 1
                if _order_idx >= len(_S[_cter_idx]):
                    _loop_cond[_cter_idx] = False
                    _cter_idx += 1
                    _order_idx = 0
                    routes_found_at_node = 0  # Reset for next node

        return _route_list,_bestSt_list