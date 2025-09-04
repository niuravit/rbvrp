import sys
sys.path.insert(0,'/Users/ravitpichayavet/Documents/GaTechOR/GraduateResearch/CTC_CVRP/Modules')
import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
# import branch_and_price as bnp
import pandas as pd
import time
import numpy as np
from gurobipy import *
# from gurobipy import Column, Constr, Var
from operator import itemgetter
from solver.pricing.PrizeCollectingDPwMATNewStorage import PrizeCollectingDPwMATNewStorage
from solver.model.timeWindowModel import pathReconstructionTWVer2
import os
# os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
epsilon = 1e-5
from typing import Dict, List, Any, Tuple, Optional
from solver.model.RouteCost import RouteCost

class avgTimeWithTimeWindowModel:
    def __init__(self, _init_route, _initializer,
                 _distance_matrix,constant_dict,
                 extra_constr=None, _model_name = "PhaseII", _mode=None,_relax_route=False):
        if (_init_route is not None):
            self.init_route = _init_route.copy()
            self.route_coeff = _init_route['PathCoeff'].values
            self.route_index = pd.Series(self.init_route.index).index.values

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
        self.total_fleet_size = self.constant_dict['total_fleet_size']
        self.max_nodes_proute_DP = self.constant_dict['max_nodes_proute_DP']
        self.max_vehicles_proute_DP = self.constant_dict['max_vehicles_proute_DP']
        
        
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
            
    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateCostOfRoutes()
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
            self.model.addConstrs( const1,name='customer_coverage' )
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)<=self.total_fleet_size)
            self.model.addConstr(const2,name='vehicles_usage' )
        else: 
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) == int(1) \
                                 for i in self.customer_index )
            self.model.addConstrs( const1,name='customer_coverage' )
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)==self.total_fleet_size)
            self.model.addConstr(const2,name='vehicles_usage' )
        print('Finish generating constrains!')
    
    def generateCostOfRoutes(self):
        t1=time.time()
        # self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.calculateCostOfRoute(col),axis=0)
        self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.route_cost_calculator.calculate_route_metrics(col),axis=0)
        print('Finish generating cost vector!....Elapsed-time:',time.time()-t1)

#     def calculateCostOfRoute(self, route):
#         visiting_nodes = pd.Series(route.iloc[self.customer_index].loc[route>=1].index)
#         visiting_arcs = pd.Series(route.iloc[self.arcs_index][route>=1].index)
#         next_node = ['STR']
#         route_cost = 0
#         qr = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
#         avg_waiting = qr*route['lr']/(2*route['m'])
#         visited_node = []
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
#         cost_dict = dict(zip(['total_cost','avg_waiting','avg_travel','dem_waiting'],[route_cost+avg_waiting,avg_waiting,route_cost,demand_travel_time_dict]))
#         return cost_dict
        
    def generateObjective(self):
        # Minimize the total cost of the used rolls
        if self.mode=='multiObjective':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['average_total_dem_weighted']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TSPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['avg_waiting']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TRPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['travel_dem_weighted']) for rt in self.route_index) ,
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
            _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list() + row.colDF.loc[row.colDF.labels=='m'].iloc[:,-1].to_list()
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
        # for index, row in _filtered_df.iterrows():
        #     _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list() +row.colDF.loc[row.colDF.labels=='m'].iloc[:,-1].to_list()
        #     newColumn = Column(_col, self.model.getConstrs())
        #     _name = row.colDF.columns[-1]
        #     self.addColumn(newColumn,row.routeCost,_name)
        #     self.init_routes_df[_name] = row.colDF.iloc[:,-1]

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
                _route_coef = np.array(sct_seq+arc_route,dtype=object)
                sc_col = pd.Series(index = _cols.index,data=0.0,name=r_name)
                sc_col.loc[_route_coef]+=1
                sc_col.loc['m'] = col['m']
                sc_col.loc['lr'] = lr
                cost_recal = self.route_cost_calculator.calculate_route_metrics(sc_col)['average_total_dem_weighted']
                #Update DF: Use sc_col for updating init_routes_df
                r_var = self.model.getVarByName(r_name)
                self.update_variable_coefficients(sc_col, r_var, cost_recal, forbidden_arcs)
                self.model.update()
                self.init_routes_df.loc[:,r_name] = sc_col.values
                sc_col_count+=1
                # _c_trp = self.calculateCostOfRoute(sc_col)['avg_travel']
                # _cost = _c_tsp+_c_trp
#                 print(r_name,'OldCost:',self.model.getVarByName(r_name).Obj ,'SCTCost:',_cost)
                #Update DF: Use sc_col for updating init_routes_df
                # self.model.getVarByName(r_name).Obj = _cost
                # self.model.update()
                # self.init_routes_df.loc[:,r_name] = sc_col.values
                # sc_col_count+=1
            count+=1
        self.shortcutCols = sc_col_count
        self.shortcutColsPc = len(DP_cols.columns)
        self.shortcutColsTe = time.time()-t1
    
    def update_variable_coefficients(self, sc_col, r_var: Var, var_cost:float, _forbidden_arcs = None):
        """
        Updates the coefficients of a variable in the model using a Gurobi Column object.

        Args:
            sc_col: A pandas Series or similar object containing the new coefficients.
            r_var: The Gurobi variable whose coefficients you want to change.
        """
        # Update the objective coefficient of the existing variable
        var_obj = 1e10 if self.col_contains_forbidden_arc(sc_col,self.convert_forbidden_links(_forbidden_arcs)) else var_cost
        r_var.obj = var_obj

        # 1. Get the list of all constraints in the model
        all_constrs = self.model.getConstrs()

        # 2. Prepare the list of tuples for bulk update
        # The format is (constraint, variable, new_coefficient)
        updates = []
        for const_idx,constr in enumerate(all_constrs):
            if (constr.constrName == "fleet_size"):
                new_coeff = sc_col.loc["m"]
            else:
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
            
    def addVehicleToColumns(self,_M,_var_keywords = None,_mode = "New"):
        if _var_keywords is None: _var_keywords = ""
        _cols = self.init_routes_df.set_index('labels')
        pc_count = 0 
        gen_count = 0
        print("Adding vehicle and generating new columns, mode:",_mode)
        if _mode == "New":
            DP_cols = _cols.loc[:,_cols.columns.str.contains(_var_keywords)].copy()
            for r_name, col in DP_cols.items():
                if ((pc_count/len(DP_cols.columns))*100 % 10)==0:
                    print('processing columns:',pc_count,'/',len(DP_cols.columns)) 
                if ("DP" in r_name):
                    node_seq = list(set(self.DPRouteDict[r_name]))
                    qr = sum([self.customer_demand[c] for c in node_seq])
                    lr = col['lr']
                    _c_trp = self.calculateCostOfRoute(col)['avg_travel']
                    for m_ass in range(int(col['m'])+1,_M+1):
                        new_col_name = r_name+"-ma"+str(m_ass)
                        new_col = col.copy()
                        new_col.loc['m'] = m_ass
                        _c_tsp = (lr*qr)/(2*m_ass)
                        _cost = _c_tsp+_c_trp
    #                     print(col[self.customer_index],col['m'])
                        _temp_col = list(col[self.customer_index].values)+ [m_ass]
                        newColObj = Column(_temp_col, self.model.getConstrs())

                        self.addColumn(newColObj,_cost,new_col_name)
                        self.model.update()
                        self.init_routes_df.loc[:,new_col_name] = new_col.values
                        gen_count+=1
                else: #route from initial set
                    if col['m'] == self.constant_dict['max_vehicles_proute']: 
                        node_seq = pd.Series(col[self.customer_index][col>=1].index)
                        qr = sum([self.customer_demand[c] for c in node_seq])
                        lr = col['lr']
                        old_cost = self.model.getVarByName(r_name).Obj
                        for m_ass in range(int(col['m'])+1,_M+1):
                            new_col_name = r_name+"-ma"+str(m_ass)
                            new_col = col.copy()
                            new_col.loc['m'] = m_ass
                            update_term = -lr*qr*0.5*(m_ass-col['m'])/(col['m']*m_ass)
                            _cost = old_cost+update_term
    #                         print(col[self.customer_index].values,col['m'])
                            _temp_col = list(col[self.customer_index].values)+ [m_ass]
                            newColObj = Column(_temp_col, self.model.getConstrs())

                            self.addColumn(newColObj,_cost,new_col_name)
                            self.model.update()
                            self.init_routes_df.loc[:,new_col_name] = new_col.values
                            gen_count+=1
                pc_count+=1
            print("Finish generating new: {0} columns from original {1} columns".format(gen_count,pc_count))
        elif _mode == "Continue": #Previously added columns with max_m = _M-1
            DP_cols = _cols.loc[:,_cols.columns.str.contains("ma")].copy()
            for r_name, col in DP_cols.items():
                if ((pc_count/len(DP_cols.columns))*100 % 10)==0:
                    print('processing columns:',pc_count,'/',len(DP_cols.columns))
                if col['m'] == _M-1: 
                    node_seq = pd.Series(col[self.customer_index][col>=1].index)
                    qr = sum([self.customer_demand[c] for c in node_seq])
                    lr = col['lr']
                    old_cost = self.model.getVarByName(r_name).Obj
                    for m_ass in range(int(col['m'])+1,_M+1):
                        new_col_name = r_name.replace('ma%s'%str(m_ass-1),'ma%s'%str(m_ass))
                        new_col = col.copy()
                        new_col.loc['m'] = m_ass
                        update_term = -lr*qr*0.5*(m_ass-col['m'])/(col['m']*m_ass)
                        _cost = old_cost+update_term
    #                         print(col[self.customer_index].values,col['m'])
                        _temp_col = list(col[self.customer_index].values)+ [m_ass]
                        newColObj = Column(_temp_col, self.model.getConstrs())

                        self.addColumn(newColObj,_cost,new_col_name)
                        self.model.update()
                        self.init_routes_df.loc[:,new_col_name] = new_col.values
                        gen_count+=1
                pc_count+=1
            print("Finish generating new: {0} columns from original {1} columns".format(gen_count,pc_count))
                        
    def calculateLr(self, route_arcs):
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
#         print(route_arcs,lr)
        return lr

    def runColumnsGeneration(self,_m_collections,_pricing_status=False,
                             _check_dominance=True,_dominance_rule=None,
                             _DP_ver=None,_time_limit=None,_filtering_mode=None,
                            _bch_cond=None,_node_count_lab=None,_acc_flag=None):
        outer_dict = dict(zip(['Duals','Inner','ttTime','ttStates'],[[],[],[],[]]))
        inner_dict = dict(zip(['m','route','reward','#states','time'],[None,None,None,None,None]))
        if _DP_ver not in ['ITER_M','SIMUL_M']:
            print("Invalid DP Mode")
        else:
            print('.Running Col. Gen. with DP mode: ', _DP_ver, "Dom Rule:",_check_dominance,_dominance_rule )
            print('| Max nodes visited: %s'%self.max_nodes_proute_DP,'| Max vehicles per route: %s'%self.max_vehicles_proute_DP)
            if (_DP_ver == "ITER_M"):
                print("Dominance Checking:",_check_dominance,', rule:',_dominance_rule)
                self.solveRelaxedModel()
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                opt_cond_vect = pd.Series(index = _m_collections,data=False)
                out_loop_counter = 0
                t1 = time.time()
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
                        S,_st_counter = prizeCollectingDP(n,self.cost_matrix,Q,_m_veh,self.getDuals(),s0,
                                                          _veh_cap=self.vehicle_capacity,
                                              _chDom=_check_dominance,_stopLim=self.max_nodes_proute_DP)
                        self.feasibleStatesExplored +=_st_counter[0]
                        _inner_t = time.time()-_inner_t
                        P,bestState = pathReconstruction(S,Q,self.cost_matrix)
    #                     print(S);return P,bestState,S
                        reward = bestState[4]
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
                            dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]])
                            route_cost = -reward+dual_r+_m_veh*self.getDuals()[-1]
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index, columns=['labels'])
                            prefix = str(_m_veh)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = _m_veh
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
                            print(bestState,'M:',_m_veh)
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
                            _innerLog['m'] = _m_veh; _innerLog['route'] = P;_innerLog['reward'] = reward
                            _innerLog['#states'] = _st_counter;_innerLog['time'] = _inner_t; 
                            _innerLogList=_innerLogList+[_innerLog]
                        else: 
                            opt_cond_vect[_m_veh] = True
                            _innerLog['m'] = _m_veh; _innerLog['#states'] = _st_counter
                            _innerLog['time'] = _inner_t; _innerLogList=_innerLogList+[_innerLog]
                            continue
                        tt_states = sum([len(l) for l in S])
                        iter_log['cols_gen'] += tt_states
                    #Resolve relax model
                    self.solveRelaxedModel()
                    duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
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
                duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                opt_cond = False; out_loop_counter = 0; iter_log = dict();self.colgenLogs = dict()
                iter_log['es_time'] = 0; iter_log['duals'] = duals_vect[0]; iter_log['cols_gen'] = 0;
                iter_log['cols_add'] = 0; iter_log['max_stops'] = 0;
                self.colgenLogs[out_loop_counter]=iter_log; self.feasibleStatesExplored = 0
                t1 = time.time(); _outerLogList = []
                if _acc_flag is not None: 
                    in_duals = np.array([0]*len(self.customers + ['m']))
                    out_duals = np.array(self.getDuals())
                    primal_bound = self.relaxedModel.ObjVal
                    dual_bound = 0
                    print("\n ==== Using In-out acc. method. Set coeff to:", _acc_flag)
                    print("In dual:", in_duals)
                    print("Out dual:", out_duals)
                    print("---------------------------------")
                    print("Primal bound:", primal_bound)
                    print("Dual bound:", dual_bound)
                    print("---------------------------------")
                    
#                     out_duals = np.array(self.getDuals())
#                     conv_comb_duals = in_duals*(_acc_flag)+out_duals*(1-_acc_flag)
                    
                while not(opt_cond):
                    iter_log = dict(); proc_list = [];
                    _outerLog = outer_dict.copy(); _innerLog = inner_dict.copy()
                    iter_log['es_time'] = time.time(); iter_log['cols_gen'] = 0; 
                    iter_log['cols_add'] = 0; iter_log['max_stops'] = 0
                    
                    n = len(self.customers)
                    Q = [0]+list(self.customer_demand.loc[self.customers].values)
                    s0 = self.fixed_setup_time
                    if _acc_flag is not None: 
                        print("Primal bound:", primal_bound); print("Dual bound:", dual_bound)
                        if (primal_bound-dual_bound)<epsilon: 
                            print("\n ==== GAP LESS THAN EPSILON, OPTIMAL")
#                             print("In dual:", in_duals)
#                             print("Out dual:", out_duals)
#                             print("---------------------------------")
                            print("Primal bound:", primal_bound)
                            print("Dual bound:", dual_bound)
                            print('Feasible States explored in {}-iters:{}'.format(out_loop_counter,
                                                                                   self.feasibleStatesExplored))
                            print("---------------------------------")
                            opt_cond = True;  continue;
                        elif (dual_bound>1e6):
                            print("\n ==== INFEASIBLE NODE")
                            opt_cond = True; continue;
                        # Duals being used
                        _duals = in_duals*(_acc_flag)+out_duals*(1-_acc_flag)
                        print("\n ==== GAP STILL LARGER THAN EPSILON")
                        print("In dual:", in_duals)
                        print("Out dual:", out_duals)
                        print("CONVEX COMB DUAL:", _duals)
#                         print("---------------------------------")
                    else:
                        if _pricing_status:
                            print(' Out-loop-%s'%out_loop_counter)
                            print(' Start Running DP','| Max nodes visited: %s'%self.max_nodes_proute_DP,'| Max vehicles per route: %s'%self.max_vehicles_proute_DP,)
                            print(' Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom,"Domination Rule:", _dom_rule)
                        # Duals being used
                        _duals = self.getDuals()
                    print("\n DUALS:",np.round(_duals, 4))
                    _inner_t = time.time()

                    self.forbid_link_dict, self.necess_link_dict = self.parse_branching_conditions(_bch_cond)
                    solver = PrizeCollectingDPwMATNewStorage(
                        _n=n, 
                        _C=self.cost_matrix, 
                        _Q=Q,
                        _dual=self.getDuals(), 
                        _s0=s0,
                        _veh_cap=self.vehicle_capacity,
                        _time_window=self.constant_dict['time_window'],
                        _wavg_factor=self.constant_dict['tw_avg_factor'],
                        _m_lim=self.constant_dict['max_vehicles_proute_DP'],
                        _total_fleet_size = self.constant_dict['total_fleet_size'],
                        _dom_ver=_dominance_rule,
                        _ch_dom =_check_dominance,
                        _time_limit=_time_limit,
                        _stop_lim=self.constant_dict['max_nodes_proute_DP'],
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
                    PList,bestStateList = pathReconstructionTWVer2(S,Q,
                                            self.cost_matrix,_filtering_mode,
                                            self.max_vehicles_proute_DP,
                                            _bch_cond=_bch_cond)
                    rwdList = [((b[5]>0.000001) and (b[0]>0)) for b in bestStateList]
                    print('RWDList:',rwdList) #,'BestStateList:',bestStateList)
                    _innerLogList=[]; 
#                     print(bestStateList)
                    if _acc_flag is not None: 
                        if not(any(rwdList)): # Dual is feasible, so no improved route can be found!
                            # Update in-dual and dual-bound
                            in_duals = _duals
                            dual_bound = sum(_duals[:-1])+self.max_vehicles*_duals[-1]
#                             print("\n ==== NOT FOUND IMPROVEMENT, DUAL FEASIBLE, UPDATE IN-DUAL")
#                             print("In dual:", in_duals)
#                             print("Out dual:", out_duals)
#                             print("---------------------------------")
#                             print("Primal bound:", primal_bound)
#                             print("Dual bound:", dual_bound)
#                             print("---------------------------------")
                            continue # start the new DP with new dual
                        else: # Dual is infeasible, so update out-dual
                            pass
#                             out_duals = _duals
                            # Then, add new columns and update the primal bound
                            
                            
                    else:
                        if not(any(rwdList)): 
                            opt_cond = True; 
                            continue;
                        # for idx in range(len(bestStateList)):
                        #     _innerLog = inner_dict.copy()
                        #     P = PList[idx];bestState = bestStateList[idx]
                        #     reward = bestState[5]         
                        for idx in range(len(bestStateList)):
                            _innerLog = inner_dict.copy()
                            P = PList[idx];bestState = bestStateList[idx]
                            reward = bestState[5]
                            ## Filtering columns
                            if (reward>0.000001) and (bestState[0]>0):
                                dual_r = sum([_duals[i-1] for i in P[1:-1]]) #count repeat visits
                                route_cost = -reward+dual_r+bestState[4]*_duals[-1]
                                prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                                arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                                col_coeff = prx_route+arc_route
                                nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index,
                                                    columns=['labels'])
                                if _node_count_lab is not None: prefix =f"BnP-P2{_node_count_lab}-{idx}-{out_loop_counter}"
                                else: prefix = str(idx)+str(out_loop_counter)
                                name = f'sDP_C{prefix}-{bestState[7]}'
                                nCol[name] = 0.0
                                nCol.loc[nCol.labels=='m',name] = bestState[4]
                                nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
                                print('RouteName:',name,
                                  'Route:',P,
                                  'RouteCost:',f"{route_cost}",
                                  'm_r:',f"{bestState[4]}",
                                  'Reward:',np.round(reward,4))
                                self.DPRouteDict[name] = prx_route
                                # nCol
                                for idx in col_coeff:
                                    nCol.loc[nCol.labels==idx,name] +=1
                                adColDf = pd.DataFrame(columns=['routeCost','colDF'])
                                adColDf.loc[name,['routeCost']] =[route_cost]
                                adColDf.loc[name,['colDF']] = [nCol]
                                #Add columns
        #                         return nCol,adColDf
    #                             self.generateColumns(adColDf, duals_vect)
                                self.generateColumns(adColDf, _duals)
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
                        tt_states = sum([len(l) for l in S])
                        iter_log['cols_gen'] += tt_states 
                        #Resolve relax model
                        self.solveRelaxedModel()
                        if _acc_flag is not None: 
                            primal_bound = self.relaxedModel.ObjVal
                            out_duals = np.array(self.getDuals())
#                         print("\n ==== NOT FOUND IMPROVEMENT, DUAL FEASIBLE, UPDATE IN-DUAL")
#                         print("In dual:", in_duals)
#                         print("Out dual:", out_duals)
#                         print("---------------------------------")
#                         print("Primal bound:", primal_bound)
#                         print("Dual bound:", dual_bound)
#                         print("---------------------------------")
#                     duals_vect =  pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
                        ######COMPARISON##########
                        _outerLog['ttTime'] = _inner_t
                        _outerLog['ttStates'] = _st_counter
                        _outerLog['Duals'] = self.getDuals()
                        _outerLog['Inner'] = _innerLogList
                        _outerLogList = _outerLogList+[_outerLog]
                        out_loop_counter+=1
                        iter_log['es_time'] = time.time()-iter_log['es_time']
                        iter_log['duals'] = _duals[0]
                        self.colgenLogs[out_loop_counter]=iter_log
                self.colGenTe = time.time()-t1
                self.colGenCompLog = _outerLogList
                print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
        
    # def calculateAverageRemainingSpace(self,_model_vars,):
    #     vars_value = pd.Series(_model_vars)
    #     sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
    #     sol_vec['value'] = vars_value.apply(lambda x:x.X).values
    #     optimal_routes_name_list = sol_vec.loc[sol_vec['value']>=0.98].index.to_list()
    #     _cumulative_space = 0
    #     for j in range(len(optimal_routes_name_list)):
    #         _route_name = optimal_routes_name_list[j]
    #         _cumulative_space+=self.getRemainingSpace(_route_name)
    #     _avg_rem_space = _cumulative_space/self.total_fleet_size
    #     return _avg_rem_space
        
    # def getRemainingSpace(self, _route_name):
    #     '''Absolute remaining space for all mr'''
    #     ref_df = self.init_routes_df.set_index('labels')
    #     _col = ref_df.loc[:][_route_name]
    #     _mr = _col['m']
    #     node_seq = pd.Series(_col.iloc[self.customer_index][_col>=0.7].index)
    #     _qr = sum([self.customer_demand[c] for c in node_seq])
    #     _lr = _col['lr']
    #     _abs_rem_space = (_mr*self.vehicle_capacity) - (_lr*_qr)
    #     print(_route_name,', Rem. space=',_abs_rem_space,', mr=',_mr)
    #     return _abs_rem_space
        
    def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
        reformatted_arcs=[]
        ref_df = self.init_routes_df.set_index('labels')
        COLORLIST = ["#FE2712","#347C98","#FC600A",
                     "#66B032","#0247FE","#B2D732",
                    "#FB9902","#4424D6","#8601AF",
                    "#FCCC1A","#C21460","#FEFE33"]
        content_array = ['arcs_list','config','route_info','info_topics',
                         'column_width','column_format']
        route_info_topic_array = ['lr','total_demand','demand_waiting','avg_waiting_per_pkg','pkgs_per_veh','utilization']
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
            _lr = sample_r.loc['lr']
            _dem_waiting = cost_dict['dem_waiting']
            _avg_waiting_per_pkg = _avg_CTC_cost/_qr
            _pkgs_per_vehicle = cost_dict['pkgs_served_per_vehicle']
            _util = cost_dict['utilization']*100
            # _pkgs = _lr*(_qr)
            # _util = (self.vehicle_capacity*_mr-self.getRemainingSpace(idx))*100/(self.vehicle_capacity*_mr)
            ###########
            route_info_value = [_lr,_qr,_dem_waiting,_avg_waiting_per_pkg,_pkgs_per_vehicle,_util]
            route_info_dict = dict(zip(route_info_topic_array,route_info_value))
            route_plot_dict = dict(zip(content_array,
                           [sample_arcs.index.to_list(),curr_route_config,
                            route_info_dict,route_info_topic_array,column_width,column_format]))
            reformatted_arcs += [route_plot_dict]
        return reformatted_arcs    

#     def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
#         reformatted_arcs=[]
#         COLORLIST = ["#FE2712","#347C98","#FC600A",
#                      "#66B032","#0247FE","#B2D732",
#                     "#FB9902","#4424D6","#8601AF",
#                     "#FCCC1A","#C21460","#FEFE33"]
#         for j in range(len(_route_name_list)):
#             idx = _route_name_list[j]
#             col_idx = j%12
# #             print(idx)
#             curr_route_config = _route_config.copy()
#             curr_route_config['line_color'] = COLORLIST[col_idx]
#             sample_r = _colums_df.loc[:][idx]
#             curr_route_config['name'] = idx+"-"+str(round(_colums_df.loc['m'][idx]))+"m"
#             sample_arcs = sample_r[sample_r.index.isin(self.arcs)][sample_r==1]
#             route_plot_dict = dict(zip(['arcs_list','config'],
#                            [sample_arcs.index.to_list(),curr_route_config]))
#             reformatted_arcs += [route_plot_dict]
#         return reformatted_arcs
    
    def getRouteSolution(self,_model_vars,_edge_plot_config,_node_trace,_cus_dem):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes = sol_vec.loc[sol_vec['value']>=0.98]
        ref_df = self.init_routes_df.set_index('labels')
#         print(ref_df.loc[['m','lr']][optimal_routes.index])
        formatted_routes_list =  self.getRoute4Plot(optimal_routes.index.to_list(),
                                                                ref_df,_edge_plot_config)
        return formatted_routes_list
     
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

    def parse_branching_conditions(self, _bch_cond: List[Tuple[Tuple[str, str], int]]):
        """Parses branching conditions into dictionaries of forbidden and necessary links."""
        forbid_link_dict = {i: [] for i in range(self.n + 1)}
        necess_link_dict = {i: [] for i in range(self.n + 1)}

        if _bch_cond is None: 
            return forbid_link_dict, necess_link_dict
        
        forbid_link = [bh[0] for bh in _bch_cond if (bh[1] == 0)]
        necess_link = [bh[0] for bh in _bch_cond if (bh[1] == 1)]
        
        
        
        # Helper to parse node labels from strings like 'c_5' or 'O'
        def parse_node(node_str: str) -> int:
            if node_str == 'O':
                return 0
            return int(node_str.split("_")[-1])
            
        for arc in forbid_link:
            i = parse_node(arc[0]); j = parse_node(arc[1])
            forbid_link_dict[i].append(j)
            
        for arc in necess_link:
            i = parse_node(arc[0]); j = parse_node(arc[1])
            necess_link_dict[i].append(j)
            
            # Propagate necessary links to forbid others
            if i == 0: # Necessary link from depot: other links from depot are forbidden
                for k in range(self.n + 1):
                    if k != j and k not in forbid_link_dict[i]:
                        forbid_link_dict[i].append(k)
            elif j == 0: # Necessary link to depot: other links to depot are forbidden
                for k in range(self.n + 1):
                    if k != i and j not in forbid_link_dict[k]:
                        forbid_link_dict[k].append(j)
            else: # Necessary link between customers
                for k in range(self.n + 1):
                    if k != i and j not in forbid_link_dict[k]:
                        forbid_link_dict[k].append(j) # No other path to j
                    if k != j and k not in forbid_link_dict[i]:
                        forbid_link_dict[i].append(k) # No other path from i
                        
        return forbid_link_dict, necess_link_dict