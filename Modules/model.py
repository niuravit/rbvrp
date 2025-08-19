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
from operator import itemgetter
import os
os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
epsilon = 1e-5



class phaseIModel:
    def __init__(self, _init_route, _initializer,
                 _distance_matrix,constant_dict,
                 extra_constr=None, _model_name = "PhaseI"):
        
        self.init_route = _init_route.copy()
        self.route_coeff = _init_route['PathCoeff'].values
        
        self.depot = _initializer.depot
        self.depot_s = _initializer.depot_s
        self.depot_t = _initializer.depot_t
        self.all_depot = _initializer.all_depot
        self.customers = _initializer.customers
        self.nodes = _initializer.nodes
        self.arcs = _initializer.arcs
    

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
        
        self.route_index = pd.Series(self.init_route.index).index.values
        self.model = Model(_model_name)

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()
        
    def generateVariables(self):
        self.route = self.model.addVars(self.route_index, lb=0,
                                       vtype=GRB.BINARY, name='route')
        print('Finish generating variables!')
        
    def generateConstraints(self):          
        const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) ==1 \
                             for i in self.customer_index )
        self.model.addConstrs( const1,name='customer_coverage' )
        print('Finish generating constrains!')
        
    def generateObjective(self):
        # Minimize the total cost of the used rolls
        self.model.setObjective( quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index) ,
                                sense=GRB.MINIMIZE)
        print('Finish generating objective!')
        
    def solveModel(self, timeLimit = None,GAP=None):
        if timeLimit is not None: self.model.setParam('TImeLimit', timeLimit)
        if GAP is not None: self.model.setParam('MIPGap',GAP)
        self.model.setParam('SolutionNumber',2)
        self.model.setParam(GRB.Param.PoolSearchMode, 2)
        self.model.optimize()
        
    def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
        reformatted_arcs=[]
        for idx in _route_name_list:
            print(idx)
            curr_route_config = _route_config.copy()
            sample_r = _colums_df.loc[:][idx]
            curr_route_config['name'] = idx
            sample_arcs = sample_r[sample_r.index.isin(self.arcs)][sample_r==1]
            route_plot_dict = dict(zip(['arcs_list','config'],
                           [sample_arcs.index.to_list(),curr_route_config]))
            reformatted_arcs += [route_plot_dict]
        return reformatted_arcs


class phaseIIModel:
    def __init__(self, _init_route, _initializer,
                 _distance_matrix,constant_dict,
                 extra_constr=None, _model_name = "PhaseII", _mode=None,_relax_route=False):
        
        self.init_route = _init_route.copy()
        self.route_coeff = _init_route['PathCoeff'].values
        self.init_routes_df = _initializer.init_routes_df.copy()
        self.relax_route_flag = _relax_route
        
        self.depot = _initializer.depot
        self.depot_s = _initializer.depot_s
        self.depot_t = _initializer.depot_t
        self.all_depot = _initializer.all_depot
        self.customers = _initializer.customers
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
        self.max_vehicles_proute_DP = self.constant_dict['max_vehicles_proute_DP']
        
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
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)<=self.max_vehicles)
            self.model.addConstr(const2,name='vehicles_usage' )
        else: 
            const1 = ( quicksum(self.route_coeff[rt][i]*self.route[rt] for rt in self.route_index) == int(1) \
                                 for i in self.customer_index )
            self.model.addConstrs( const1,name='customer_coverage' )
            const2 = (quicksum(self.route[rt]*(self.route_coeff[rt][self.veh_no_index[0]]) for rt in self.route_index)==self.max_vehicles)
            self.model.addConstr(const2,name='vehicles_usage' )
        print('Finish generating constrains!')
    
    def generateCostOfRoutes(self):
        t1=time.time()
        self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.calculateCostOfRoute(col),axis=0)
        print('Finish generating cost vector!....Elapsed-time:',time.time()-t1)
    def calculateCostOfRoute(self, route):
        visiting_nodes = pd.Series(route[self.customer_index][route>=1].index)
        visiting_arcs = pd.Series(route[self.arcs_index][route>=1].index)
        next_node = ['STR']
        route_cost = 0
        qr = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        avg_waiting = qr*route['lr']/(2*route['m'])
        visited_node = []
        demand_travel_time_dict = dict(zip(visiting_nodes,[route['lr']*self.constant_dict['tw_avg_factor']/route['m']]*len(visiting_nodes)))
        acc_distance = 0
        while next_node[0]!=self.depot[0]:
            if next_node[0] == 'STR': next_node.pop(0);selecting_node = self.depot[0] #only for the first arc
            else: selecting_node = next_node.pop(0)
#             print(selecting_node)
#             print(visited_node)
            outgoing_arc_list = visiting_arcs[visiting_arcs.apply(lambda x: ((x[0]==selecting_node) and (x[1] not in visited_node) ))].to_list()
            if (selecting_node != self.depot[0]): visited_node.append(selecting_node)
#             print("outgoing",outgoing_arc_list)
            outgoing_arc = outgoing_arc_list[0]
            node_j = outgoing_arc[1]
            next_node.append(node_j)
            qj = self.customer_demand[node_j]
            traveling_time_carrying_pkg = qr*(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            acc_distance +=(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            route_cost+=traveling_time_carrying_pkg
            qr = qr-qj
            if node_j!=self.depot[0]:
                demand_travel_time_dict[node_j] += acc_distance
#         print(avg_waiting,route_cost)
#         route_cost += avg_waiting
#         print(route_cost)
        cost_dict = dict(zip(['total_cost','avg_waiting','avg_travel','dem_waiting'],[route_cost+avg_waiting,avg_waiting,route_cost,demand_travel_time_dict]))
        return cost_dict
        
    def generateObjective(self):
        # Minimize the total cost of the used rolls
        if self.mode=='multiObjective':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['total_cost']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TSPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['avg_waiting']) for rt in self.route_index) ,
                                    sense=GRB.MINIMIZE)
        elif self.mode=='TRPOnly':
            self.model.setObjective( quicksum(self.route[rt]*(self.route_cost[rt]['avg_travel']) for rt in self.route_index) ,
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
        return self.relaxedModel.getAttr('Pi',self.model.getConstrs())
    
    ##COLUMNS GENERATION
    def addColumn(self,_col_object,_col_cost,_name):
        self.model.addVar(lb=0,vtype=GRB.BINARY,column=_col_object, obj= _col_cost, name=_name)
        self.model.update()
    
    def generateColumns(self,_filtered_df,_duals, ):
        for index, row in _filtered_df.iterrows():
            _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list() +row.colDF.loc[row.colDF.labels=='m'].iloc[:,-1].to_list()
            newColumn = Column(_col, self.model.getConstrs())
            _name = row.colDF.columns[-1]
            self.addColumn(newColumn,row.routeCost,_name)
            self.init_routes_df[_name] = row.colDF.iloc[:,-1]

    def shortCuttingColumns(self,_var_keywords = 'DP'):
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
                _c_tsp = (lr*qr)/(2*col['m'])
                _route_coef = np.array(sct_seq+arc_route,dtype=object)
                sc_col = pd.Series(index = _cols.index,data=0,name=r_name)
                sc_col.loc[_route_coef]+=1
                sc_col.loc['m'] = col['m']
                sc_col.loc['lr'] = lr
#                 print(sc_col)
#                 return sc_col
                _c_trp = self.calculateCostOfRoute(sc_col)['avg_travel']
                _cost = _c_tsp+_c_trp
#                 print(r_name,'OldCost:',self.model.getVarByName(r_name).Obj ,'SCTCost:',_cost)
                #Update DF: Use sc_col for updating init_routes_df
                self.model.getVarByName(r_name).Obj = _cost
                self.model.update()
                self.init_routes_df.loc[:,r_name] = sc_col.values
                sc_col_count+=1
            count+=1
        self.shortcutCols = sc_col_count
        self.shortcutColsPc = len(DP_cols.columns)
        self.shortcutColsTe = time.time()-t1
            
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
#                     print("\n DUALS:",_duals)
                    _inner_t = time.time()
                    if _bch_cond is None:
                        S,_st_counter = prizeCollectingDPVer2(
                                    n,self.cost_matrix,Q,
                                    _duals,s0,_mprDP=self.max_vehicles_proute_DP,
                                    _ttM=self.max_vehicles,
                                    _veh_cap=self.vehicle_capacity,
                                    _chDom=_check_dominance,
                                    _stopLim=self.max_nodes_proute_DP,
                                    _domVer=_dominance_rule,
                                    _time_limit=_time_limit)
                    else:
                        S,_st_counter = prizeCollectingDPVer2(
                                    n,self.cost_matrix,Q,
                                    _duals,s0,_mprDP=self.max_vehicles_proute_DP,
                                    _ttM=self.max_vehicles,
                                    _veh_cap=self.vehicle_capacity,
                                    _chDom=_check_dominance,
                                    _stopLim=self.max_nodes_proute_DP,
                                    _domVer=_dominance_rule,
                                    _time_limit=_time_limit,
                                    _bch_cond=_bch_cond)
                    self.feasibleStatesExplored +=_st_counter[0];
#                     print('States explored in {0}-iters:{1}'.format(out_loop_counter,_st_counter))
                    _inner_t = time.time()-_inner_t
                    #*******Need to be completed
                    PList,bestStateList = pathReconstructionCTCVer2(S,Q,
                                            self.cost_matrix,_filtering_mode,self.max_vehicles_proute_DP,
                                            _bch_cond=_bch_cond)
                    rwdList = [((b[5]>0.000001) and (b[0]>0)) for b in bestStateList]
#                     print('RWDList:',rwdList) #,'BestStateList:',bestStateList)
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
                        if not(any(rwdList)): opt_cond = True; continue;
                    for idx in range(len(bestStateList)):
                        _innerLog = inner_dict.copy()
                        P = PList[idx];bestState = bestStateList[idx]
                        reward = bestState[5]
#                         print(bestState,P)
                        ## Filtering columns
                        if (reward>0.000001) and (bestState[0]>0):
#                             dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]]) #count repeat visits
                            dual_r = sum([_duals[i-1] for i in P[1:-1]]) #count repeat visits
#                             route_cost = -reward+dual_r+bestState[4]*self.getDuals()[-1]
                            route_cost = -reward+dual_r+bestState[4]*_duals[-1]
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index,
                                                columns=['labels'])
                            if _node_count_lab is not None: prefix ="BnP%s-"%(_node_count_lab)+ str(idx)+str(out_loop_counter)
                            else: prefix = str(idx)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = bestState[4]
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
#                             print("SanityCheck: lr_recal and lr_state",nCol.loc[nCol.labels=='lr',name][0],bestState[2]+self.cost_matrix[(bestState[0],0)])
                            print(bestState,'RouteCost:',route_cost,'RouteName:',name)
#                             print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
                            print('Route:',P,'M:',bestState[4],'Reward:',reward)
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
                
                        
                    
        
    def calculateAverageRemainingSpace(self,_model_vars,):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes_name_list = sol_vec.loc[sol_vec['value']>=0.98].index.to_list()
        _cumulative_space = 0
        for j in range(len(optimal_routes_name_list)):
            _route_name = optimal_routes_name_list[j]
            _cumulative_space+=self.getRemainingSpace(_route_name)
        _avg_rem_space = _cumulative_space/self.max_vehicles
        return _avg_rem_space
        
    def getRemainingSpace(self, _route_name):
        '''Absolute remaining space for all mr'''
        ref_df = self.init_routes_df.set_index('labels')
        _col = ref_df.loc[:][_route_name]
        _mr = _col['m']
        node_seq = pd.Series(_col[self.customer_index][_col>=0.7].index)
        _qr = sum([self.customer_demand[c] for c in node_seq])
        _lr = _col['lr']
        _abs_rem_space = (_mr*self.vehicle_capacity) - (_lr*_qr)
        print(_route_name,', Rem. space=',_abs_rem_space,', mr=',_mr)
        return _abs_rem_space
        
    def getRoute4Plot(self, _route_name_list, _colums_df,_route_config):
        reformatted_arcs=[]
        ref_df = self.init_routes_df.set_index('labels')
        COLORLIST = ["#FE2712","#347C98","#FC600A",
                     "#66B032","#0247FE","#B2D732",
                    "#FB9902","#4424D6","#8601AF",
                    "#FCCC1A","#C21460","#FEFE33"]
        content_array = ['arcs_list','config','route_info','info_topics',
                         'column_width','column_format']
        route_info_topic_array = ['lr','total_demand','demand_waiting','avg_waiting_per_pkg','pkgs','utilization']
        column_width = [3,3,3.5,3,3.2,3.2]
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
            cost_dict = self.calculateCostOfRoute(sample_r)
            _avg_CTC_cost = cost_dict['total_cost']
            ############
            _lr = sample_r.loc['lr']
            _dem_waiting = cost_dict['dem_waiting']
            _avg_waiting_per_pkg = _avg_CTC_cost/_qr
            _pkgs = _lr*(_qr)
            _util = (self.vehicle_capacity*_mr-self.getRemainingSpace(idx))*100/(self.vehicle_capacity*_mr)
            ###########
            route_info_value = [_lr,_qr,_dem_waiting,_avg_waiting_per_pkg,_pkgs,_util]
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
    
        
def prizeCollectingDPVer2(_n,_C,_Q,_dual,_s0,_veh_cap,_mprDP,_ttM,_chDom=True,_stopLim=None,_domVer=None,
                          _time_limit=None,_bch_cond=[]):
    if _stopLim is None: _stopLim = _n
    if _time_limit is None: _time_limit = np.inf; 
#     print('Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom)
    if _mprDP is None: _mprDP = _ttM
#     print('Setting max-vehicles-proute DP to:',_mprDP)
#     print('Setting max-stops-proute DP to:',_stopLim)
    if _chDom:
        if (_domVer is None): print("Please specify dominance version! _domVer is ",_domVer)
        elif (_domVer not in [2]): print("Wrong input of dominance version! _domVer should be in",[1,2,3])
#         else: print("Dominance Version:",_domVer)
    
    ####### Data from braching cond #####
    forbid_link = [bh[0] for bh in _bch_cond if (bh[1]==0)] # 0-branch
    necess_link = [bh[0] for bh in _bch_cond if (bh[1]==1)] # 1-branch
    forbid_link_dict = dict(zip([x for x in range(_n+1)],[[]]*(_n+1)))
    necess_link_dict = dict(zip([x for x in range(_n+1)],[[]]*(_n+1)))
    for arc in forbid_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
        forbid_link_dict[i]=forbid_link_dict[i]+[j]
    for arc in necess_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
        necess_link_dict[i]=necess_link_dict[i]+[j]
        for k in range(_n+1):
            if k!=i and k!=j: 
                if i==0: # Only restrict movement from k!=0 -> j
                    if (j not in forbid_link_dict[k]): forbid_link_dict[k]=forbid_link_dict[k]+[j]
                elif j==0: # Only restrict movement from i -> all other k!=0
                    if (k not in forbid_link_dict[i]): forbid_link_dict[i]=forbid_link_dict[i]+[k]
                else: # Only restrict movement from k!=i -> j and i -> all other k!=j
                    if (j not in forbid_link_dict[k]): forbid_link_dict[k]=forbid_link_dict[k]+[j]
                    if (k not in forbid_link_dict[i]): forbid_link_dict[i]=forbid_link_dict[i]+[k]
                
#     print("bch-conds:",_bch_cond)
#     print("forbidden_link:",forbid_link_dict)
#     print("necessary_link:",necess_link_dict)
    ####################################
    _counter = 0; _rch_counter = 0; _h0 = 0
    # Def I: L = [i,d,l,p,m,rwd,pFlag,prevN,counter]
    # L0 = [0,0,0,0,0,0,False,None,_counter] 
    # Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf]
    
    # Def II: L = [i,d,l,p,m,rwd,pFlag,prevN,counter]
    L0 = [0,0,0,0,0,0,False,None,_counter,_h0] 
    Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf,np.inf]
    S = [[L0]]+[[[x+1]+Linf] for x in range(_n)]; _mu = _dual[-1]
    tFlag = False
    _time = time.time()
    _neg_cost_counter=0
    while not(tFlag):
        for i in range(_n+1):
            #TIME limit checking
            if (time.time()-_time)>_time_limit:
                print('Reach time limit!! ','{0}/{1}'.format(time.time()-_time,_time_limit))
                print('Rwd each state:',[S[i][0][5] for i in range(_n+1)])
                pos_rwd_exist = any([s[5]>0.000001 for s in [S[i][0] for i in range(_n+1)]])
                if pos_rwd_exist: tFlag=True;break;
                else: 
                    _neg_cost_counter+=1;
                    if _neg_cost_counter>=5: tFlag=True;break;
                    print(_neg_cost_counter,'No positive reduced cost found! Reset time & continue searching... ','states explored:',sum([len(x) for x in S]));_time = time.time()
            for w in range(len(S[i])):
                if not(S[i][w][6]):
                    if S[i][w][3] >= _stopLim:
                        S[i][w][6] = True
                    else:
                        # Avoid forbidden link (0-branch)
                        reachTo = [x for x in range(1,_n+1) if ((x!=i) and (x not in forbid_link_dict[i]) and (x!=S[i][w][7]))] 
                        S[i][w][6] = True
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][w][1] + _Q[j]
                            _l = S[i][w][2] + _C[(i,j)]
                            _p = S[i][w][3] + 1
                            
                            if (_mprDP*_veh_cap < (_d*(_l+_C[(j,0)]))):
                                # Biggest m is infeasible
                                continue
                            else:
                                _phi = np.sqrt((_d)*(_l+_C[(j,0)])/(2*np.abs(_mu)))
                                _m_opt = getOptM(_phi,_mprDP, _d, _l+_C[(j,0)], _mu)    
                                # Optimal m is greater than DP limit, set it equal to the limit
                                if _m_opt>_mprDP: _m_opt = _mprDP
                                # Optimal m is less than or equal to DP limit
                                else:
                                    _serving_cap_feas = (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)])))
                                    # If Opt m is not demand feasible, increase until feasible or reaching the DP limit
                                    while (not(_serving_cap_feas) and (_m_opt < _mprDP)):
                                        _m_opt+=1
                                        _serving_cap_feas = (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)])))
    #                             print(_m_opt)
                            # Only allows demand feasible state to be constructed
                            if (_m_opt*_veh_cap >= (_d*(_l+_C[(j,0)]))):
                                _h_opt = h_func(_m_opt, _d, _l+_C[(j,0)], _mu);
                                _rwd = S[i][w][5] + transRwdVer2(S[i][w],j,_C,_Q,_m_opt,_dual,_s0)
                                nS = [j,_d,_l,_p,_m_opt,_rwd,False,i]
                                _counter+=1
                                nS = nS+[_counter,_h_opt]
                                if _chDom: 
                                    if _domVer==2: S = checkDominanceCTCVer2(nS,S,_C,_mu)
                                    elif _domVer==3: S = checkDominanceCTCVer3(nS,S,_C,_mu,_mprDP)
#                                     elif _domVer==2:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                     elif _domVer==3:S = checkDominanceTWVer4(nS,S,_mLim)
                                    else: print("ERROR: WRONG DOM. VER."); return;
                                else: S[j].append(nS)
                            S[i][w][6] = True
            #Sort S[i]
            _temp = sorted(S[i], key=itemgetter(5),reverse=True)
            S[i] = _temp
#         print(S)
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][6]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break
        if _all_pc: tFlag=True
    return S,(_counter,_rch_counter)

def h_func(_phi, _q, _lr, _mu):
    if _phi>0: return (_mu*(_phi)) - ((_q*_lr)/(2*_phi))
    else: return 0

def getOptM(_phi,_M, _q, _lr, _mu):
    cphi = np.ceil(_phi); fphi = np.floor(_phi);
    cel_h = h_func(cphi, _q, _lr, _mu);flr_h = h_func(fphi, _q, _lr, _mu);
    if (cel_h >= flr_h) and (cphi <= _M):
        opt_m = cphi; 
    elif (cel_h < flr_h) and (fphi <= _M):
        opt_m = fphi; 
    else:
        opt_m = _M; 
    return opt_m


def transRwdVer2(_cS,_j,_C,_Q,_opt_m,_dual,_s0):
    _i = _cS[0]
    _nD = _cS[1]+_Q[_j]
    _nL = _cS[2]+ _C[(_i,_j)]
    _nPi = _dual[_j-1] #dual's index doesnt have depot
    _nM = _opt_m
    _mu = _dual[-1]
    _reCostI = _C[(_i,0)]
    _reCostJ = _C[(_j,0)]
    if (_cS[4]==0):
        _tRwd = _nPi +_mu*(_nM)+\
                - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_nM)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))
    else:
        _tRwd = _nPi +_mu*(_nM-_cS[4])+\
                - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_nM)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))\
                + ((0.5/_cS[4])*_cS[1]*(_cS[2]+_reCostI+_s0))
    return _tRwd

def checkDominanceCTCVer2(_nS,_S,_C,_mu):
    i = _nS[0]; adFlag = True;
#     h_nS = h_func(_nS[4], _nS[1], _nS[2]+_C[(_nS[0],0)], _mu)
    h_nS = _nS[9]
    for w in range(len(_S[i])-1,-1,-1):
        cpS = _S[i][w]
#         h_cpS = h_func(cpS[4], cpS[1], cpS[2]+_C[(cpS[0],0)], _mu)
        h_cpS = cpS[9]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(h_cpS-h_nS)>=cpS[5])):
#             print(_nS[4],"<=",cpS[4],(_nS[4]<=cpS[4]), "|",h_nS,">=",h_cpS, (h_nS>=h_cpS))
            del _S[i][w]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(h_cpS-h_nS)<=cpS[5])):
#             print(_nS[4],">=",cpS[4],(_nS[4]>=cpS[4]), "|",h_nS,"<=",h_cpS, (h_nS<=h_cpS))
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    return _S

def checkDominanceCTCVer3(_nS,_S,_C,_mu,_mprDP):
    #dominance with m_cap
    i = _nS[0]; adFlag = True;
#     h_nS = h_func(_nS[4], _nS[1], _nS[2]+_C[(_nS[0],0)], _mu)
    h_nS = _nS[9]; m_nS = _nS[4]
    for w in range(len(_S[i])-1,-1,-1):
        cpS = _S[i][w]; h_cpS = cpS[9]; m_cpS = cpS[4]
        if ((m_nS==m_cpS) and (m_nS==_mprDP)): #Only Za>Zb is sufficent
#         if ((m_nS==m_cpS)) : #For test! & it's wrong!
            if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]>=cpS[5])):
                del _S[i][w]
            elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]<=cpS[5])):
                adFlag = False #Throw away _nS
                break
        else: #Need correction terms 
            if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(h_cpS-h_nS)>=cpS[5])):
                del _S[i][w]
            elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(h_cpS-h_nS)<=cpS[5])):
                adFlag = False #Throw away _nS
                break
    if adFlag:
        _S[i].append(_nS)
    return _S


def pathReconstructionCTCVer2(_S,_Q,_C,_filtering_mode=None,_mprDP=None,_bch_cond=None):
    if _filtering_mode is None: _filtering_mode = "BestRwdPerI"
    if _filtering_mode not in ["BestRwdPerI","BestRwdPerM"]: print("Incorrect filtering mode!")
#     print(' Filtering Mode:',_filtering_mode)
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
#     print("skip_ending_n",skip_ending_n)
#     print("bch-conds:",_bch_cond)
#     print("forbidden_link:",forbid_link)
#     print("necessary_link:",necess_link)
    ####################################
    
    
    if _filtering_mode == "BestRwdPerM":
        _temp_S = [_S[i][:-1] for i in range(len(_S))]
        _maxMPerI=[]
        for i in range(len(_S)):
            if len(_temp_S[i])==0:
                _maxMPerI.append(0)
            else:
                _maxMPerI.append(np.max(np.array(_temp_S[i])[:,4],axis=0))
        if _mprDP is None: _mprDP = int(np.max(_maxMPerI))
        _maxMdict = dict(zip([i for i in range(1,_mprDP+1)],[[]]*_mprDP))
        _bestStList = [_S[i][0] for i in range(len(_S))]
        _route_list=[];_bestSt_list=[]
        for _idx in range(1,len(_S)):
            for _m in range(1,int(_mprDP+1)):
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
        for _idx in range(1,_mprDP+1):
            _route = [0]
            throw_away_flag = False
            lSt = _maxMdict[_idx]
            if ((len(lSt)==0)): continue
            if ((lSt[0]==0)): continue
            lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
            lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
            _route = [lI]+_route
            _bestSt = lSt
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
                    lSt = f_list[0]; lI = lSt[0]; lD = lSt[1]
                    lL = lSt[2]; lP = lSt[3]; lM = lSt[4]
                    prevN = lSt[7]; _route = [lI]+_route
#             print(_idx,_route,_bestSt)
            if (throw_away_flag):
                _route =[] ;_bestSt = [-1,0,0,0,_idx,-np.inf,True,None,None]
            _route_list.append(_route)
            _bestSt_list.append(_bestSt)
                
    elif _filtering_mode == "BestRwdPerI":
        _loop_cond = [True]*len(_S)
        _cter_idx = 0 # node idx
        _order_idx = 0 # First rank of highest reward
        
        _bestStList = [_S[i][0] for i in range(len(_S))]
#         for _idx in range(1,len(_bestStList)):
#             _route = [0]
#             throw_away_flag = False
#             lSt = _bestStList[_idx]
#     #         print(lSt)
#     #         print(sorted(_temp, key=itemgetter(5),reverse=True),lSt)
#             if ((lSt[0]==0)): continue
#             lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
#             lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
#             _route = [lI]+_route
#             _route_arcs = [(0,lI),(lI,0)]
#             _bestSt = lSt
            
        while (any(_loop_cond)) and (_cter_idx<len(_S)):
#             print(_loop_cond,_cter_idx,_order_idx)
            _route = [0]
            _route_arcs = []
            throw_away_flag = False
            lSt = _S[_cter_idx][_order_idx]
            if ((lSt[0]==0) or (lSt[7] is None)): 
#                 print("No improvement for state last visit at:",lSt)
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
            
            while lP!=0:
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and (np.abs(x[1]-(lD-_Q[lI]))<0.00001) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)), _S[prevN]))
    #             print(f_list)
                if len(f_list)==0:
#                     _f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[4]<=lM), _S[prevN]))
#                     print("Stop at state:",lSt)
#                     print("Filtered:",_f_list)
                    throw_away_flag=True;break
                else:
                    if len(f_list)>1:
#                         print("ALERT!:",f_list)
                        _temp = sorted(f_list, key=itemgetter(5),reverse=True)
                        f_list = _temp[:1]
                    lSt = f_list[0]
                    lI = lSt[0]; lD = lSt[1]; lL = lSt[2]
                    lP = lSt[3]; lM = lSt[4]; prevN = lSt[7]
                    _route = [lI]+_route;
                    _route_arcs = [(0,lI),(lI,_route_arcs[0][1])]+_route_arcs[1:]
#                 print("Route:",_route,throw_away_flag)

            # check branching conditions
            if not(throw_away_flag):
                for tup in necess_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if tup[0]==0: # there is (i,j) in route where i!=0
                        if (tup not in _route_arcs) and (tup[1] in _route): 
                            throw_away_flag = True; break
                    elif tup[1]==0: # there is (i,j) in route where j!=0
                        if (tup not in _route_arcs) and (tup[0] in _route): 
                            throw_away_flag = True; break
                    else:
                        if (tup not in _route_arcs):
                            throw_away_flag = True; break
    #               
                    if throw_away_flag:  print("Skipped:",_route,"No arc:",tup,"State:",)
            
            # filtering out 0-branch as DP cannot forbid ending at i
            if not(throw_away_flag):
                for tup in forbid_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if (tup in _route_arcs): 
                        print("Skipped:",_route,"Forbidden arc:",tup,"State:",) #_S[_cter_idx][_order_idx])
                        throw_away_flag = True
                        break
    #         print("Route:",_route,throw_away_flag)
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
    #             print("RouteList:",_route,_bestSt)
#         print(_route_list,_bestSt_list)
    return _route_list,_bestSt_list
    
        
def prizeCollectingDP(_n,_C,_Q,_M,_dual,_s0,_veh_cap,_chDom=True,_stopLim=None):
    if _stopLim is None: _stopLim = _n
    _counter = 0;_rch_counter=0
    L0 = [0,0,0,0,_M*_dual[-1],False,None,_counter]
    S = [[L0]]+[[] for x in range(_n)]
    tFlag = False
    while not(tFlag):
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]):
                    if S[i][l][3] >= _stopLim:
                        S[i][l][5] = True
                    else:
                        reachTo = [x for x in range(1,_n+1) if x!=i] #dont need 0 and i
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][l][1] + _Q[j]
                            _l = S[i][l][2] + _C[(i,j)]
                            _p = S[i][l][3] + 1
                            _rwd = S[i][l][4] + transRwd(S[i][l],j,_C,_Q,_M,_dual,_s0)
                            nS = [j,_d,_l,_p,_rwd,False,i]
                            if checkFeasibility(nS,_veh_cap,_M,_C):
                                _counter+=1
                                nS = nS+[_counter]
                                if _chDom: S = checkDominance(nS,S)
                                else: 
    #                                 print("HEYYY")
                                    S[j].append(nS)
                                    temp = sorted(S[j], key=itemgetter(4),reverse=True)
                                    S[j] = temp
                            S[i][l][5] = True                        
#             print(S)
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break

        if _all_pc: tFlag=True
    return S,(_counter,_rch_counter)

def checkFeasibility(_nS,_veh_cap, _m, _C):
    _rt_cost = _C[(_nS[0],0)]
    deliver_cap = _nS[1]*(_nS[2]+_rt_cost)
    limit_cap = _veh_cap*_m
#     print('.....FEASIBILITY:','resNode:',_nS[0],'| deliver_cap:',round(deliver_cap,2),'| cap_limit:',round(limit_cap,2))
    if deliver_cap <= limit_cap: return True
    else: return False
    
def transRwd(_cS,_j,_C,_Q,_M,_dual,_s0):
    _i = _cS[0]
    _nD = _cS[1]+_Q[_j]
    _nL = _cS[2]+ _C[(_i,_j)]
    _nPi = _dual[_j-1] #dual's index doesnt have depot
    _reCostI = _C[(_i,0)]
    _reCostJ = _C[(_j,0)]
    _tRwd = _nPi - (_Q[_j]*(_cS[2]+_C[(_i,_j)])) \
                - ((0.5/_M)*_nD*(_cS[2]+_C[(_i,_j)]+_reCostJ+_s0))\
                + ((0.5/_M)*_cS[1]*(_cS[2]+_reCostI+_s0))
    return _tRwd
    
def checkDominance(_nS,_S):
    i = _nS[0]
    adFlag = True
#     print(_S)
    for l in range(len(_S[i])-1,-1,-1):
#         print(l)
        cpS = _S[i][l]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]>cpS[4])):
            del _S[i][l]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]<cpS[4])):
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    #sort _S[i] by reward
    _temp = sorted(_S[i], key=itemgetter(4),reverse=True)
    _S[i] = _temp
#     print('AfterDominance:',_S[i])
    return _S

def pathReconstruction(_S,_Q,_C):
    _route = [0]
    _temp = [_S[i][0] for i in range(len(_S))]
    lSt = sorted(_temp, key=itemgetter(4),reverse=True)[0]
#     print(sorted(_temp, key=itemgetter(4),reverse=True),lSt)
    if ((lSt[0]==0)): lSt = sorted(_temp, key=itemgetter(4),reverse=True)[1]
    lI = lSt[0]
    lD = lSt[1]
    lL = lSt[2]
    lP = lSt[3]
    prevN = lSt[6]
    _route = [lI]+_route
    _bestSt = lSt
    while lP!=0:
        f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.000001)), _S[prevN]))
#         print(f_list)
        if len(f_list)>1:
            print("ALERT!:",f_list)
            _temp = sorted(f_list, key=itemgetter(4),reverse=True)
            f_list = _temp[:1]
        lSt = f_list[0]
        lI = lSt[0]
        lD = lSt[1]
        lL = lSt[2]
        lP = lSt[3]
        prevN = lSt[6]
        _route = [lI]+_route
    return _route,_bestSt



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
    
#     def generateCostOfRoutes(self):
#         t1=time.time()
#         self.route_cost = self.init_routes_df.set_index('labels').apply(lambda col: self.calculateCostOfRoute(col),axis=0)
#         print('Finish generating cost vector!....Elapsed-time:',time.time()-t1)
    def calculateCostOfRoute(self, route):
        visiting_nodes = pd.Series(route[self.customer_index][route>=1].index)
        visiting_arcs = pd.Series(route[self.arcs_index][route>=1].index)
        next_node = ['STR']
        route_cost = 0
        qr = visiting_nodes.apply(lambda x: self.customer_demand[x]).sum()
        avg_waiting = qr*route['lr']/(2*route['m'])
        visited_node = []
        demand_travel_time_dict = dict(zip(visiting_nodes,[route['lr']*self.constant_dict['tw_avg_factor']/route['m']]*len(visiting_nodes)))
        acc_distance = 0
        while next_node[0]!=self.depot[0]:
            if next_node[0] == 'STR': next_node.pop(0);selecting_node = self.depot[0] #only for the first arc
            else: selecting_node = next_node.pop(0)
#             print(selecting_node)
#             print(visited_node)
            outgoing_arc_list = visiting_arcs[visiting_arcs.apply(lambda x: ((x[0]==selecting_node) and (x[1] not in visited_node) ))].to_list()
            if (selecting_node != self.depot[0]): visited_node.append(selecting_node)
#             print("outgoing",outgoing_arc_list)
            outgoing_arc = outgoing_arc_list[0]
            node_j = outgoing_arc[1]
            next_node.append(node_j)
            qj = self.customer_demand[node_j]
            traveling_time_carrying_pkg = qr*(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            acc_distance +=(self.distance_matrix[outgoing_arc])/self.constant_dict['truck_speed']
            route_cost+=traveling_time_carrying_pkg
            qr = qr-qj
            if node_j!=self.depot[0]:
                demand_travel_time_dict[node_j] += acc_distance
#         print(avg_waiting,route_cost)
#         route_cost += avg_waiting
#         print(route_cost)
        cost_dict = dict(zip(['total_cost','avg_waiting','avg_travel','dem_waiting'],[route_cost+avg_waiting,avg_waiting,route_cost,demand_travel_time_dict]))
        return cost_dict
        
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
        return self.relaxedModel.getAttr('Pi',self.model.getConstrs())
    
    ##COLUMNS GENERATION
    def addColumn(self,_col_object,_col_cost,_name):
        self.model.addVar(lb=0,vtype=GRB.BINARY,column=_col_object, obj= _col_cost, name=_name)
        self.model.update()
    
    def generateColumns(self,_filtered_df,_duals, ):
        for index, row in _filtered_df.iterrows():
            _col = row.colDF.loc[row.colDF.index[self.customer_index]].iloc[:,-1].to_list()
            newColumn = Column(_col, self.model.getConstrs())
            _name = row.colDF.columns[-1]
            self.addColumn(newColumn,row.routeCost,_name)
            self.init_routes_df[_name] = row.colDF.iloc[:,-1]

    def shortCuttingColumns(self,_var_keywords = 'DP'):
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
                _c_tsp = (lr*qr)/(2*col['m'])
                _route_coef = np.array(sct_seq+arc_route,dtype=object)
                sc_col = pd.Series(index = _cols.index,data=0,name=r_name)
                sc_col.loc[_route_coef]+=1
                sc_col.loc['m'] = col['m']
                sc_col.loc['lr'] = lr
#                 print(sc_col)
#                 return sc_col
#                 print(r_name,'OldCost:',self.model.getVarByName(r_name).Obj ,'SCTCost:',_cost)
                #Update DF: Use sc_col for updating init_routes_df
                r_var = self.model.getVarByName(r_name)
                m_constrs = self.model.getConstrs()
                r_var.Obj = col['m'] # update new m
                up_list =[]
#                 for const_idx in range(len(m_constrs)):
#                     new_coeff = sc_col.loc["c_%s"%(const_idx+1)]
#                     self.model.chgCoeff(m_constrs[const_idx],r_var,new_coeff)
#                     up_list.append(["c_%s"%(const_idx+1),new_coeff])
#                 print(sc_col[:20],up_list)
                self.model.update()
                self.init_routes_df.loc[:,r_name] = sc_col.values
                sc_col_count+=1
            count+=1
        self.shortcutCols = sc_col_count
        self.shortcutColsPc = len(DP_cols.columns)
        self.shortcutColsTe = time.time()-t1
                        
    def calculateLr(self, route_arcs):
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
#         print(route_arcs,lr)
        return lr

    def runColumnsGeneration(self,_m_collections,_pricing_status=False, _check_dominance=True,_dominance_rule=None,_DP_ver=None, _update_m_ub=False, _time_limit=None,_filtering_mode=None,_heu_add_m=False, _bch_cond=None,_node_count_lab=None):
        outer_dict = dict(zip(['Duals','Inner','ttTime','ttStates'],[[],[],[],[]]))
        inner_dict = dict(zip(['m','route','reward','#states','time'],[None,None,None,None,None]))
        if _DP_ver not in ['ITER_M','SIMUL_M']:
            print("Invalid DP Mode")
        else:
            print('.Running Col. Gen. with DP mode: ', _DP_ver )
            _m_ub_dp = self.constant_dict['max_vehicles_proute_DP']
            if _m_ub_dp == np.inf:
                self.solveModel()
                print('Bound upper bound mprDp with initial IP sol:', self.model.ObjVal)
                _m_ub_dp = np.ceil(self.model.ObjVal)
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
                    print("\n DUALS:",self.getDuals(),"mMAX:",_m_ub_dp)
                    _inner_t = time.time()
                    if _bch_cond is None:
                         S,_st_counter = prizeCollectingDPwTWVer2(
                                    n,self.cost_matrix,Q,self.getDuals(),s0,
                                    _veh_cap=self.vehicle_capacity,
                                    _time_window=self.constant_dict['time_window'],
                                    _wavg_factor=self.constant_dict['tw_avg_factor'],
                                    _mLim=_m_ub_dp,_chDom=_check_dominance,
                                    _stopLim=self.max_nodes_proute_DP,
                                    _time_limit=_time_limit,_heu_add_m=_heu_add_m,
                                    _domVer=_dominance_rule)
                    else:
                        S,_st_counter = prizeCollectingDPwTWVer3(
                                    n,self.cost_matrix,Q,self.getDuals(),s0,
                                    _veh_cap=self.vehicle_capacity,
                                    _time_window=self.constant_dict['time_window'],
                                    _wavg_factor=self.constant_dict['tw_avg_factor'],
                                    _mLim=_m_ub_dp,_chDom=_check_dominance,
                                    _stopLim=self.max_nodes_proute_DP,
                                    _time_limit=_time_limit,_heu_add_m=_heu_add_m,
                                    _domVer=_dominance_rule,
                                    _bch_cond=_bch_cond)
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
#                         print(bestState,P)
                        ## Filtering columns
                        if (reward>0.00001) and (bestState[0]>0):
                            dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]]) #count repeat visits
                            route_cost = int(round(-reward+dual_r)) #or we can use m from DP?
                            prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
                            arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
                            col_coeff = prx_route+arc_route
                            nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index,
                                                columns=['labels'])
                            if _node_count_lab is not None: prefix ="BnP%s-"%(_node_count_lab)+ str(idx)+str(out_loop_counter)
                            else: prefix = str(idx)+str(out_loop_counter)
                            name = 'sDP_C%s-%s'%(prefix,bestState[7])
                            nCol[name] = 0
                            nCol.loc[nCol.labels=='m',name] = route_cost
                            nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
#                             print(bestState,'M:',route_cost,'RouteName:',name)
#                             print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
                            print('RouteName:',name, 'Route:',P,'RouteCost:',route_cost,'Reward:',reward)
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
        
    def calculateAverageRemainingSpace(self,_model_vars,):
        vars_value = pd.Series(_model_vars)
        sol_vec = pd.DataFrame(index = vars_value.apply(lambda x:x.VarName))
        sol_vec['value'] = vars_value.apply(lambda x:x.X).values
        optimal_routes_name_list = sol_vec.loc[sol_vec['value']>=0.98].index.to_list()
        _cumulative_space = 0
        for j in range(len(optimal_routes_name_list)):
            _route_name = optimal_routes_name_list[j]
            _cumulative_space+=self.getRemainingSpace(_route_name)
        _avg_rem_space = _cumulative_space/self.model.ObjVal
        return _avg_rem_space
        
    def getRemainingSpace(self, _route_name):
        '''Absolute remaining space for all mr'''
        ref_df = self.init_routes_df.set_index('labels')
        _col = ref_df.loc[:][_route_name]
        _mr = _col['m']
        node_seq = pd.Series(_col[self.customer_index][_col>=0.7].index)
        _qr = sum([self.customer_demand[c] for c in node_seq])
        _lr = _col['lr']
        _abs_rem_space = (_mr*self.vehicle_capacity) - (_lr*_qr)
        print(_route_name,', Rem. space=',_abs_rem_space,', mr=',_mr)
        return _abs_rem_space
    
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
        route_info_topic_array = ['tw_avg_factor','lr','demand_waiting','avg_waiting_per_pkg','pkgs','utilization']
        column_width = [3,2.5,3.5,4,3,3.2]
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
            cost_dict = self.calculateCostOfRoute(sample_r)
            _avg_CTC_cost = cost_dict['total_cost']
            ############
            _tw_avg = self.constant_dict['tw_avg_factor']
            _lr = sample_r.loc['lr']
            _dem_waiting = cost_dict['dem_waiting']
            _avg_waiting_per_pkg = _avg_CTC_cost/_qr
            _pkgs = _lr*(_qr)
            _util = (self.vehicle_capacity*_mr-self.getRemainingSpace(idx))*100/(self.vehicle_capacity*_mr)
            ###########
            route_info_value = [_tw_avg,_lr,_dem_waiting,_avg_waiting_per_pkg,_pkgs,_util]
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
                                         key=operator.itemgetter(1),reverse=True)[:-1]
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



def prizeCollectingDPwTWVer2(_n,_C,_Q,_dual,_s0,_veh_cap,_time_window,_wavg_factor,_mLim,_chDom=True,_stopLim=None,_time_limit=None,_heu_add_m=False,_domVer=None):
    if _stopLim is None: _stopLim = _n
    if _time_limit is None: _time_limit = np.inf; 
    print('Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom)
    if _chDom:
        if (_domVer is None): print("Please specify dominance version! _domVer is ",_domVer)
        elif (_domVer not in [2,3,4]): print("Wrong input of dominance version! _domVer should be in",[2,3])
        else: print("Dominance Version:",_domVer)
    _counter = 0; _rch_counter = 0
    L0 = [0,0,0,0,0,0,False,None,_counter]
    Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf]
    S = [[L0]]+[[[x+1]+Linf] for x in range(_n)]
    tFlag = False
    _time = time.time()
    _neg_cost_counter=0
    _l_threshold = _time_window-(_veh_cap/(_stopLim*(max(_Q)))); 
#     print("l-threshold: ",_l_threshold)
    while not(tFlag):
        for i in range(_n+1):
            #TIME limit checking
            if (time.time()-_time)>_time_limit:
                print('Reach time limit!! ','{0}/{1}'.format(time.time()-_time,_time_limit))
                print('Rwd each state:',[S[i][0][5] for i in range(_n+1)])
                pos_rwd_exist = any([s[5]>0.000001 for s in [S[i][0] for i in range(_n+1)]])
                if pos_rwd_exist: tFlag=True;break;
                else: 
                    _neg_cost_counter+=1;
                    if _neg_cost_counter>=5: tFlag=True;break;
                    print(_neg_cost_counter,'No positive reduced cost found! Reset time & continue searching... ','states explored:',sum([len(x) for x in S]));_time = time.time()
            for w in range(len(S[i])):
                if not(S[i][w][6]):
                    if S[i][w][3] >= _stopLim:
                        S[i][w][6] = True
                    else:
                        reachTo = [x for x in range(1,_n+1) if x!=i] #dont need 0 and i
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][w][1] + _Q[j]
                            _l = S[i][w][2] + _C[(i,j)]
                            _p = S[i][w][3] + 1
                            if (_time_window-(_l)>0): # time feasibility
                                m_ctc = np.ceil((_d*(_l+_C[(j,0)]))/(_veh_cap))
                                m_tw = np.ceil((_l+_C[(j,0)])/(_time_window-_l))
                                m = max(m_ctc,m_tw)
                                _rwd = S[i][w][5] + _dual[j-1] + S[i][w][4] - m # dual's index doesnt have depot
                                if m<=_mLim:
                                    nS = [j,_d,_l,_p, m,_rwd,False,i]
                                    _counter+=1
                                    nS = nS+[_counter]
                                    if _chDom:
                                        if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
                                        elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
                                        elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
                                        else: print("ERROR: WRONG DOM. VER."); return;
                                    else: S[j].append(nS)
                            S[i][w][6] = True
            #Sort S[i]
            _temp = sorted(S[i], key=itemgetter(5),reverse=True)
            S[i] = _temp
#         print(_counter,[S[i][0] for i in range(_n+1)],"\n")
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][6]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break
        if _all_pc: tFlag=True
#         for i in range(_n+1):
#             _temp = sorted(S[i], key=itemgetter(5),reverse=True)
#             S[i] = _temp
    return S,(_counter,_rch_counter)

def prizeCollectingDPwTWVer3(_n,_C,_Q,_dual,_s0,_veh_cap,_time_window,_wavg_factor,_mLim,_chDom=True,_stopLim=None,_time_limit=None,_heu_add_m=False,_domVer=None,_bch_cond=[]):
    if _stopLim is None: _stopLim = _n
    if _time_limit is None: _time_limit = np.inf; 
    print('Solving time limit set to:',_time_limit,'secs.',"Dominance Checking:",_chDom)
    if _chDom:
        if (_domVer is None): print("Please specify dominance version! _domVer is ",_domVer)
        elif (_domVer not in [2,3,4]): print("Wrong input of dominance version! _domVer should be in",[2,3])
        else: print("Dominance Version:",_domVer)
    ####### Data from braching cond #####
    forbid_link = [bh[0] for bh in _bch_cond if (bh[1]==0)] # 0-branch
    necess_link = [bh[0] for bh in _bch_cond if (bh[1]==1)] # 1-branch
    forbid_link_dict = dict(zip([x for x in range(_n+1)],[[]]*(_n+1)))
    necess_link_dict = dict(zip([x for x in range(_n+1)],[[]]*(_n+1)))
    for arc in forbid_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
        forbid_link_dict[i]=forbid_link_dict[i]+[j]
    for arc in necess_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
        necess_link_dict[i]=necess_link_dict[i]+[j]
        for k in range(_n+1):
            if k!=i and k!=j: 
                if i==0: # Only restrict movement from k!=0 -> j
                    if (j not in forbid_link_dict[k]): forbid_link_dict[k]=forbid_link_dict[k]+[j]
                elif j==0: # Only restrict movement from i -> all other k!=0
                    if (k not in forbid_link_dict[i]): forbid_link_dict[i]=forbid_link_dict[i]+[k]
                else: # Only restrict movement from k!=i -> j and i -> all other k!=j
                    if (j not in forbid_link_dict[k]): forbid_link_dict[k]=forbid_link_dict[k]+[j]
                    if (k not in forbid_link_dict[i]): forbid_link_dict[i]=forbid_link_dict[i]+[k]
                
    print("bch-conds:",_bch_cond)
    print("forbidden_link:",forbid_link_dict)
    print("necessary_link:",necess_link_dict)
    ####################################
    
    _counter = 0; _rch_counter = 0;_abandon_n = []
    # State = [i, acc_demand , acc_length, stops, m, reward, reached_flg, prevN, counter, one_bch]
    L0 = [0,0,0,0,0,0,False,None,_counter,False]
    Linf = [np.inf,np.inf,np.inf,np.inf,-np.inf,True,None,np.inf,False]
    S = [[L0]]+[[[x+1]+Linf] for x in range(_n)]
    tFlag = False
    _time = time.time()
    _neg_cost_counter=0
    _l_threshold = _time_window-(_veh_cap/(_stopLim*(max(_Q)))); 
#     print("l-threshold: ",_l_threshold)
    while not(tFlag):
        for i in range(_n+1):
            #TIME limit checking
            if (time.time()-_time)>_time_limit:
                print('Reach time limit!! ','{0}/{1}'.format(time.time()-_time,_time_limit))
                print('Rwd each state:',[S[i][0][5] for i in range(_n+1)])
                pos_rwd_exist = any([s[5]>0.000001 for s in [S[i][0] for i in range(_n+1)]])
                if pos_rwd_exist: tFlag=True;break;
                else: 
                    _neg_cost_counter+=1;
                    if _neg_cost_counter>=5: tFlag=True;break;
                    print(_neg_cost_counter,'No positive reduced cost found! Reset time & continue searching... ','states explored:',sum([len(x) for x in S]));_time = time.time()
            for w in range(len(S[i])):
#             for w in range(len(S[i])-1, -1, -1):
                if not(S[i][w][6]):
                    if S[i][w][3] >= _stopLim:
                        S[i][w][6] = True
                    else:
#                         Avoid forbidden link (0-branch)
                        reachTo = [x for x in range(1,_n+1) if ((x!=i) and (x not in forbid_link_dict[i]) and (x!=S[i][w][7]))] 
                        S[i][w][6] = True
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][w][1] + _Q[j]
                            _l = S[i][w][2] + _C[(i,j)]
                            _p = S[i][w][3] + 1
                            if (_time_window-(_l)>0): # time feasibility
                                m_ctc = np.ceil((_d*(_l+_C[(j,0)]))/(_veh_cap))
                                m_tw = np.ceil((_l+_C[(j,0)])/(_time_window-_l))
                                m = max(m_ctc,m_tw)
                                _rwd = S[i][w][5] + _dual[j-1] + S[i][w][4] - m # dual's index doesnt have depot
                                if m<=_mLim:
                                    nS = [j,_d,_l,_p, m,_rwd,False,i]
                                    _counter+=1
                                    nS = nS+[_counter]+[False]
                                    if _chDom:
                                        if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
                                        elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
                                        elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
                                        else: print("ERROR: WRONG DOM. VER."); return;
                                    else: S[j].append(nS)
                            





#                         if (len(necess_link_dict[i])>0): 
#                             # Goes only necessary link (1-branch)
#                             S[i][w][9] = True # dominance 1-branch
#                             S[i][w][6] = True
#                             reachTo = [x for x in  necess_link_dict[i] if ((x!=i) and (x not in forbid_link_dict[i]))] 
# #                             print("ReachTo:",reachTo)
#                             for j in reachTo:
#                                 _rch_counter+=1
#                                 _d = S[i][w][1] + _Q[j]
#                                 _l = S[i][w][2] + _C[(i,j)]
#                                 _p = S[i][w][3] + 1
#                                 if (_time_window-(_l)>0): # time feasibility
#                                     m_ctc = np.ceil((_d*(_l+_C[(j,0)]))/(_veh_cap))
#                                     m_tw = np.ceil((_l+_C[(j,0)])/(_time_window-_l))
#                                     m = max(m_ctc,m_tw)
#                                     _rwd = S[i][w][5] + _dual[j-1] + S[i][w][4] - m # dual's index doesnt have depot
#                                     if m<=_mLim:
#                                         nS = [j,_d,_l,_p, m,_rwd,False,i]
#                                         _counter+=1
#                                         nS = nS+[_counter]+[True]
#                                         S[j].append(nS)    
                                
#                         else:
#                             # Avoid forbidden link (0-branch)
#                             reachTo = [x for x in range(1,_n+1) if ((x!=i) and (x not in forbid_link_dict[i]))] 
#                             for j in reachTo:
#                                 _rch_counter+=1
#                                 _d = S[i][w][1] + _Q[j]
#                                 _l = S[i][w][2] + _C[(i,j)]
#                                 _p = S[i][w][3] + 1
#                                 if (_time_window-(_l)>0): # time feasibility
#                                     m_ctc = np.ceil((_d*(_l+_C[(j,0)]))/(_veh_cap))
#                                     m_tw = np.ceil((_l+_C[(j,0)])/(_time_window-_l))
#                                     m = max(m_ctc,m_tw)
#                                     _rwd = S[i][w][5] + _dual[j-1] + S[i][w][4] - m # dual's index doesnt have depot
#                                     if m<=_mLim:
#                                         nS = [j,_d,_l,_p, m,_rwd,False,i]
#                                         _counter+=1
#                                         nS = nS+[_counter]+[False]
#                                         if _chDom:
#                                             if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
#                                             elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                             elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
#                                             else: print("ERROR: WRONG DOM. VER."); return;
#                                         else: S[j].append(nS)
#                                 S[i][w][6] = True
                                    
#                                     next_stop = necess_link_dict[j]
#                                     if (len(next_stop)>0) and (nS[3] < _stopLim):
#                                         # Necessary Link (1-branch)
#                                         print("Force reaching from %s to %s"%(j,necess_link_dict[j]))
#                                         print("j:",j)
#                                         print("next_stop:",next_stop)
#                                         print("SHOW Sj:",S[j])
#                                         k = necess_link_dict[j][0]
#                                         print("k:",k)
#                                         _rch_counter+=1; _d = nS[1] + _Q[k];
#                                         _l = nS[2] + _C[(j,k)]; _p = nS[3] + 1;
#                                         if (_p > _stopLim): # exceed stop lim
#                                             print("Exceed stop lim!")
#                                             throw_away_flag = True; break;
#                                         if (_time_window-(_l)>0): # time feasibility
#                                             m_ctc = np.ceil((_d*(_l+_C[(k,0)]))/(_veh_cap))
#                                             m_tw = np.ceil((_l+_C[(k,0)])/(_time_window-_l))
#                                             m = max(m_ctc,m_tw)
#                                             _rwd = nS[5] + _dual[k-1] + nS[4] - m # dual's index doesnt have depot
#                                             if m<=_mLim:
#                                                 nS = [k,_d,_l,_p, m,_rwd,False,j]
#                                                 _counter+=1
#                                                 nS = nS+[_counter]
#                                                 if (len(next_stop)>0) and (nS[3] < _stopLim): nS[6] = True # force link only
#                                                 if _chDom:
#                                                     if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
#                                                     elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                                     elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
#                                                     else: print("ERROR: WRONG DOM. VER."); return;
#                                                 else: S[k].append(nS)
#                                                 print("SHOW Sk:",S[k])
#                                             else: 
#                                                 print("Not demand feasible!")
#                                                 throw_away_flag = True; break;
#                                         else:
#                                             print("Not time feasible!")
#                                             throw_away_flag = True; break;
#                                         print(nS)
#                                         print("==========================")
                        
                        
                        
                        
#                                         while len(next_stop)>0:
#                                             print("Force reaching from %s to %s"%(j,necess_link_dict[j]))
#                                             print("j:",j)
#                                             print("next_stop:",next_stop)
#                                             print("SHOW Sj:",S[j])
#                                             k = necess_link_dict[j][0]
#                                             print("k:",k)
#                                             _rch_counter+=1; _d = nS[1] + _Q[k];
#                                             _l = nS[2] + _C[(j,k)]; _p = nS[3] + 1;
#                                             if (_p > _stopLim): # exceed stop lim
#                                                 print("Exceed stop lim!")
#                                                 throw_away_flag = True; break;
#                                             if (_time_window-(_l)>0): # time feasibility
#                                                 m_ctc = np.ceil((_d*(_l+_C[(k,0)]))/(_veh_cap))
#                                                 m_tw = np.ceil((_l+_C[(k,0)])/(_time_window-_l))
#                                                 m = max(m_ctc,m_tw)
#                                                 _rwd = nS[5] + _dual[k-1] + nS[4] - m # dual's index doesnt have depot
#                                                 if m<=_mLim:
#                                                     nS = [k,_d,_l,_p, m,_rwd,False,j]
#                                                     _counter+=1
#                                                     nS = nS+[_counter]
#                                                     # Check next link
#                                                     j=k
#                                                     next_stop = necess_link_dict[j]
#                                                     if (len(next_stop)>0) and (nS[3] < _stopLim): nS[6] = True # force link only
#                                                     if _chDom:
#                                                         if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
#                                                         elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                                         elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
#                                                         else: print("ERROR: WRONG DOM. VER."); return;
#                                                     else: S[k].append(nS)
#                                                     print("SHOW Sk:",S[k])
#                                                 else: 
#                                                     print("Not demand feasible!")
#                                                     throw_away_flag = True; break;
#                                             else:
#                                                 print("Not time feasible!")
#                                                 throw_away_flag = True; break;
#                                             # Update looping condition
                                            
#                                             print(nS)
                                            
#                                             print("j:",j)
#                                             print("next_stop:",next_stop)
#                                             print("==========================")
#                                         print("S[j]:",S[j])
#                                     print("throw_flag:",throw_away_flag)
#                                     else:
#                                         if not throw_away_flag:
#                                             if _chDom:
#                                                 if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
#                                                 elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                                 elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
#                                                 else: print("ERROR: WRONG DOM. VER."); return;
#                                             else: S[j].append(nS)
#                             S[i][w][6] = True
                            
#                             if (_time_window-(_l)>0): # time feasibility
#                                 m_ctc = np.ceil((_d*(_l+_C[(j,0)]))/(_veh_cap))
#                                 m_tw = np.ceil((_l+_C[(j,0)])/(_time_window-_l))
#                                 m = max(m_ctc,m_tw)
#                                 _rwd = S[i][w][5] + _dual[j-1] + S[i][w][4] - m # dual's index doesnt have depot
#                                 if m<=_mLim:
#                                     nS = [j,_d,_l,_p, m,_rwd,False,i]
#                                     _counter+=1
#                                     nS = nS+[_counter]+[_ab_n]
#                                     if _chDom:
#                                         if _domVer==2:S = checkDominanceTWVer2(nS,S,_mLim)
#                                         elif _domVer==3:S = checkDominanceTWVer3(nS,S,_mLim,_l_threshold)
#                                         elif _domVer==4:S = checkDominanceTWVer4(nS,S,_mLim)
#                                         else: print("ERROR: WRONG DOM. VER."); return;
#                                     else: S[j].append(nS)
#                             S[i][w][6] = True
            #Sort S[i]
            _temp = sorted(S[i], key=itemgetter(5),reverse=True)
            S[i] = _temp
#         print(_counter,[S[i][0] for i in range(_n+1)],"\n")
#         print("S[1]",S[1])
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][6]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break
        if _all_pc: tFlag=True
#     print(S)
    return S,(_counter,_rch_counter)


def checkDominanceTWVer2(_nS,_S,_mpr):
    i = _nS[0]
    adFlag = True
#     print(_S)
    for w in range(len(_S[i])-1,-1,-1):
#         print(l)
        cpS = _S[i][w]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]>=cpS[5])):
#         if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]>cpS[5])):
#             print('Delete:',_S[i][w])
            del _S[i][w]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]<=cpS[5])):
#         elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]<cpS[5])):
            adFlag = False #Throw away _nS
#             print('Throw away:',_nS)
#             print('S[i]:',_S[i])
            break
    if adFlag:
        _S[i].append(_nS)
    #sort _S[i] by reward
    return _S

####Heuristics Dom: Ignore m if l> TH.
def checkDominanceTWVer3(_nS,_S,_mpr,_l_thsd):
    i = _nS[0]; adFlag = True;
    if _nS[2]>=_l_thsd: 
        for w in range(len(_S[i])-1,-1,-1):
            cpS = _S[i][w]
            if cpS[2]>=_l_thsd: #IGNORE M
                if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]>cpS[5])):
                    del _S[i][w]
                elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]<cpS[5])):
                    adFlag = False #Throw away _nS
                    break
            else:
                if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]>=cpS[5])):
                    del _S[i][w]
                elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]<=cpS[5])):
                    adFlag = False #Throw away _nS
                    break
        if adFlag:
            _S[i].append(_nS)
    else:
        for w in range(len(_S[i])-1,-1,-1):
            cpS = _S[i][w]
            if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]>=cpS[5])):
                del _S[i][w]
            elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]==cpS[4])and(_nS[5]<=cpS[5])):
                adFlag = False #Throw away _nS
                break
        if adFlag:
            _S[i].append(_nS)
    return _S

def checkDominanceTWVer4(_nS,_S,_mpr):
    i = _nS[0]; adFlag = True;
    for w in range(len(_S[i])-1,-1,-1):
        cpS = _S[i][w]
#         if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(_nS[4]-cpS[4])>=cpS[5])and(not(cpS[9]))):
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[5]+(_nS[4]-cpS[4])>=cpS[5])):
            del _S[i][w]
#         elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(_nS[4]-cpS[4])<=cpS[5])and(not(_nS[9]))):
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[5]+(_nS[4]-cpS[4])<=cpS[5])):
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    return _S


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
        _cter_idx = 0 # node idx
        _order_idx = 0 # First rank of highest reward
        while (any(_loop_cond)) and (_cter_idx<len(_S)):
#             print(_loop_cond,_cter_idx,_order_idx)
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
                f_list = list(filter(lambda x:(x[3]==(lP-1)) and (np.abs(x[1]-(lD-_Q[lI]))<0.00001) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)), _S[prevN]))
#                 print('f_list:',f_list)
                if len(f_list)==0:
#                     _order_idx+=1
                    throw_away_flag=True;break
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
            if not(throw_away_flag):
                for tup in necess_link:
                    tup = (int(tup[0].split("_")[-1].replace("O","0")),int(tup[1].split("_")[-1].replace("O","0")))
                    if tup[0]==0: # there is (i,j) in route where i!=0
                        if (tup not in _route_arcs) and (tup[1] in _route): 
                            throw_away_flag = True; break
                    elif tup[1]==0: # there is (i,j) in route where j!=0
                        if (tup not in _route_arcs) and (tup[0] in _route): 
                            throw_away_flag = True; break
                    else:
                        if (tup not in _route_arcs):
                            throw_away_flag = True; break
    #              
                    if throw_away_flag:  print("Skipped:",_route,"No arc:",tup,"State:",)#_S[_cter_idx][_order_idx])
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
        
        
#         _bestStList = [_S[i][0] for i in range(len(_S))]
#         for _idx in range(1,len(_bestStList)):
#             _chk_d = dict(zip(one_b,[[0,0]]*len(one_b)))
#             _route = [0]
#             throw_away_flag = False
#             lSt = _bestStList[_idx]
#         #     print(lSt)
#         #     print(sorted(_temp, key=itemgetter(5),reverse=True),lSt)
#             if ((lSt[0]==0)): continue # lSt = sorted(_temp, key=itemgetter(5),reverse=True)[1]
#             lI = lSt[0]
#             lD = lSt[1]
#             lL = lSt[2]
#             lP = lSt[3]
#             lM = lSt[4]
#             prevN = lSt[7]
#             _route = [lI]+_route
#             _bestSt = lSt
#             # chk bch condition
#             if lI in _ab_n: 
#                 for tup in one_b:
#                     if lI in tup:
#                         _chk_d[tup] = [a + b for a, b in zip(_chk_d[tup], _updater[tup.index(lI)])]
#             while lP!=0:
#                 f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and (x[4]<=lM) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.00001)), _S[prevN]))
#         #       print(f_list)
#                 if len(f_list)==0:
#                     throw_away_flag=True;break
#                 else:
#                     if len(f_list)>1:
#                         print("ALERT!:",f_list)
#                         _temp = sorted(f_list, key=itemgetter(5),reverse=True)
#                         f_list = _temp[:1]
#                     lSt = f_list[0]
#                     lI = lSt[0]
#                     lD = lSt[1]
#                     lL = lSt[2]
#                     lP = lSt[3]
#                     lM = lSt[4]
#                     prevN = lSt[7]
#                     _route = [lI]+_route
#                     # chk bch condition
#                     if lI in _ab_n: 
#                         for tup in one_b:
#                             if lI in tup:
#                                 _chk_d[tup] = [a + b for a, b in zip(_chk_d[tup], _updater[tup.index(lI)])]
#     #             print("Route:",_route,throw_away_flag)
#             if not(throw_away_flag):
#                 _route_list.append(_route)
#                 _bestSt_list.append(_bestSt)
#     #           print("RouteList:",_route_list)
#     #     print(_route_list,_bestSt_list)
#     return _route_list,_bestSt_list
        
def prizeCollectingDPwTWVer1(_n,_C,_Q,_M,_dual,_s0,_veh_cap,_time_window,_wavg_factor,_chDom=True,_stopLim=None):
    if _stopLim is None: _stopLim = _n
    _counter = 0; _rch_counter=0
    L0 = [0,0,0,0,-_M,False,None,_counter]
    Linf = [np.inf,np.inf,np.inf,-np.inf,True,None,np.inf]
    S = [[L0]]+[[[x+1]+Linf] for x in range(_n)]
    tFlag = False
    while not(tFlag):
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]):
                    if S[i][l][3] >= _stopLim:
                        S[i][l][5] = True
                    else:
                        reachTo = [x for x in range(1,_n+1) if x!=i] #dont need 0 and i
                        for j in reachTo:
                            _rch_counter+=1
                            _d = S[i][l][1] + _Q[j]
                            _l = S[i][l][2] + _C[(i,j)]
                            _p = S[i][l][3] + 1
                            _rwd = S[i][l][4] + _dual[j-1] #dual's index doesnt have depot
                            nS = [j,_d,_l,_p,_rwd,False,i]
                            cycleFeast = checkFeasibility(nS,_veh_cap,_M,_C)
                            timeWindowFeast = checkFeasibilityTWVer1(nS,_time_window,_wavg_factor,_M,_C)
                            if (cycleFeast and timeWindowFeast) :
                                _counter+=1
                                nS = nS+[_counter]
                                if _chDom: S = checkDominanceTWVer1(nS,S)
                                else: 
#                                     print("NOT CHECKING DOMINANCE")
                                    S[j].append(nS)
                                    temp = sorted(S[j], key=itemgetter(4),reverse=True)
                                    S[j] = temp
                            S[i][l][5] = True
#             print(S)
        _all_pc = True #All processed
        _b_pt = False #Break point
        for i in range(_n+1):
            for l in range(len(S[i])):
                if not(S[i][l][5]): _b_pt=True;break;
            if _b_pt:_all_pc=False;break

        if _all_pc: tFlag=True
    return S,(_counter,_rch_counter)

def checkFeasibilityTWVer1(_nS,_timeWindow,_wavg_factor,_m, _C):
    _rt_cost = _C[(_nS[0],0)]
    _lr_length = _nS[2]+_rt_cost
    _wait_t = _wavg_factor*_lr_length/_m
    _trav_t = _nS[2]
    if (_wait_t+_trav_t) <= _timeWindow: 
#         print('.....TIMEWINDOW FEASIBILITY:','resNode:',_nS[0],'| avg_factor:',_wavg_factor,'| timeWindowLimit:',_timeWindow,'| nsTime:',_wait_t+_trav_t)
        return True
    else: return False
    
def checkDominanceTWVer1(_nS,_S):
    i = _nS[0]
    adFlag = True
#     print(_S)
    for l in range(len(_S[i])-1,-1,-1):
#         print(l)
        cpS = _S[i][l]
        if ((_nS[1]<=cpS[1])and(_nS[2]<=cpS[2])and(_nS[3]<=cpS[3])and(_nS[4]>=cpS[4])):
            del _S[i][l]
        elif ((_nS[1]>=cpS[1])and(_nS[2]>=cpS[2])and(_nS[3]>=cpS[3])and(_nS[4]<=cpS[4])):
            adFlag = False #Throw away _nS
            break
    if adFlag:
        _S[i].append(_nS)
    #sort _S[i] by reward
    _temp = sorted(_S[i], key=itemgetter(4),reverse=True)
    _S[i] = _temp
#     print('AfterDominance:',_S[i])
    return _S

def pathReconstructionTWVer1(_S,_Q,_C):
    _route = [0]
    _temp = [_S[i][0] for i in range(len(_S))]
    lSt = sorted(_temp, key=itemgetter(4),reverse=True)[0]
    print(_temp,lSt)
#     print(sorted(_temp, key=itemgetter(4),reverse=True),lSt)
    if ((lSt[0]==0)): lSt = sorted(_temp, key=itemgetter(4),reverse=True)[1]
    lI = lSt[0]
    lD = lSt[1]
    lL = lSt[2]
    lP = lSt[3]
    prevN = lSt[6]
    _route = [lI]+_route
    _bestSt = lSt
    while lP!=0:
        if prevN is None: break
        f_list = list(filter(lambda x:(x[3]==(lP-1)) and (x[1]==lD-_Q[lI]) and ((np.abs(x[2]-(lL-_C[(prevN,lI)]))<0.000001)), _S[prevN]))
#         print(f_list)
        if len(f_list)>1:
            print("ALERT!:",f_list)
            _temp = sorted(f_list, key=itemgetter(4),reverse=True)
            f_list = _temp[:1]
        lSt = f_list[0]
        lI = lSt[0]
        lD = lSt[1]
        lL = lSt[2]
        lP = lSt[3]
        prevN = lSt[6]
        _route = [lI]+_route
    return _route,_bestSt
        
        
        
#BackUp ver of phaseII runcolgen module:
#     def runColumnsGeneration(self,_m_collections,_pricing_status=False, _check_dominance=True,_dominance_rule=1,_DP_ver=2):
#         print("Dominance Checking:",_check_dominance,', rule:',_dominance_rule)
#         self.solveRelaxedModel()
#         duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
#         opt_cond_vect = pd.Series(index = _m_collections,data=False)
#         out_loop_counter = 0
#         t1 = time.time()
#         iter_log = dict()
#         self.colgenLogs = dict()
#         iter_log['es_time'] = 0
#         iter_log['duals'] = duals_vect[0]
#         iter_log['cols_gen'] = 0
#         iter_log['cols_add'] = 0
#         iter_log['max_stops'] = 0
#         self.colgenLogs[out_loop_counter]=iter_log
#         self.feasibleStatesExplored = 0
#         while opt_cond_vect.sum()<len(_m_collections):
#             iter_log = dict()
#             iter_log['es_time'] = time.time()
#             iter_log['cols_gen'] = 0
#             iter_log['cols_add'] = 0
#             iter_log['max_stops'] = 0
#             proc_list = []
#             for _m_veh in _m_collections:
#                 if _pricing_status:
#                     print('.Running Col. Gen. for m_r:', _m_veh,'| Max nodes visited: %s'%self.max_nodes_proute_DP, '| Out-loop-%s'%out_loop_counter)
#                 if _DP_ver ==1:
#                     print("Incorrect DP version")
# #                     solution_pricing = pricingDP(_m_veh, duals_vect, self,self.max_nodes_proute_DP,_print_status=_pricing_status,_prefix=str(_m_veh)+str(out_loop_counter),_check_dominance=_check_dominance,_dominance_rule=_dominance_rule)
# #                     # len(solution_pricing['reward'])
# #                     (P,prev,reward) = (solution_pricing['P'],solution_pricing['prev'],solution_pricing['reward'])
# #                     #Max stops reached
# #                     if _pricing_status: print(np.max(P.nodeVisited))
# #                     iter_log['max_stops'] = np.max([iter_log['max_stops'],np.max(P.nodeVisited)])
# #                     #Preprocess columns
# #     #                 reward_ss = pd.Series(reward)
# #     #                 col_reward_df = P.loc[:,['colDF','reward']]
# #                     iter_log['cols_gen'] += len(P)
# #     #                 print('P:',P)

# #                     ## Filtering columns
# #                     reward_ss_filtered = P[P.reward>0.00001]
# #                     iter_log['cols_add'] += len(reward_ss_filtered)
# #     #                 print('rwd_positive_cols:',reward_ss_filtered)

# #                     ## Filtering columns
# #                     if len(reward_ss_filtered)>0:
# #                         #Add columns
# # #                         return reward_ss_filtered
# #                         self.generateColumns(reward_ss_filtered, duals_vect)
# #                     else: 
# #                         opt_cond_vect[_m_veh] = True
# #                         continue
#                 elif _DP_ver ==2:
#                     n = len(self.customers)
#                     Q = [0]+list(self.customer_demand.loc[self.customers].values)
# #                     M = 3 #max m per route from collection of m
#                     s0 = self.fixed_setup_time
#                     S,_st_counter = prizeCollectingDP(n,self.cost_matrix,Q,_m_veh,self.getDuals(),s0,
#                                                       _veh_cap=self.vehicle_capacity,
#                                           _chDom=_check_dominance,_stopLim=self.max_nodes_proute_DP)
#                     self.feasibleStatesExplored +=_st_counter
#                     P,bestState = pathReconstruction(S,Q,self.cost_matrix)
            
# #                     print(S)
# #                     return P,bestState,S
#                     reward = bestState[4]
#                     ## Filtering columns
#                     if (reward>0.000001) and (bestState[0]>0):
#                         dual_r = sum([self.getDuals()[i-1] for i in P[1:-1]])
#                         route_cost = -reward+dual_r+_m_veh*self.getDuals()[-1]
#                         prx_route = ['O']+['c_%s'%(x) for x in P[1:-1]]+['O']
#                         arc_route = [(prx_route[i],prx_route[i+1]) for i in range(len(prx_route)-1)]
#                         col_coeff = prx_route+arc_route
#                         nCol = pd.DataFrame(self.init_routes_df.set_index('labels').index, columns=['labels'])
#                         prefix = str(_m_veh)+str(out_loop_counter)
#                         name = 'sDP_C%s-%s'%(prefix,bestState[7])
#                         nCol[name] = 0
#                         nCol.loc[nCol.labels=='m',name] = _m_veh
#                         nCol.loc[nCol.labels=='lr',name] = sum([self.distance_matrix[tup]/self.truck_speed for tup in arc_route])
#                         print(bestState,'M:',_m_veh)
#                         print('PrxRoute:',prx_route,'ArcRoute:',arc_route)
#                         print('Route:',P,'RouteCost:',route_cost,'Reward:',reward)
#                         self.DPRouteDict[name] = prx_route
#                         # nCol
#                         for idx in col_coeff:
#                             nCol.loc[nCol.labels==idx,name] +=1
#                         adColDf = pd.DataFrame(columns=['routeCost','colDF'])
#                         adColDf.loc[name,['routeCost']] =[route_cost]
#                         adColDf.loc[name,['colDF']] = [nCol]
#                         #Add columns
# #                         return nCol,adColDf
#                         self.generateColumns(adColDf, duals_vect)
#                         iter_log['cols_add'] +=1
#                     else: 
#                         opt_cond_vect[_m_veh] = True
#                         continue
#                     tt_states = sum([len(l) for l in S])
#                     iter_log['cols_gen'] += tt_states
                    
                    
                    
                    
#             #Resolve relax model
#             self.solveRelaxedModel()
#             duals_vect = pd.DataFrame(self.getDuals(), index = self.customers + ['m'])
#             out_loop_counter+=1
#             iter_log['es_time'] = time.time()-iter_log['es_time']
#             iter_log['duals'] = duals_vect[0]
#             self.colgenLogs[out_loop_counter]=iter_log
        
#         self.colGenTe = time.time()-t1
#         print('Col.Gen. Completed!...Elapsed-time:',self.colGenTe)
        
        
           
# def pricingDP(_m_veh, _duals, _model_instance,_max_stops=None,_print_status=False,_prefix='',_check_dominance=True,_dominance_rule=1):
#     '''DP Pricing subproblem
#         Input: m_veh:= no. of vehicle using in the route that we're going to search for the best reduced cost
#               duals:= vector of dual variables
#               _max_stops:= no. of max visited. Default is total number of customers.
#         Output: solution_object:=
#                 better_cols = List of columns having better reduced cost
#                node_processed = List of state being processed
#                node_fathomed = List of state being dominated
#     '''
#     if _max_stops is None: _max_stops = len(_model_instance.customers)
#     label_counter = 0
#     #Master colDF Pattern  
#     master_colDF = pd.DataFrame(_model_instance.init_routes_df.loc[:,'labels']);
#     _row_label = master_colDF['labels']
#     master_colDF['M'] = 0; master_colDF.loc[_row_label=='m','M']=_m_veh
    
#     #Initialization
#     _depot_node = _model_instance.depot[0]
#     s_0 = SPPState(_depot_node, 0,0,0,0,master_colDF)
# #     prev = dict(); reward = dict()
# #     prev[s_0] = -1
# #     reward[s_0] = _m_veh*_duals.loc['m',0]

#     s_0.DF.loc[0,'reward'] = _m_veh*_duals.loc['m',0]
#     L = s_0.DF;P = s_0.DF
#     arcs_ss = pd.Series(_model_instance.arcs)
#     return_arcs = arcs_ss[arcs_ss.apply(lambda x: x[-1]==_depot_node)]            
#     return_arcs_cost_dict = pd.Series(index = return_arcs, data = return_arcs.apply(lambda x: _model_instance.distance_matrix[x]/_model_instance.truck_speed).values).to_dict()
#     return_arcs_cost_dict[(_depot_node,_depot_node)] = 0; #boundary condition

#     print('..Starting DP subproblem for m\' =', _m_veh)
#     t1 = time.time()
#     count_dominance = 0
#     dominanceCheckingtime = 0
#     while len(L) != 0:
# #         currState = L.pop(0)
#         currState = L.iloc[:1,:]
#         L = L.iloc[1:,:]
#         if currState.resNode.values == _depot_node: 
#             delta_plus = arcs_ss[arcs_ss.apply(lambda x: x[0]==_depot_node)];
#         else: delta_plus = arcs_ss[arcs_ss.apply(lambda x: x[0]==currState.resNode.values[0])]
#         for arc in delta_plus.to_list():
#             label_counter +=1; 
# #             if _print_state: print(label_counter,currState.resNode, arc)
#             if (arc[-1] == _depot_node) or (currState.nodeVisited.values >= _max_stops): continue
#             else:
#                 ext_node = arc[-1]
#                 curr_node = arc[0]
#                 return_cost_ext = return_arcs_cost_dict[(ext_node,_depot_node)]
#                 return_cost_curr = return_arcs_cost_dict[(curr_node,_depot_node)]
#                 ext_col = pd.DataFrame(currState.colDF.values[0].labels)
#                 ext_col['sDP_C%s-%s'%(_prefix,label_counter)] = currState.colDF.values[0].iloc[:,-1]
#                 ext_col.loc[currState.colDF.values[0].labels==ext_node,'sDP_C%s-%s'%(_prefix,label_counter)]+=1 #Node visited
#                 ext_col.loc[currState.colDF.values[0].labels.isin(return_arcs),'sDP_C%s-%s'%(_prefix,label_counter)] = 0 #Deleted previous return arc
#                 ext_col.loc[currState.colDF.values[0].labels.isin([arc,(ext_node,_depot_node)]),'sDP_C%s-%s'%(_prefix,label_counter)]+=1 #Arc passed
                
#             extStateOBJ = SPPState(ext_node, currState.accDemand.values[0]+_model_instance.customer_demand[ext_node], 
#                                currState.accDistance.values[0]+(_model_instance.distance_matrix[arc]/_model_instance.truck_speed), 
#                                 currState.nodeVisited.values[0]+1,label_counter,ext_col)
#             extState = extStateOBJ.DF
            
#             feas_flag = extStateOBJ.checkFeasibility(_model_instance.vehicle_capacity,_m_veh,return_cost_ext,_print_status=_print_status)
#             if feas_flag:
#                 #Can be improved!
#                 transition_reward = transitionReward(currState,extState,_duals,_m_veh,_model_instance,
#                                                     return_cost_curr,return_cost_ext,_print_status=_print_status)
# #                 reward_ext = reward[currState]+transition_reward
#                 reward_ext = currState.reward.values[0]+transition_reward
#                 extState.loc[label_counter,'reward'] = reward_ext
#                 #Calculate route cost
#                 duals_idx = pd.Series(_duals.index)
#                 duals_coeff = duals_idx.apply(lambda x:ext_col.loc[_row_label==x].iloc[:,-1].values[0]*_duals.loc[x].values[0])
#                 route_cost = duals_coeff.sum()-reward_ext
# #                 extState.route_cost = route_cost
#                 extState.loc[label_counter,'routeCost'] = route_cost
#                 #Check Dominance (if true: add to unproc list)
#                 if _check_dominance: 
#                     comparing_index = _row_label.isin(_model_instance.customers)
# #                     print('comp indx',comparing_index)
#                     t2 = time.time()
#                     if _dominance_rule==1:
#                         added_flag,L = updateDominance(extState, L,comparing_index,_print_status = _print_status,_rule=_dominance_rule)
#                     elif _dominance_rule==3:
#                         added_flag,L,P = updateDominanceV3(extState,L,P,_print_status = _print_status)
#                     dominanceCheckingtime+= time.time()-t2
# #                     print(L)
#                 else: 
#                     added_flag = True
#                     L = L.append(extState)
#                 L.sort_values(by = ['nodeVisited'],inplace=True)
                
# #                 if added_flag:
# # #                     prev[extState] = currState
# # #                     reward[extState] = reward_ext
# #                 else:
#                 if added_flag==False: count_dominance+=1
            
#             if _print_status: 
# #                 print(currState.reward)
#                 print('.....REACHING:','currS: %s-%s'%(currState.resNode,currState.index.values[0]),'| reward:',round(currState.reward.values[0],2),'| transReward:',round(transition_reward,2))
#                 if (feas_flag and added_flag): print('\t \t nextS: %s-%s'%(extState.resNode,extState.index.values[0]),'| reward:',round(extState.reward.values[0],2))
#                 else: print('\t \t nextS: %s-%s'%(extState.resNode,extState.index.values[0]),'Infeasible!')
#                 print('\t \t Unproc-size:',len(L))
#         P = P.append(currState)
# #         print(P) 
#     solution_obj = dict(zip(['P','prev','reward'],[P,None,None]))
#     print('No. dominated states:',count_dominance)
#     print('Elapsed-time:',time.time()-t1)
#     print('Dominance-checking time:',dominanceCheckingtime)
#     return solution_obj

# class SPPState:
#     def __init__(self, _resNode, _accDemand, _accDistance, _nodeVisited,_label,_colDF):
#         _c_name = ['resNode', 'accDemand', 'accDistance', 'nodeVisited','routeCost','colDF','reward']
#         self.DF = pd.DataFrame([_resNode, _accDemand, _accDistance, _nodeVisited,np.inf,_colDF,0],index=_c_name, columns=[_label]).transpose()
# #         self.resNode = _resNode
# #         self.accDemand = _accDemand
# #         self.accDistance = _accDistance
# #         self.nodeVisited = _nodeVisited
#         self.label = _label
# #         self.colDF = _colDF
# #         self.route_cost = np.inf
#         #Create DataFrame to collect State
        
#     def checkFeasibility(self, _total_demand, _max_vehicle_per_route, _return_cost,_print_status=False):
#         deliver_cap = self.DF.accDemand.values[0]*(self.DF.accDistance.values[0]+_return_cost)
#         limit_cap = _total_demand*_max_vehicle_per_route
#         if _print_status : print('.....FEASIBILITY:','resNode:',self.DF.resNode.values[0],'| deliver_cap:',round(deliver_cap,2),'| cap_limit:',round(limit_cap,2))
#         if deliver_cap <= limit_cap: return True
#         else: return False


# def updateDominanceV3(_cur_s,_unproc_list,_proc_list,_print_status=False):
#     _u_del_idx = []
#     _p_del_idx = []
#     _added_flag = True
# #     print(_cur_s.iloc[0,:].loc['resNode'])
#     curr_resNode = _cur_s.iloc[0,:].loc['resNode']
#     curr_accDem = _cur_s.iloc[0,:].loc['accDemand']
#     curr_accDist = _cur_s.iloc[0,:].loc['accDistance']
#     curr_nodeVisited = _cur_s.iloc[0,:].loc['nodeVisited']
#     curr_reward = _cur_s.iloc[0,:].loc['reward']
# #     print("curr-i/d/l/p/rwd:",curr_resNode,curr_accDem,curr_accDist,curr_nodeVisited,curr_reward)
#     for i,r in _unproc_list.iterrows():
#         A_resNode = r.resNode
#         A_accDem = r.accDemand
#         A_accDist = r.accDistance
#         A_nodeVisited = r.nodeVisited
#         A_reward = r.reward
#         if (A_resNode==curr_resNode):
#             if (curr_accDem<=A_accDem and curr_accDist<=A_accDist and curr_nodeVisited<=A_nodeVisited and curr_reward>A_reward):
#                 _u_del_idx.append(i)
# #                 print("u_dominated-",i,"-i/d/l/p/rwd:",A_resNode,A_accDem,A_accDist,A_nodeVisited,A_reward)
#             elif (curr_accDem>=A_accDem and curr_accDist>=A_accDist and curr_nodeVisited>=A_nodeVisited and curr_reward<A_reward):
#                 _added_flag = False
#                 break
# #     print(_proc_list,_unproc_list)
# #     print(len(_proc_list),len(_unproc_list))
#     for j,r in _proc_list.iterrows():
# #         print(r.accDemand)
#         B_resNode = r.resNode
#         B_accDem = r.accDemand
#         B_accDist = r.accDistance
#         B_nodeVisited = r.nodeVisited
#         B_reward = r.reward
#         if (B_resNode==curr_resNode):
#             if (curr_accDem<=B_accDem and curr_accDist<=B_accDist and curr_nodeVisited<=B_nodeVisited and curr_reward>B_reward):
#                 _p_del_idx.append(j)
# #                 print("p_dominated-",j,"i/d/l/p/rwd:",B_resNode,B_accDem,B_accDist,B_nodeVisited,B_reward)
#             elif (curr_accDem>=B_accDem and curr_accDist>=B_accDist and curr_nodeVisited>=B_nodeVisited and curr_reward<B_reward):
#                 _added_flag = False
#                 break
# #     print(len(_unproc_list))
#     _unproc_list = _unproc_list.drop(index = _u_del_idx)
# #     print(len(_unproc_list))
#     _proc_list = _proc_list.drop(index = _p_del_idx)
#     if _added_flag: _unproc_list = _unproc_list.append(_cur_s)
#     return _added_flag,_unproc_list,_proc_list





# def updateDominanceV2(_cur_s, _unproc_list,_comparing_index,_print_status=False,_rule=1):
# #     _re_comp = _unproc_list.apply(lambda x: compareVisitedNodes(x,_cur_s.iloc[0],_comparing_index),axis=1)
#     _unproc_list['rst_comp'] = None
#     for index,row in _unproc_list.iterrows():
#         _unproc_list.loc[index,'rst_comp'] = compareVisitedNodes(row,_cur_s.iloc[0],_comparing_index)
#     _re_comp = _unproc_list['rst_comp']
#     if _print_status: print('LEN_UNProc:',len(_unproc_list))
#     if np.any(_re_comp)==False:
#         _unproc_list=_unproc_list.append(_cur_s)
#         if _print_status: print('DOMINANCE CHECK: No same partition. CurState is added.')
#         _added_flag = True
#     else:
#         _same_partition = _unproc_list[_re_comp]
#         _dominating_flag = _same_partition.routeCost>=_cur_s.routeCost.values[0]
#         if np.any(_dominating_flag)==False: #_x is useless
#             if _print_status: print('DOMINANCE CHECK: CurState is dominated, not added.')
#             _added_flag = False
#         else:#_x is added anyway,
#             _added_flag = True
#             _comp_flag = _same_partition.routeCost>_cur_s.routeCost.values[0]
#             if _print_status: print('Dominated/dropping:',len(_comp_flag))
#             _unproc_list.drop(index = _comp_flag.index,inplace=True)
#             _unproc_list = _unproc_list.append(_cur_s)
#             if len(_dominating_flag)>0: 
#                 if _print_status: print('DOMINANCE CHECK: CurState is dominating and added. Dominated unprocStates are removed')
#             else: 
#                 if _print_status: print('DOMINANCE CHECK: There is a tie! CurState is added.')
#     if _print_status: print('Update LEN_UNProc',len(_unproc_list),'\n===========')
#     return _added_flag,_unproc_list

# def updateDominance(_cur_s, _unproc_list,_comparing_index,_print_status=False,_rule=1):
#     _re_comp = _unproc_list.apply(lambda x: compareVisitedNodes(x,_cur_s.iloc[0],_comparing_index),axis=1)
# #     if _print_status: print('LEN_UNProc:',len(_unproc_list))
#     if np.any(_re_comp)==False:
#         _unproc_list=_unproc_list.append(_cur_s)
# #         if _print_status: print('DOMINANCE CHECK: No same partition. CurState is added.')
#         _added_flag = True
#     else:
#         _same_partition = _unproc_list[_re_comp]
#         _dominating_flag = _same_partition.routeCost>=_cur_s.routeCost.values[0]
#         if np.any(_dominating_flag)==False: #_x is useless
# #             if _print_status: print('DOMINANCE CHECK: CurState is dominated, not added.')
#             _added_flag = False
#         else:#_x is added anyway,
#             _added_flag = True
#             _comp_flag = _same_partition.routeCost>_cur_s.routeCost.values[0]
# #             if _print_status: print('Dominated/dropping:',len(_comp_flag))
#             _unproc_list.drop(index = _comp_flag.index,inplace=True)
#             _unproc_list = _unproc_list.append(_cur_s)
# #             if len(_dominating_flag)>0: 
# #                 if _print_status: print('DOMINANCE CHECK: CurState is dominating and added. Dominated unprocStates are removed')
# #             else: 
# #                 if _print_status: print('DOMINANCE CHECK: There is a tie! CurState is added.')
# #     if _print_status: print('Update LEN_UNProc',len(_unproc_list),'\n===========')
#     return _added_flag,_unproc_list


# def compareVisitedNodes(_ps_r1,_ps_r2,_comparing_labels,_rule=1):
#     cp_val1 = _ps_r1.colDF.loc[_comparing_labels].iloc[:,-1]
#     cp_val2 = _ps_r2.colDF.loc[_comparing_labels].iloc[:,-1]
# #     print('cp_val1:',cp_val1)
# #     print('cp_val2:',cp_val2)
#     if _rule==1:
#         e_wise_comp = np.all((cp_val1>0)==(cp_val2>0))
#     elif _rule==2:
#         e_wise_comp = np.all(cp_val1==cp_val2)
#     return e_wise_comp

# def transitionReward(_current_state, _next_state, _duals, _m_veh, _model_instance, _return_cost_i,_return_cost_j,_print_status=False):
# #     if _current_state.resNode == depot_s[0]:
# #         arc_ij = tuple([_current_state.resNode.replace('_s',''), _next_state.resNode])
# #         cost_i0 = 0
# #     elif _next_state.resNode == depot_t[0]:return 0
# #     else: 
#     next_accDem = _next_state.accDemand.values[0]
#     next_accDist = _next_state.accDistance.values[0]
#     next_resNode = _next_state.resNode.values[0]
#     curr_accDem = _current_state.accDemand.values[0]
#     curr_accDist = _current_state.accDistance.values[0]
#     curr_resNode = _current_state.resNode.values[0]
    
#     arc_ij = tuple([curr_resNode, next_resNode])
#     cost_i0 = _return_cost_i
#     pi_next = _duals.loc[next_resNode][0]
#     cost_ij = _model_instance.distance_matrix[arc_ij]/_model_instance.truck_speed
#     cost_j0 = _return_cost_j
#     if _print_status:
#         print('.....TRANSITION:','pi_next:',pi_next,'| cost_ij:',round(cost_ij,2),'| cost_j0:',round(cost_j0,2),'| cost_i0:',round(cost_i0,2))
#         print('\t \t j_cus_dem:',_model_instance.customer_demand[next_resNode],
#               '| next_s.accDem:',round(next_accDem,2),
#               '| next_s.accDist:',round(next_accDist,2),
#               '| curr_s.accDem:',round(curr_accDem,2),
#               '| curr_s.accDist:',round(curr_accDist,2))
#     transition_reward = pi_next \
#             - (_model_instance.customer_demand[next_resNode]*(curr_accDist+cost_ij)) \
#             - ((0.5/_m_veh)*(next_accDem)*(curr_accDist+cost_ij+cost_j0+_model_instance.fixed_setup_time))\
#             + ((0.5/_m_veh)*(curr_accDem)*(curr_accDist+cost_i0+_model_instance.fixed_setup_time))
# #     transition_reward = pi_next \
# #             - (_model_instance.customer_demand[_next_state.DF.resNode.values[0]]*(_current_state.DF.accDistance.values[0]+cost_ij)) \
# #             - ((0.5/_m_veh)*(_next_state.accDemand.values[0])*(_current_state.accDistance.values[0]+cost_ij+cost_j0+_model_instance.fixed_setup_time))\
# #             + ((0.5/_m_veh)*(_current_state.accDemand.values[0])*(_current_state.accDistance.values[0]+cost_i0+_model_instance.fixed_setup_time))
#     return transition_reward



    


# def reconstructPath(_last_state, _prev_dict,_str_node):
#     curr_state = _last_state
#     path = [curr_state.resNode]
#     while curr_state.resNode !=_str_node:
# #         print('currNode:',curr_state.resNode )
#         prev_state = _prev_dict[curr_state]
#         path = [prev_state.resNode] + path
#         curr_state = prev_state
# #         print('nextNode:',curr_state.resNode )
#     return path


# class VRPdMasterProblem:
#     def __init__(self, path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_drone,no_truck,_truck_distance,_drone_distance,constant_dict,extra_constr=None):
# #                  ,truck_count,drone_carried_by_truck,transportation_truck_cost,transportation_drone_cost):
                
#         self.NoTruckPerPath = no_truck
#         self.PathCoeff = path['PathCoeff'].values
#         self.Path = path.copy()
#         self.TruckCusNodes = truck_cus_nodes
#         self.TruckCusNodesDict = truck_cus_nodes_dict
#         self.DroneCusNodes = drone_cus_nodes
#         self.Nodes = all_depot+truck_cus_nodes+drone_cus_nodes+docking
#         self.Docking = docking
#         self.AllDepot = all_depot
#         self.depot_s = [all_depot[0]]
#         self.depot_t = [all_depot[1]]
#         self.depot = ['depot']
        
#         self.TruckArcs = arcs_truck
#         self.DroneArcs = arcs_drone
#         self.Arcs = arcs_truck+arcs_drone
        
#         self.CoeffSeries = pd.Series(all_depot+truck_cus_nodes+drone_cus_nodes+docking+self.Arcs)
#         self.DepotIndex = self.CoeffSeries[self.CoeffSeries.isin(all_depot)].index.values
#         self.DepotSIndex = self.CoeffSeries[self.CoeffSeries.isin([all_depot[0]])].index.values
#         self.DepotTIndex = self.CoeffSeries[self.CoeffSeries.isin([all_depot[1]])].index.values
#         self.TruckCusIndex = self.CoeffSeries[self.CoeffSeries.isin(truck_cus_nodes)].index.values
#         self.TruckCusIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCusIndexDict[i+1] = self.CoeffSeries[self.CoeffSeries.isin(self.TruckCusNodesDict[i+1])].index.values
        
#         self.DroneCusIndex = self.CoeffSeries[self.CoeffSeries.isin(drone_cus_nodes)].index.values
        
#         self.AllCustomerIndex = self.CoeffSeries[self.CoeffSeries.isin(truck_cus_nodes+drone_cus_nodes)].index.values
#         self.DockingIndex = self.CoeffSeries[self.CoeffSeries.isin(docking)].index.values
#         self.NodesIndex = self.CoeffSeries[self.CoeffSeries.isin(self.Nodes)].index.values
        
#         self.ArcsIndex = self.CoeffSeries[self.CoeffSeries.str.contains(',')].index.values
    
#         self.TruckArcsIndex = self.CoeffSeries[self.CoeffSeries.isin(self.TruckArcs)].index.values
#         self.DroneArcsIndex = self.CoeffSeries[self.CoeffSeries.isin(self.DroneArcs)].index.values
        
        
#         self.K = constant_dict['truck_count']
#         self.Lr = constant_dict['drone_carried_by_truck']
#         self.transportation_truck_cost=constant_dict['transportation_truck_cost']
#         self.transportation_drone_cost=constant_dict['transportation_drone_cost']
#         self.truck_speed = constant_dict['truck_speed']
#         self.drone_speed = constant_dict['drone_speed']
#         self.truck_fix_cost = constant_dict['truck_fix_cost']
#         self.constant_dict = constant_dict
        
#         self.TruckDistance = _truck_distance
#         self.DroneDistance = _drone_distance

        
#         self.model = Model("MasterProblem")
#         self.PathIndex = pd.Series(path.index).index.values
#         if extra_constr is not None: self.K = extra_constr 

        
#     def buildModel(self):
#         self.generateVariables()
#         self.generateConstraints()
#         self.generateObjective()
#         self.model.update()
        
#     def convertDroneArc2TruckArcIndex(self,d_a_idx):
#         arc_name_split = self.CoeffSeries[d_a_idx].split(',')
#         t_arc_name = ','.join(arc_name_split[:2]+['T'])
#         t_a_idx = self.CoeffSeries[self.CoeffSeries==t_arc_name].index.values
#         return t_a_idx[0]
    
#     def mergeDepotCusArcsVar(self,a_var):
#         '''INPUT: ['depot_s,customer_1_T1,T1','customer_2_T1,depot_t,T1']
#         OUTPUT: ['depot,customer_1,T1','customer_2,depot,T1']'''
#         new_a_var =[]
#         for a in a_var:
#             v = a.split(',')
#             if len(v)>1:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0]=self.depot[0]
#                 elif v[0] in self.TruckCusNodes+self.DroneCusNodes: v[0]='_'.join(v[0].split('_')[:-1])
#                 if v[1]==self.depot_s[0] or v[1]==self.depot_t[0]:v[1]=self.depot[0]
#                 elif (v[1] in self.TruckCusNodes+self.DroneCusNodes): v[1]='_'.join(v[1].split('_')[:-1])   
#             else:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0] = self.depot[0]
#             new_a_var.append(','.join(v))
#         return new_a_var
    
#     def generateRouteCost(self,index):
#         def cost_truck_arc(a_idx):
#             a_name = self.mergeDepotCusArcsVar([self.CoeffSeries[a_idx]])[0]
#             return self.TruckDistance[a_name]
#         def cost_drone_arc(a_idx):
#             a_name = self.mergeDepotCusArcsVar([self.CoeffSeries[a_idx]])[0]
#             return self.DroneDistance[a_name]
# #         print(cost_truck_arc)
#         path_cost = quicksum(self.Path.iloc[index][0][a]*cost_truck_arc(a)*self.transportation_truck_cost*60/self.truck_speed for a in self.TruckArcsIndex)\
#                     +quicksum(self.Path.iloc[index][0][a]*cost_drone_arc(a)*self.transportation_drone_cost*60/self.drone_speed for a in self.DroneArcsIndex)
# #         print(path_cost)
#         return path_cost
    
#     def getBothTypeCustomerIndex(self, index):
#         node_name = self.CoeffSeries[index]
#         customer_name = '_'.join(node_name.split('_')[:-1])+'_'
# #         print(node_name,customer_name)
#         truck_node_index = self.CoeffSeries[self.CoeffSeries.isin(self.TruckCusNodes)&self.CoeffSeries.str.contains(customer_name)].index
# #         print(truck_node_index)
#         drone_node_index = self.CoeffSeries[self.CoeffSeries.isin(self.DroneCusNodes)&self.CoeffSeries.str.contains(node_name[:-2])].index
#         return [list(truck_node_index),drone_node_index[0]]
    
#     def generateArcsIndexFromNode(self,vech_type,node_i_idx=None,node_j_idx=None):
#         list_of_arcs_index = []
#         if (node_i_idx is not None): node_i_name = [self.CoeffSeries[node_i_idx]] 
#         else: node_i_name = self.Nodes
#         if node_j_idx is not None: node_j_name = [self.CoeffSeries[node_j_idx]] 
#         else: node_j_name = self.Nodes
#         if 'T' in vech_type:
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         '''If vech_type=T: find all arcs from all truck'''
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.CoeffSeries[self.CoeffSeries.str.contains(arcs_name)].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)>1: list_of_arcs_index+=list(np.reshape(arcs_index,(len(arcs_index),-1)))
#                         elif len(arcs_index)!=0: list_of_arcs_index.append(arcs_index)
        
#         elif vech_type =='D':
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.CoeffSeries[self.CoeffSeries==arcs_name].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)!=0: list_of_arcs_index.append(arcs_index) 
#         return list_of_arcs_index
    
#     def generateVariables(self):
#         self.path = self.model.addVars(self.PathIndex, lb=0,
#                                        vtype=GRB.BINARY, name='path')
#         print('Finish generating variables!')
        
#     def generateConstraints(self):
#         # Fix number of vehicles used (truck+total_drone)
#         const0 = ( quicksum(self.PathCoeff[rt][i]*self.path[rt] for rt in self.PathIndex)<=(self.K+self.K*self.Lr) \
#          for i in self.DepotIndex)
#         self.model.addConstrs(const0,name='Depot')
                  
#         const1 = ( quicksum(( quicksum(self.PathCoeff[rt][self.getBothTypeCustomerIndex(i)[0][j]] for j in range(self.NoTruckPerPath))  \
#                              +self.PathCoeff[rt][self.getBothTypeCustomerIndex(i)[1]])*self.path[rt] for rt in self.PathIndex)>=1 \
#                              for i in self.TruckCusIndexDict[1] )
#         self.model.addConstrs( const1,name='customer_coverage' )

# #         const2 = ( quicksum(self.path[rt] for rt in self.PathIndex)<= self.K )
        
#         const2 = ( quicksum(self.path[rt]*quicksum(self.PathCoeff[rt][a_idx[0]] for a_idx in self.generateArcsIndexFromNode('T',node_i_idx=self.DepotSIndex[0]) ) \
#                             for rt in self.PathIndex)<= self.K )
#         self.model.addConstr( const2,name='truck_number' )
#         print('Finish generating constrains!')
    
#     def getTruckNumberFromRoute(self,path_name,show_value=False):
#         '''path_name: name of the var'''
#         truck_no = quicksum(self.Path.loc[path_name][0][a_idx[0]] for a_idx in self.generateArcsIndexFromNode('T',node_i_idx=self.DepotSIndex[0]) )
#         if show_value: print('Truck no. of path',path_name,':', np.round(truck_no.getValue()))
#         return  np.round(truck_no.getValue())
    
#     def getDroneNumberFromRoute(self,path_name):
#         '''path_name: name of the var'''
#         drone_arcs_name = pd.Series(self.DroneArcs)
#         drone_departing_arc = drone_arcs_name[drone_arcs_name.str.contains('_T')|drone_arcs_name.str.contains('depot_s')|drone_arcs_name.str.contains(r'dock_.*?,cus')]
#         drone_departing_arc_index = self.CoeffSeries[self.CoeffSeries.isin(drone_departing_arc.values)].index
#         drone_no = quicksum(self.Path.loc[path_name][0][a_idx] for a_idx in drone_departing_arc_index)
#         print('Drone no. of path',path_name,':', np.round(drone_no.getValue()))
#         return  np.round(drone_no.getValue())
#     def getNumberCustomerServedByTruck(self,path_name):
#         path_sol = pd.Series(self.Path.loc[path_name][0],index=self.CoeffSeries)
#         served_truck = path_sol[(path_sol.index.isin(self.TruckCusNodes))&(path_sol>0)]
#         return served_truck
    
#     def getNumberCustomerServedByDrone(self,path_name):
#         path_sol = pd.Series(self.Path.loc[path_name][0],index=self.CoeffSeries)
#         served_drone = path_sol[(path_sol.index.isin(self.DroneCusNodes))&(path_sol>0)]
#         return served_drone
        
#     def generateObjective(self):
#         # Minimize the total cost of the used rolls
#         self.model.setObjective( quicksum(self.path[rt]*(self.generateRouteCost(rt)) for rt in self.PathIndex) ,
#                                 sense=GRB.MINIMIZE)
       
#         print('Finish generating objective!')
#     #################################
#     def solveRelaxedModel(self):
#         #Relax integer variables to continous variables
#         self.relaxedModel = self.model.relax()
#         self.relaxedModel.optimize()
        
#     def getRelaxSolution(self):
#         a = pd.Series(self.relaxedModel.getAttr('X'))
#         return a[a>0]

#     def getDuals(self):
#         return self.relaxedModel.getAttr('Pi',self.model.getConstrs())
#     #################################
#     def addColumn(self, objective,newPath, get_var,_pathname):
# #         ctName = ('path[%s]'%len(self.model.getVars()))
#         ctName = _pathname
#         newPathSeries = pd.Series(newPath,index = self.CoeffSeries)
#         LHS_Coeff =list()
#         for i in self.TruckCusIndexDict[1]:
#             d_idx = self.getBothTypeCustomerIndex(i)[1]
#             theta_d_var_name = 'theta['+str(d_idx)+']'
# #             print(theta_d_var_name)
# #             print(i)
#             t_idx = self.getBothTypeCustomerIndex(i)[0]
#             truck_passed_var_name = ['alpha'+str(j+1)+'['+str(t_idx[j])+']' for j in range(len(t_idx))]
#             truck_released_var_name = ['alpha'+str(j+1)+'[0]' for j in range(len(t_idx))]
# #             print(truck_passed_var_name)
#             LHS_Coeff.append(get_var(theta_d_var_name).X+quicksum(get_var(t_var_name).X for t_var_name in truck_passed_var_name).getValue())
# #         new_col = [1]*len(self.DepotIndex)+LHS_Coeff+[1]
#         new_col = [1]*len(self.DepotIndex)+LHS_Coeff+[quicksum(get_var(t_var_name).X for t_var_name in truck_released_var_name).getValue()]
# #             t_idx = self.getBothTypeCustomerIndex(i)[0]
# #             truck_passed_var_name = ['alpha'+str(j+1)+'['+str(t_idx[j])+']' for j in range(len(t_idx))]
# # #             print(truck_passed_var_name)
# #             LHS_Coeff.append(get_var(theta_d_var_name).X+quicksum(get_var(t_var_name).X for t_var_name in truck_passed_var_name).getValue())
# #         new_col = [1]*len(self.DepotIndex)+LHS_Coeff+[1]
#         newColumn = Column(new_col, self.model.getConstrs())
# #         print(new_col)
#         print(ctName,new_col)
#         print("NewCol:",newColumn)
# #         self.Path = self.Path.append({'PathCoeff':newPathSeries.values},ignore_index=True)
# #         print(newPathSeries.values)
#         self.Path.loc[ctName] = [newPathSeries.values]
#         self.model.addVar(vtype = GRB.BINARY, lb=0, obj=objective, column=newColumn,name= ctName)
#         self.model.update()
#     #################################    
#     def solveModel(self, timeLimit = None,GAP=None):
#         self.model.setParam('SolutionNumber',2)
#         self.model.setParam(GRB.Param.PoolSearchMode, 2)
#         self.model.setParam('TImeLimit', timeLimit)
#         self.model.setParam('MIPGap',GAP)
#         self.model.optimize()
#     #########EXPLICIT COL_GEN###############
#     def addColumnExplicitly(self, newCol):
#         for ncol,cost in newCol:
#             ctName = ('path[%s]'%ncol)
#             newPathSeries = pd.Series(x[x.columns[ncol]],index = self.CoeffSeries)
#             new_col = list(newPathSeries[newPathSeries.index.isin(self.depot+customers)].values)+[1]
#             print(new_col)
#             newColumn = Column(new_col, self.model.getConstrs())
#             self.model.addVar(vtype = GRB.BINARY, lb=0, obj=cost, column=newColumn,name= ctName)
#             self.model.update()

            
# ############################################################################################           
# ############################################################################################           
            

# class VRPdPricingProblem:
#     def __init__(self, all_depot, truck_cus_nodes, truck_cus_nodes_dict, drone_cus_nodes,docking, arcs_truck,arcs_truck_dict, arcs_drone, duals, customer_demand, no_truck, _truck_distance, _drone_distance, constant_dict, all_indices=None, processed_indices=None):
# #truck_count,drone_carried_by_truck,max_capacity_truck,max_distance_drone, max_weight_drone,
#         self.NoTruckPerPath = no_truck
#         # SET OF NODES AND ARCS
#         self.AllDepot = all_depot
#         self.Nodes = all_depot+truck_cus_nodes+drone_cus_nodes+docking
#         self.TruckCusNodes = truck_cus_nodes
#         self.TruckCusNodesDict = truck_cus_nodes_dict
#         self.AllDepot = all_depot
#         self.depot_s = [all_depot[0]]
#         self.depot_t = [all_depot[1]]
#         self.depot = ['depot']
        
#         self.DroneCusNodes = drone_cus_nodes
#         self.Docking = docking
#         self.TruckArcs = arcs_truck
#         self.TruckArcsDict = arcs_truck_dict
        
#         self.DroneArcs = arcs_drone
#         self.Arcs = arcs_truck+arcs_drone
#         self.PathConstructSeries = pd.Series(all_depot+truck_cus_nodes+drone_cus_nodes+docking+self.Arcs)
        
#         # INDEXING OF NODES
#         self.NodesIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.Nodes)].index.values
#         self.TruckCusIndex = self.PathConstructSeries[self.PathConstructSeries.isin(truck_cus_nodes)].index.values
#         self.TruckCusIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCusIndexDict[i+1] = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckCusNodesDict[i+1])].index.values
        
#         self.DroneCusIndex = self.PathConstructSeries[self.PathConstructSeries.isin(drone_cus_nodes)].index.values
#         self.AllCustomerIndex = self.PathConstructSeries[self.PathConstructSeries.isin(truck_cus_nodes+drone_cus_nodes)].index.values
#         self.DepotAllIndex = self.PathConstructSeries[self.PathConstructSeries.isin(all_depot)].index.values
#         self.DepotSIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.depot_s)].index.values
#         self.DepotTIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.depot_t)].index.values
#         self.DockingIndex = self.PathConstructSeries[self.PathConstructSeries.isin(docking)].index.values
#         # INDEXING OF ARCS
#         self.ArcsIndex = self.PathConstructSeries[self.PathConstructSeries.str.contains(',')].index.values
#         self.TruckArcsIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckArcs)].index.values
#         self.TruckArcsIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckArcsIndexDict[i+1] = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckArcsDict[i+1])].index.values
        
#         self.DroneArcsIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.DroneArcs)].index.values
#         # OTHER PARAMETERS
#         self.CustomerDemand = pd.Series(customer_demand,index=self.NodesIndex)
#         self.TruckDistance = _truck_distance
#         self.DroneDistance = _drone_distance
#         self.transportation_truck_cost=constant_dict['transportation_truck_cost']
#         self.transportation_drone_cost=constant_dict['transportation_drone_cost']
#         self.truck_speed = constant_dict['truck_speed']
#         self.drone_speed = constant_dict['drone_speed']
#         self.K = constant_dict['truck_count']
#         self.Lr = constant_dict['drone_carried_by_truck']
#         self.Ld = constant_dict['max_weight_drone']
#         self.Lt = constant_dict['max_capacity_truck']
#         self.Dd = constant_dict['max_distance_drone']
#         self.UbAmount = 1+self.Lr
#         self.M = 1000000
#         # DUALS
#         self.Duals = duals
#         # DECLARE MODEL
#         self.model = Model('Subproblem')
#         self.PathConstructIndex = self.PathConstructSeries.index.values
    
#     def buildModel(self):
#         self.generateVariables()
#         self.generateConstrains()
#         self.generateObjective(self.Duals)
#         self.model.update()
        
#     def generateVariables(self):
#         self.NodesDelta = self.model.addVars(self.NodesIndex, lb=0, ub=self.UbAmount,\
#                                                         vtype=GRB.INTEGER, name="theta")
#         self.TruckArcs = self.model.addVars(self.TruckArcsIndex,\
#                                                         vtype=GRB.BINARY, name='truck_arcs')
        
#         self.DroneArcs = self.model.addVars(self.DroneArcsIndex,\
#                                                         vtype=GRB.BINARY, name='drone_arcs')
            
#         self.DroneCumulativeDemand = self.model.addVars(self.NodesIndex,vtype=GRB.CONTINUOUS,\
#                                                         lb=0,ub=self.CustomerDemand.sum(),name='z')
        
#         self.DroneCumulativeDistance = self.model.addVars(self.NodesIndex,vtype=GRB.CONTINUOUS,\
#                                                         lb=0,ub=np.sum(list(self.DroneDistance.values())),name='v')

#         self.TruckCumulativeDemand = dict()
#         self.TruckPassedNode = dict()
#         self.TruckCumulativeDistance = dict()
#         self.DecisionIfThen = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCumulativeDemand[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.CONTINUOUS,lb=0,ub=self.CustomerDemand.sum(),\
#                                                         name='y'+str(i+1))
#             self.TruckPassedNode[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.BINARY,name = 'alpha'+str(i+1))
        
#             self.TruckCumulativeDistance[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.CONTINUOUS,lb=0,ub=np.sum(list(self.TruckDistance.values())),\
#                                                         name='g'+str(i+1))
        
#             self.DecisionIfThen[i+1] = self.model.addVars(np.concatenate((self.TruckCusIndexDict[i+1],self.DockingIndex)), vtype=GRB.BINARY,name='gamma'+str(i+1))
        
#         self.NodesBeta = self.model.addVars(self.NodesIndex, lb=0, ub=self.UbAmount,\
#                                                         vtype=GRB.INTEGER, name="beta")
        
#     def generateArcsIndexFromNode(self,vech_type,node_i_idx=None,node_j_idx=None):
#         list_of_arcs_index = []
#         if (node_i_idx is not None): node_i_name = [self.PathConstructSeries[node_i_idx]] 
#         else: node_i_name = self.Nodes
#         if node_j_idx is not None: node_j_name = [self.PathConstructSeries[node_j_idx]] 
#         else: node_j_name = self.Nodes
 
#         if 'T' in vech_type:
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         '''If vech_type=T: find all arcs from all truck'''
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.PathConstructSeries[self.PathConstructSeries.str.contains(arcs_name)].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)>1: list_of_arcs_index+=list(np.reshape(arcs_index,(len(arcs_index),-1)))
#                         elif len(arcs_index)!=0: list_of_arcs_index.append(arcs_index)
        
#         elif vech_type =='D':
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.PathConstructSeries[self.PathConstructSeries==arcs_name].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)!=0: list_of_arcs_index.append(arcs_index) 
#         return list_of_arcs_index
        
#     def convertArcsIndex2Node(self,arcs_idx):
#         [node_i_name,node_j_name]=self.PathConstructSeries[arcs_idx[0]].split(',')[:2]
#         node_i_idx = self.PathConstructSeries[self.PathConstructSeries==node_i_name].index.values[0]
#         node_j_idx = self.PathConstructSeries[self.PathConstructSeries==node_j_name].index.values[0]
#         return [node_i_idx,node_j_idx]
    
#     def generateConstrains(self):
#         # TRUCK CONDITIONS
#         ## TRUCK FLOW'S CONSERVATION (ONLY CUS&DOCK)
#         for p in range(self.NoTruckPerPath):
#             constr1 = ( quicksum(self.TruckArcs[self.generateArcsIndexFromNode('T'+str(p+1),i,h)[0][0]]\
#                         for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotSIndex)) if i!=h)\
#                    -quicksum(self.TruckArcs[self.generateArcsIndexFromNode('T'+str(p+1),h,j)[0][0]]\
#                         for j in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)) if j!=h)==0\
#                             for h in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex)) )
#             self.model.addConstrs(constr1, name = 'conservation_truck'+str(p+1))

#         ## TRUCK Cumulative weight & Sub-Tour Elimination
#             for a_idx in self.TruckArcsIndexDict[p+1]:
#                 [node_i_idx,node_j_idx] = self.convertArcsIndex2Node([a_idx])
#                 constr2_1 = ( self.TruckCumulativeDemand[p+1][node_j_idx] >= \
#                             self.TruckCumulativeDemand[p+1][node_i_idx]+self.CustomerDemand[node_j_idx]*self.TruckArcs[a_idx]\
#                            -self.M*(1-self.TruckArcs[a_idx]))
#                 self.model.addConstr(constr2_1,name='cumulative_demand_truck'+str(p+1))

#                 arc_ij = self.mergeDepotCusArcsVar([','.join([self.PathConstructSeries[node_i_idx],\
#                                                               self.PathConstructSeries[node_j_idx]]+['T'+str(p+1)])])
#                 constr2_2 = ( self.TruckCumulativeDistance[p+1][node_j_idx] >= \
#                             self.TruckCumulativeDistance[p+1][node_i_idx]+\
#                             self.TruckDistance[arc_ij[0]]*self.TruckArcs[a_idx]\
#                            -self.M*(1-self.TruckArcs[a_idx]))
#                 self.model.addConstr(constr2_2,name='cumulative_distance_truck'+str(p+1))  
# #                 if node_j_idx not in np.concatenate((self.DepotTIndex,self.DepotSIndex)):
# #                 if node_j_idx in self.DockingIndex:
#                 #To force g=>0
#                 extra_const_2 = ( self.TruckCumulativeDistance[p+1][node_j_idx] <= \
#                             self.TruckCumulativeDistance[p+1][node_i_idx]+\
#                             self.TruckDistance[arc_ij[0]]*self.TruckArcs[a_idx])
                                 
# #                 extra_const = ( self.TruckCumulativeDistance[p+1][node_j_idx] <= \
# #                            self.M*(self.TruckArcs[a_idx]))
# #                 self.model.addConstr(extra_const_2,name='force_not_visit_to_zero'+str(p+1))
                
                
# #             ## TRUCK distance limit
# #             constr2_3 = (self.TruckCumulativeDistance[p+1][i]<=self.Dt for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)))
# #             self.model.addConstrs(constr2_3,name='truck_distance_limit'+str(p+1))
#             ## TRUCK capacity limit
#             constr2_3 = (self.TruckCumulativeDemand[p+1][i]<=self.Lt for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)))
#             self.model.addConstrs(constr2_3,name='truck_capacity_limit'+str(p+1))
            
#             ## Truck passed count
#             for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotSIndex)):
#                 all_poss_p_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type ='T'+str(p+1))
#                 if len(all_poss_p_truck_arc_idx_start)!=0:
#                     constr4 = (self.TruckPassedNode[p+1][i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                                                for a_idx in all_poss_p_truck_arc_idx_start))
#                     self.model.addConstr(constr4,name='alpha'+str(p+1)+'_'+str(i))
        
#         ## ARC SPLIT COUNT
#         for i in np.concatenate((self.DepotSIndex,self.DepotTIndex,self.TruckCusIndex,self.DroneCusIndex,self.DockingIndex)):
#             if i!=self.DepotTIndex[0]:   
#                 all_poss_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'T')
# #                 print(i,all_poss_truck_arc_idx_start)
#                 all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                 constr3 = self.NodesDelta[i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                                     for a_idx in all_poss_truck_arc_idx_start)\
#                                                 +quicksum(self.DroneArcs[a_idx[0]]\
#                                                     for a_idx in all_poss_drone_arc_idx_start)
#                 self.model.addConstr(constr3,name='arc_split_count'+str(i))
#             if i!=self.DepotSIndex[0]:
#                 all_poss_truck_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=i,vech_type = 'T')
#                 all_poss_drone_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=i,vech_type = 'D')
#                 constr3 = self.NodesBeta[i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                     for a_idx in all_poss_truck_arc_idx_end)\
#                                 +quicksum(self.DroneArcs[a_idx[0]]\
#                                     for a_idx in all_poss_drone_arc_idx_end)
#                 self.model.addConstr(constr3,name='arc_incoming_count'+str(i))

#         ## TRUCK RELEASE
#         all_poss_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=self.DepotSIndex[0],vech_type = 'T')
#         constr5 = quicksum(self.TruckArcs[a_idx[0]] for a_idx in all_poss_truck_arc_idx_start)>=1
#         self.model.addConstr(constr5,name='truck_release')
        
#         #####################################################################
#         #####################################################################
#         # DRONE CONDITION 
#         ## DRONE FLOW'S CONSERVATION (ONLY CUS&DOCK)   
#         constr6 = ( quicksum(self.DroneArcs[self.generateArcsIndexFromNode('D',i,h)[0][0]]\
#                             for i in np.concatenate((self.TruckCusIndex,self.DroneCusIndex,self.DockingIndex,self.DepotSIndex))\
#                                 if (i!=h and len(self.generateArcsIndexFromNode('D',i,h))!=0))\
#                    -quicksum(self.DroneArcs[self.generateArcsIndexFromNode('D',h,j)[0][0]] \
#                             for j in np.concatenate((self.DroneCusIndex,self.DockingIndex,self.DepotTIndex))\
#                                 if (j!=h and len(self.generateArcsIndexFromNode('D',h,j))!=0)) == 0\
#                                     for h in self.DroneCusIndex)
#         self.model.addConstrs(constr6, name = 'conservation_drone')
        
#         ## DRONE Cumulation weight & Sub-Tour Elimination
#         for a_idx in self.DroneArcsIndex:
#             [node_i_idx,node_j_idx] = self.convertArcsIndex2Node([a_idx])
#             constr7_1 = ( self.DroneCumulativeDemand[node_j_idx] >= \
#                         self.DroneCumulativeDemand[node_i_idx]+\
#                         self.CustomerDemand[node_j_idx]*self.DroneArcs[a_idx]\
#                        -self.M*(1-self.DroneArcs[a_idx]))
#             self.model.addConstr(constr7_1,name='cumulative_demand_drone')
            
#             arc_ij = self.mergeDepotCusArcsVar([','.join([self.PathConstructSeries[node_i_idx],\
#                                                           self.PathConstructSeries[node_j_idx]]+['D'])])
#             constr7_2 = ( self.DroneCumulativeDistance[node_j_idx] >= \
#                         self.DroneCumulativeDistance[node_i_idx]+\
#                         self.DroneDistance[arc_ij[0]]*self.DroneArcs[a_idx]\
#                        -self.M*(1-self.DroneArcs[a_idx]))
#             self.model.addConstr(constr7_2,name='cumulative_distance_drone')
            
#         ## DRONE capacity limit
#         constr7_3 = (self.DroneCumulativeDemand[i]<=self.Ld for i in self.DroneCusIndex)
#         self.model.addConstrs(constr7_3,name='drone_capacity')
        
#         ## DRONE flying range limit
#         constr7_4 = (self.DroneCumulativeDistance[i]<=self.Dd for i in np.concatenate((self.DroneCusIndex,self.DockingIndex,self.DepotTIndex)))
#         self.model.addConstrs(constr7_4,name='drone_flying_range')
        
#         ## Prevent DRONE from spliting new drone path at drone node
#         constr8 = (self.NodesDelta[i]<=1 for i in self.DroneCusIndex)
#         self.model.addConstrs(constr8,name='limit_no_drone_landing')
        
#         for i in self.DockingIndex:
#             all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#             constr10_2 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                     (self.M*quicksum(self.TruckPassedNode[p+1][i] for p in range(self.NoTruckPerPath) )) )
#             self.model.addConstr(constr10_2,name='drone_takeoff_nodes_docking'+'_'+str(i))

#         ## Limit number of DRONE taking-off from truck
#         for p in range(self.NoTruckPerPath):
#             constr9_1 = quicksum(self.NodesDelta[i]-self.TruckPassedNode[p+1][i] for i in self.TruckCusIndexDict[p+1])<=self.Lr
#             self.model.addConstr(constr9_1,name='limit_no_drone_takeoff_from_truck'+str(p+1))
            
#             constr9_2 = quicksum(self.NodesBeta[i]-self.TruckPassedNode[p+1][i] for i in self.DockingIndex) <=self.Lr
#             self.model.addConstr(constr9_2,name='limit_no_drone_incoming2dock_on_each_truck'+str(p+1))
        
#             ## DRONE can take-off from customer nodes or docking that truck visited
#             for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex)):
#                 if i not in self.DockingIndex: 
#                     all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                     constr10_1 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                             (self.M*self.TruckPassedNode[p+1][i]))
#                     self.model.addConstr(constr10_1,name='drone_takeoff_nodes'+str(p+1)+'_'+str(i))
                
#                 ## EXTRA Eliminate drone reverse path
#                 all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                 for dock_idx in self.DockingIndex:
#                     if i!= dock_idx:
#                         constr_ext1 = (self.TruckCumulativeDistance[p+1][i]-self.TruckCumulativeDistance[p+1][dock_idx])<=\
#                                         self.M*self.DecisionIfThen[p+1][i]
#                         self.model.addConstr(constr_ext1,name='if_then_1'+str(p+1)+str(i))

#                         constr_ext2 = quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                                         self.M*(1-self.DecisionIfThen[p+1][i])
#                         self.model.addConstr(constr_ext2,name='if_then_2'+str(p+1)+str(i))


#         ## Prevent DRONE from visiting customer nodes that truck already visited (treat i as destination node)
#         for i in self.TruckCusIndexDict[1]:
#             drone_cus_node = self.getBothTypeCustomerIndex(i)[1]
#             all_truck_cus_node = self.getBothTypeCustomerIndex(i)[0]
#             all_poss_drone_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=drone_cus_node,vech_type = 'D')
#             constr11 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_end)<=\
#                         self.M*(1-quicksum(self.TruckPassedNode[p+1][all_truck_cus_node[p]] for p in range(self.NoTruckPerPath) )))
#             self.model.addConstr(constr11,name='drone_landing_nodes'+str(i))
        
#         ## DRONE can land on docking only if truck visited it
#         for d_idx in self.DockingIndex:
#             all_poss_drone_arc_dock_end = self.generateArcsIndexFromNode(node_j_idx=d_idx,vech_type = 'D')
#             constr12 = quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_dock_end) <=\
#                     self.M*quicksum(self.TruckPassedNode[p+1][d_idx] for p in range(self.NoTruckPerPath))
#             self.model.addConstr(constr12,name='drone_landing_docking'+str(d_idx))

#     def convertDroneArc2TruckArcIndex(self,d_a_idx):
#         arc_name_split = self.PathConstructSeries[d_a_idx].split(',')
#         t_arc_name = ','.join(arc_name_split[:2]+['T'])
#         t_a_idx = self.PathConstructSeries[self.PathConstructSeries==t_arc_name].index.values
#         return t_a_idx[0]

#     def mergeDepotCusArcsVar(self,a_var):
#         '''INPUT: ['depot_s,customer_1_T1,T1','customer_2_T1,depot_t,T1']
#         OUTPUT: ['depot,customer_1,T1','customer_2,depot,T1']'''
#         new_a_var =[]
#         for a in a_var:
#             v = a.split(',')
#             if len(v)>1:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0]=self.depot[0]
#                 elif v[0] in self.TruckCusNodes+self.DroneCusNodes: v[0]='_'.join(v[0].split('_')[:-1])
#                 if v[1]==self.depot_s[0] or v[1]==self.depot_t[0]:v[1]=self.depot[0]
#                 elif (v[1] in self.TruckCusNodes+self.DroneCusNodes): v[1]='_'.join(v[1].split('_')[:-1])   
#             else:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0] = self.depot[0]
#             new_a_var.append(','.join(v))
#         return new_a_var   
    
#     def generateCost(self,a_idx):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if 'T' in arcs_name.split(',')[2]:
#             # Convert all depot to depot, calculating distance
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.TruckDistance[a_name]*self.transportation_truck_cost*60/self.truck_speed 
# #             print(arcs_name,cost)
#         elif arcs_name.split(',')[2]=='D' :
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = (self.DroneDistance[a_name]*self.transportation_drone_cost*60/self.drone_speed) 
# #         print(arcs_name)
#         return cost
    
#     def getBothTypeCustomerIndex(self, index):
#         node_name = self.PathConstructSeries[index]
#         customer_name = '_'.join(node_name.split('_')[:-1])+'_'
# #         print(node_name,customer_name)
#         truck_node_index = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckCusNodes)&self.PathConstructSeries.str.contains(customer_name)].index
# #         print(truck_node_index)
#         drone_node_index = self.PathConstructSeries[self.PathConstructSeries.isin(self.DroneCusNodes)&self.PathConstructSeries.str.contains(node_name[:-2])].index
#         return [list(truck_node_index),drone_node_index[0]]
    
#     def generateObjective(self,_duals):
#         OBJ = quicksum(self.TruckArcs[idx]*self.generateCost(idx) for idx in self.TruckArcsIndex)\
#             + quicksum(self.DroneArcs[idx]*self.generateCost(idx) for idx in self.DroneArcsIndex)\
#             - quicksum( ( self.NodesDelta[self.getBothTypeCustomerIndex(i)[1]] +\
#                         quicksum(self.TruckPassedNode[p+1][self.getBothTypeCustomerIndex(i)[0][p]] for p in range(self.NoTruckPerPath)) )\
#                         *_duals[i]\
#                       for i in self.TruckCusIndexDict[1]) - _duals[-1]*quicksum(self.TruckPassedNode[p+1][self.DepotSIndex[0]] for p in range(self.NoTruckPerPath))
        
#         self.model.setObjective(OBJ,sense = GRB.MINIMIZE)
#         return OBJ
    
#     #In case found the new sol that give better reduce cost, 
#     #we will output it and add to new col in master problem
#     def getNewPath(self):
#         return self.model.getAttr('X',self.model.getVars())
    
#     def generateCostOfVars_implicit(self,a_idx,new_vars):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if 'T' in arcs_name.split(',')[2]:
#             # Convert all depot to depot, calculating distance
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.TruckDistance[a_name]*self.transportation_truck_cost*60/self.truck_speed 
#         elif arcs_name.split(',')[2]=='D' :
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.DroneDistance[a_name]*self.transportation_drone_cost*60/self.drone_speed
# #         print(arcs_name,cost,cost*new_vars[a_idx])
#         return cost
    
#     def getNewPathCost(self,new_vars):
#         cost = quicksum(new_vars[idx]*self.generateCostOfVars_implicit(idx,new_vars) for idx in self.ArcsIndex)
#         return cost
    
#     def solveModel(self, timeLimit = None,GAP=None):
#         self.model.setParam('PoolGap',10)
#         self.model.setParam(GRB.Param.PoolSearchMode, 2)
#         self.model.setParam('TImeLimit', timeLimit)
#         self.model.setParam('MIPGap',GAP)
#         self.model.optimize()
# # FOR EXPLICIT COLUMN GENERATION
#     def generateCostOfVars(self,var,a_idx):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if arcs_name.split(',')[2]=='T':
#             cost = self.TruckDistance[arcs_name]*self.transportation_truck_cost*60/self.truck_speed 
#         elif arcs_name.split(',')[2]=='D' :
#             truck_arc_index = self.convertDroneArc2TruckArcIndex(a_idx)
#             cost = (1-var[truck_arc_index])*(self.DroneDistance[arcs_name]*self.transportation_drone_cost*60/self.drone_speed) 
            
#         return cost

#     def calculateReduceCost(self,var_idx):
#         var = x[x.columns[var_idx]]
#         Cost = quicksum(var[idx]*self.generateCostOfVars(var,idx) for idx in self.ArcsIndex)+truck_fix_cost
#         Pricing = quicksum(var[i]*self.Duals[i] for i in self.CustomerIndex)+self.Duals[-1]
#         return Cost,Pricing

#     #In case found the new sol that give better reduce cost, we will output it and add to new col in master problem
#     def getNewVar(self):
#         return self.added_indices
        
#     def update_processed_indices(self):
#         return self.processed_indices

#     def generatePathWithNegReduceCost(self,):
#         for var in self.unexplored_indices:
#             self.processed_indices.append(var)
#             Cost,Pricing = self.calculateReduceCost(var)
#             reduce_cost = Cost-Pricing
#             print(reduce_cost)
#             if reduce_cost.getValue()<0:
#                 print('REDCUCE COST',reduce_cost,var)
#                 self.added_indices.append([var,Cost.getValue()])
                
# def sort_index_sol_for_plot(sol,model):
#     sol_index = model.NodesDelta.keys()+model.TruckArcs.keys()+model.DroneArcs.keys()
#     return pd.Series(sol,index = sol_index).sort_index().values


# ############################################################################################           
# ############################################################################################ 

# class VRPdMasterProblemDebugged:
#     def __init__(self, path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_drone,no_truck,_truck_distance,_drone_distance,constant_dict,extra_constr=None):
# #                  ,truck_count,drone_carried_by_truck,transportation_truck_cost,transportation_drone_cost):
                
#         self.NoTruckPerPath = no_truck
#         self.PathCoeff = path['PathCoeff'].values
#         self.Path = path.copy()
#         self.TruckCusNodes = truck_cus_nodes
#         self.TruckCusNodesDict = truck_cus_nodes_dict
#         self.DroneCusNodes = drone_cus_nodes
#         self.Nodes = all_depot+truck_cus_nodes+drone_cus_nodes+docking
#         self.Docking = docking
#         self.AllDepot = all_depot
#         self.depot_s = [all_depot[0]]
#         self.depot_t = [all_depot[1]]
#         self.depot = ['depot']
        
#         self.TruckArcs = arcs_truck
#         self.DroneArcs = arcs_drone
#         self.Arcs = arcs_truck+arcs_drone
        
#         self.CoeffSeries = pd.Series(all_depot+truck_cus_nodes+drone_cus_nodes+docking+self.Arcs)
#         self.DepotIndex = self.CoeffSeries[self.CoeffSeries.isin(all_depot)].index.values
#         self.DepotSIndex = self.CoeffSeries[self.CoeffSeries.isin([all_depot[0]])].index.values
#         self.DepotTIndex = self.CoeffSeries[self.CoeffSeries.isin([all_depot[1]])].index.values
#         self.TruckCusIndex = self.CoeffSeries[self.CoeffSeries.isin(truck_cus_nodes)].index.values
#         self.TruckCusIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCusIndexDict[i+1] = self.CoeffSeries[self.CoeffSeries.isin(self.TruckCusNodesDict[i+1])].index.values
        
#         self.DroneCusIndex = self.CoeffSeries[self.CoeffSeries.isin(drone_cus_nodes)].index.values
        
#         self.AllCustomerIndex = self.CoeffSeries[self.CoeffSeries.isin(truck_cus_nodes+drone_cus_nodes)].index.values
#         self.DockingIndex = self.CoeffSeries[self.CoeffSeries.isin(docking)].index.values
#         self.NodesIndex = self.CoeffSeries[self.CoeffSeries.isin(self.Nodes)].index.values
        
#         self.ArcsIndex = self.CoeffSeries[self.CoeffSeries.str.contains(',')].index.values
    
#         self.TruckArcsIndex = self.CoeffSeries[self.CoeffSeries.isin(self.TruckArcs)].index.values
#         self.DroneArcsIndex = self.CoeffSeries[self.CoeffSeries.isin(self.DroneArcs)].index.values
        
        
#         self.K = constant_dict['truck_count']
#         self.Lr = constant_dict['drone_carried_by_truck']
#         self.transportation_truck_cost=constant_dict['transportation_truck_cost']
#         self.transportation_drone_cost=constant_dict['transportation_drone_cost']
#         self.truck_speed = constant_dict['truck_speed']
#         self.drone_speed = constant_dict['drone_speed']
#         self.truck_fix_cost = constant_dict['truck_fix_cost']
#         self.constant_dict = constant_dict
        
#         self.TruckDistance = _truck_distance
#         self.DroneDistance = _drone_distance

        
#         self.model = Model("MasterProblem")
#         self.PathIndex = pd.Series(path.index).index.values
#         if extra_constr is not None: self.K = extra_constr 

        
#     def buildModel(self):
#         self.generateVariables()
#         self.generateConstraints()
#         self.generateObjective()
#         self.model.update()
        
#     def convertDroneArc2TruckArcIndex(self,d_a_idx):
#         arc_name_split = self.CoeffSeries[d_a_idx].split(',')
#         t_arc_name = ','.join(arc_name_split[:2]+['T'])
#         t_a_idx = self.CoeffSeries[self.CoeffSeries==t_arc_name].index.values
#         return t_a_idx[0]
    
#     def mergeDepotCusArcsVar(self,a_var):
#         '''INPUT: ['depot_s,customer_1_T1,T1','customer_2_T1,depot_t,T1']
#         OUTPUT: ['depot,customer_1,T1','customer_2,depot,T1']'''
#         new_a_var =[]
#         for a in a_var:
#             v = a.split(',')
#             if len(v)>1:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0]=self.depot[0]
#                 elif v[0] in self.TruckCusNodes+self.DroneCusNodes: v[0]='_'.join(v[0].split('_')[:-1])
#                 if v[1]==self.depot_s[0] or v[1]==self.depot_t[0]:v[1]=self.depot[0]
#                 elif (v[1] in self.TruckCusNodes+self.DroneCusNodes): v[1]='_'.join(v[1].split('_')[:-1])   
#             else:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0] = self.depot[0]
#             new_a_var.append(','.join(v))
#         return new_a_var
    
#     def generateRouteCost(self,index):
#         def cost_truck_arc(a_idx):
#             a_name = self.mergeDepotCusArcsVar([self.CoeffSeries[a_idx]])[0]
#             return self.TruckDistance[a_name]
#         def cost_drone_arc(a_idx):
#             a_name = self.mergeDepotCusArcsVar([self.CoeffSeries[a_idx]])[0]
#             return self.DroneDistance[a_name]
# #         print(cost_truck_arc)
#         path_cost = quicksum(self.Path.iloc[index][0][a]*cost_truck_arc(a)*self.transportation_truck_cost*60/self.truck_speed for a in self.TruckArcsIndex)\
#                     +quicksum(self.Path.iloc[index][0][a]*cost_drone_arc(a)*self.transportation_drone_cost*60/self.drone_speed for a in self.DroneArcsIndex)
# #         print(path_cost)
#         return path_cost
    
#     def getBothTypeCustomerIndex(self, index):
#         node_name = self.CoeffSeries[index]
#         customer_name = '_'.join(node_name.split('_')[:-1])+'_'
# #         print(node_name,customer_name)
#         truck_node_index = self.CoeffSeries[self.CoeffSeries.isin(self.TruckCusNodes)&self.CoeffSeries.str.contains(customer_name)].index
# #         print(truck_node_index)
#         drone_node_index = self.CoeffSeries[self.CoeffSeries.isin(self.DroneCusNodes)&self.CoeffSeries.str.contains(node_name[:-2])].index
#         return [list(truck_node_index),drone_node_index[0]]
    
#     def generateArcsIndexFromNode(self,vech_type,node_i_idx=None,node_j_idx=None):
#         list_of_arcs_index = []
#         if (node_i_idx is not None): node_i_name = [self.CoeffSeries[node_i_idx]] 
#         else: node_i_name = self.Nodes
#         if node_j_idx is not None: node_j_name = [self.CoeffSeries[node_j_idx]] 
#         else: node_j_name = self.Nodes
#         if 'T' in vech_type:
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         '''If vech_type=T: find all arcs from all truck'''
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.CoeffSeries[self.CoeffSeries.str.contains(arcs_name)].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)>1: list_of_arcs_index+=list(np.reshape(arcs_index,(len(arcs_index),-1)))
#                         elif len(arcs_index)!=0: list_of_arcs_index.append(arcs_index)
        
#         elif vech_type =='D':
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.CoeffSeries[self.CoeffSeries==arcs_name].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)!=0: list_of_arcs_index.append(arcs_index) 
#         return list_of_arcs_index
    
#     def generateVariables(self):
#         self.path = self.model.addVars(self.PathIndex, lb=0,
#                                        vtype=GRB.BINARY, name='path')
#         print('Finish generating variables!')
        
#     def generateConstraints(self):
#         # Fix number of vehicles used (truck+total_drone)
#         const0 = ( quicksum(self.PathCoeff[rt][i]*self.path[rt] for rt in self.PathIndex)<=(self.K+self.K*self.Lr) \
#          for i in self.DepotIndex)
#         self.model.addConstrs(const0,name='Depot')
                  
#         const1 = ( quicksum(( quicksum(self.PathCoeff[rt][self.getBothTypeCustomerIndex(i)[0][j]] for j in range(self.NoTruckPerPath))  \
#                              +self.PathCoeff[rt][self.getBothTypeCustomerIndex(i)[1]])*self.path[rt] for rt in self.PathIndex)==1 \
#                              for i in self.TruckCusIndexDict[1] )
#         self.model.addConstrs( const1,name='customer_coverage' )

# #         const2 = ( quicksum(self.path[rt] for rt in self.PathIndex)<= self.K )
        
#         const2 = ( quicksum(self.path[rt]*quicksum(self.PathCoeff[rt][a_idx[0]] for a_idx in self.generateArcsIndexFromNode('T',node_i_idx=self.DepotSIndex[0]) ) \
#                             for rt in self.PathIndex)<= self.K )
#         self.model.addConstr( const2,name='truck_number' )
#         print('Finish generating constrains!')
    
#     def getTruckNumberFromRoute(self,path_name,show_value=False):
#         '''path_name: name of the var'''
#         truck_no = quicksum(self.Path.loc[path_name][0][a_idx[0]] for a_idx in self.generateArcsIndexFromNode('T',node_i_idx=self.DepotSIndex[0]) )
#         if show_value: print('Truck no. of path',path_name,':', np.round(truck_no.getValue()))
#         return  np.round(truck_no.getValue())
#     def getDroneNumberFromRoute(self,path_name):
#         '''path_name: name of the var'''
#         drone_arcs_name = pd.Series(self.DroneArcs)
#         drone_departing_arc = drone_arcs_name[drone_arcs_name.str.contains('_T')|drone_arcs_name.str.contains('depot_s')|drone_arcs_name.str.contains(r'dock_.*?,cus')]
#         drone_departing_arc_index = self.CoeffSeries[self.CoeffSeries.isin(drone_departing_arc.values)].index
#         drone_no = quicksum(self.Path.loc[path_name][0][a_idx] for a_idx in drone_departing_arc_index)
#         print('Drone no. of path',path_name,':', np.round(drone_no.getValue()))
#         return  np.round(drone_no.getValue())
#     def getNumberCustomerServedByTruck(self,path_name):
#         path_sol = pd.Series(self.Path.loc[path_name][0],index=self.CoeffSeries)
#         served_truck = path_sol[(path_sol.index.isin(self.TruckCusNodes))&(path_sol>0)]
#         return served_truck
    
#     def getNumberCustomerServedByDrone(self,path_name):
#         path_sol = pd.Series(self.Path.loc[path_name][0],index=self.CoeffSeries)
#         served_drone = path_sol[(path_sol.index.isin(self.DroneCusNodes))&(path_sol>0)]
#         return served_drone
        
#     def generateObjective(self):
#         # Minimize the total cost of the used rolls
# #         self.model.setObjective( quicksum(self.path[rt]*(self.generateRouteCost(rt)) for rt in self.PathIndex) ,
# #                                 sense=GRB.MINIMIZE)
#         self.model.setObjective( quicksum( self.path[rt]*(self.generateRouteCost(rt)+self.getTruckNumberFromRoute(self.Path.iloc[rt].name)*self.truck_fix_cost) for rt in self.PathIndex), sense=GRB.MINIMIZE) 
#         print('Finish generating objective!')
#     #################################
#     def solveRelaxedModel(self):
#         #Relax integer variables to continous variables
#         self.relaxedModel = self.model.relax()
#         self.relaxedModel.optimize()
        
#     def getRelaxSolution(self):
#         a = pd.Series(self.relaxedModel.getAttr('X'))
#         return a[a>0]

#     def getDuals(self):
#         return self.relaxedModel.getAttr('Pi',self.model.getConstrs())
#     #################################
#     def addColumn(self, objective,newPath, get_var,_pathname):
# #         ctName = ('path[%s]'%len(self.model.getVars()))
#         ctName = _pathname
#         newPathSeries = pd.Series(newPath,index = self.CoeffSeries)
#         LHS_Coeff =list()
#         for i in self.TruckCusIndexDict[1]:
#             d_idx = self.getBothTypeCustomerIndex(i)[1]
#             theta_d_var_name = 'theta['+str(d_idx)+']'
# #             print(theta_d_var_name)
# #             print(i)
#             t_idx = self.getBothTypeCustomerIndex(i)[0]
#             truck_passed_var_name = ['alpha'+str(j+1)+'['+str(t_idx[j])+']' for j in range(len(t_idx))]
#             truck_released_var_name = ['alpha'+str(j+1)+'[0]' for j in range(len(t_idx))]
# #             print(truck_passed_var_name)
#             LHS_Coeff.append(get_var(theta_d_var_name).X+quicksum(get_var(t_var_name).X for t_var_name in truck_passed_var_name).getValue())
# #         new_col = [1]*len(self.DepotIndex)+LHS_Coeff+[1]
#         new_col = [1]*len(self.DepotIndex)+LHS_Coeff+[quicksum(get_var(t_var_name).X for t_var_name in truck_released_var_name).getValue()]
#         newColumn = Column(new_col, self.model.getConstrs())
#         print(new_col)
#         print('Cols',newColumn)
# #         self.Path = self.Path.append({'PathCoeff':newPathSeries.values},ignore_index=True)
# #         print(newPathSeries.values)
#         self.Path.loc[ctName] = [newPathSeries.values]
#         self.model.addVar(vtype = GRB.BINARY, lb=0, obj=objective, column=newColumn,name= ctName)
#         self.model.update()
#     #################################    
#     def solveModel(self, timeLimit = None,GAP=None):
#         self.model.setParam('SolutionNumber',2)
#         self.model.setParam(GRB.Param.PoolSearchMode, 2)
#         self.model.setParam('TImeLimit', timeLimit)
#         self.model.setParam('MIPGap',GAP)
#         self.model.optimize()
#     #########EXPLICIT COL_GEN###############
#     def addColumnExplicitly(self, newCol):
#         for ncol,cost in newCol:
#             ctName = ('path[%s]'%ncol)
#             newPathSeries = pd.Series(x[x.columns[ncol]],index = self.CoeffSeries)
#             new_col = list(newPathSeries[newPathSeries.index.isin(self.depot+customers)].values)+[1]
#             print(new_col)
#             newColumn = Column(new_col, self.model.getConstrs())
#             self.model.addVar(vtype = GRB.BINARY, lb=0, obj=cost, column=newColumn,name= ctName)
#             self.model.update()



# class VRPdPricingProblemDebugged:
#     def __init__(self, all_depot, truck_cus_nodes, truck_cus_nodes_dict, drone_cus_nodes,docking, arcs_truck,arcs_truck_dict, arcs_drone, duals, customer_demand, no_truck, _truck_distance, _drone_distance, constant_dict, all_indices=None, processed_indices=None):
# #truck_count,drone_carried_by_truck,max_capacity_truck,max_distance_drone, max_weight_drone,
#         self.NoTruckPerPath = no_truck
#         # SET OF NODES AND ARCS
#         self.AllDepot = all_depot
#         self.Nodes = all_depot+truck_cus_nodes+drone_cus_nodes+docking
#         self.TruckCusNodes = truck_cus_nodes
#         self.TruckCusNodesDict = truck_cus_nodes_dict
#         self.AllDepot = all_depot
#         self.depot_s = [all_depot[0]]
#         self.depot_t = [all_depot[1]]
#         self.depot = ['depot']
        
#         self.DroneCusNodes = drone_cus_nodes
#         self.Docking = docking
#         self.TruckArcs = arcs_truck
#         self.TruckArcsDict = arcs_truck_dict
        
#         self.DroneArcs = arcs_drone
#         self.Arcs = arcs_truck+arcs_drone
#         self.PathConstructSeries = pd.Series(all_depot+truck_cus_nodes+drone_cus_nodes+docking+self.Arcs)
        
#         # INDEXING OF NODES
#         self.NodesIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.Nodes)].index.values
#         self.TruckCusIndex = self.PathConstructSeries[self.PathConstructSeries.isin(truck_cus_nodes)].index.values
#         self.TruckCusIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCusIndexDict[i+1] = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckCusNodesDict[i+1])].index.values
        
#         self.DroneCusIndex = self.PathConstructSeries[self.PathConstructSeries.isin(drone_cus_nodes)].index.values
#         self.AllCustomerIndex = self.PathConstructSeries[self.PathConstructSeries.isin(truck_cus_nodes+drone_cus_nodes)].index.values
#         self.DepotAllIndex = self.PathConstructSeries[self.PathConstructSeries.isin(all_depot)].index.values
#         self.DepotSIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.depot_s)].index.values
#         self.DepotTIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.depot_t)].index.values
#         self.DockingIndex = self.PathConstructSeries[self.PathConstructSeries.isin(docking)].index.values
#         # INDEXING OF ARCS
#         self.ArcsIndex = self.PathConstructSeries[self.PathConstructSeries.str.contains(',')].index.values
#         self.TruckArcsIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckArcs)].index.values
#         self.TruckArcsIndexDict = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckArcsIndexDict[i+1] = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckArcsDict[i+1])].index.values
        
#         self.DroneArcsIndex = self.PathConstructSeries[self.PathConstructSeries.isin(self.DroneArcs)].index.values
#         # OTHER PARAMETERS
#         self.CustomerDemand = pd.Series(customer_demand,index=self.NodesIndex)
#         self.TruckDistance = _truck_distance
#         self.DroneDistance = _drone_distance
#         self.transportation_truck_cost=constant_dict['transportation_truck_cost']
#         self.transportation_drone_cost=constant_dict['transportation_drone_cost']
#         self.truck_speed = constant_dict['truck_speed']
#         self.drone_speed = constant_dict['drone_speed']
#         self.truck_fix_cost = constant_dict['truck_fix_cost']
#         self.K = constant_dict['truck_count']
#         self.Lr = constant_dict['drone_carried_by_truck']
#         self.Ld = constant_dict['max_weight_drone']
#         self.Lt = constant_dict['max_capacity_truck']
#         self.Dd = constant_dict['max_distance_drone']
#         self.UbAmount = 1+self.Lr
#         self.M = 1000000
#         # DUALS
#         self.Duals = duals
#         # DECLARE MODEL
#         self.model = Model('Subproblem')
#         self.PathConstructIndex = self.PathConstructSeries.index.values
    
#     def buildModel(self):
#         self.generateVariables()
#         self.generateConstrains()
#         self.generateObjective(self.Duals)
#         self.model.update()
        
#     def generateVariables(self):
#         self.NodesDelta = self.model.addVars(self.NodesIndex, lb=0, ub=self.UbAmount,\
#                                                         vtype=GRB.INTEGER, name="theta")
#         self.TruckArcs = self.model.addVars(self.TruckArcsIndex,\
#                                                         vtype=GRB.BINARY, name='truck_arcs')
        
#         self.DroneArcs = self.model.addVars(self.DroneArcsIndex,\
#                                                         vtype=GRB.BINARY, name='drone_arcs')
            
#         self.DroneCumulativeDemand = self.model.addVars(self.NodesIndex,vtype=GRB.CONTINUOUS,\
#                                                         lb=0,ub=self.CustomerDemand.sum(),name='z')
        
#         self.DroneCumulativeDistance = self.model.addVars(self.NodesIndex,vtype=GRB.CONTINUOUS,\
#                                                         lb=0,ub=np.sum(list(self.DroneDistance.values())),name='v')

#         self.TruckCumulativeDemand = dict()
#         self.TruckPassedNode = dict()
#         self.TruckCumulativeDistance = dict()
#         self.DecisionIfThen = dict()
#         for i in range(self.NoTruckPerPath):
#             self.TruckCumulativeDemand[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.CONTINUOUS,lb=0,ub=self.CustomerDemand.sum(),\
#                                                         name='y'+str(i+1))
#             self.TruckPassedNode[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.BINARY,name = 'alpha'+str(i+1))
        
#             self.TruckCumulativeDistance[i+1] = self.model.addVars(np.concatenate((self.DepotAllIndex,self.TruckCusIndexDict[i+1],self.DockingIndex)),\
#                                                         vtype=GRB.CONTINUOUS,lb=0,ub=np.sum(list(self.TruckDistance.values())),\
#                                                         name='g'+str(i+1))
#             self.DecisionIfThen[i+1] = dict()
#             for dock_idx in self.DockingIndex:

#                 self.DecisionIfThen[i+1][dock_idx] = self.model.addVars(np.concatenate((self.TruckCusIndexDict[i+1],self.DockingIndex)), vtype=GRB.BINARY,name='gamma'+str(i+1)+'dock_idx'+str(dock_idx))
        
#         self.NodesBeta = self.model.addVars(self.NodesIndex, lb=0, ub=self.UbAmount,\
#                                                         vtype=GRB.INTEGER, name="beta")
        
#     def generateArcsIndexFromNode(self,vech_type,node_i_idx=None,node_j_idx=None):
#         list_of_arcs_index = []
#         if (node_i_idx is not None): node_i_name = [self.PathConstructSeries[node_i_idx]] 
#         else: node_i_name = self.Nodes
#         if node_j_idx is not None: node_j_name = [self.PathConstructSeries[node_j_idx]] 
#         else: node_j_name = self.Nodes
 
#         if 'T' in vech_type:
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         '''If vech_type=T: find all arcs from all truck'''
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.PathConstructSeries[self.PathConstructSeries.str.contains(arcs_name)].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)>1: list_of_arcs_index+=list(np.reshape(arcs_index,(len(arcs_index),-1)))
#                         elif len(arcs_index)!=0: list_of_arcs_index.append(arcs_index)
        
#         elif vech_type =='D':
#             for i in node_i_name:
#                 for j in node_j_name:
#                     if j!=i:
#                         arcs_name = ','.join([i,j,vech_type])
#                         arcs_index = self.PathConstructSeries[self.PathConstructSeries==arcs_name].index.values
# #                         print(arcs_name,arcs_index)
#                         if len(arcs_index)!=0: list_of_arcs_index.append(arcs_index) 
#         return list_of_arcs_index
        
#     def convertArcsIndex2Node(self,arcs_idx):
#         [node_i_name,node_j_name]=self.PathConstructSeries[arcs_idx[0]].split(',')[:2]
#         node_i_idx = self.PathConstructSeries[self.PathConstructSeries==node_i_name].index.values[0]
#         node_j_idx = self.PathConstructSeries[self.PathConstructSeries==node_j_name].index.values[0]
#         return [node_i_idx,node_j_idx]
    
#     def generateConstrains(self):
#         # TRUCK CONDITIONS
#         ## TRUCK FLOW'S CONSERVATION (ONLY CUS&DOCK)
#         for p in range(self.NoTruckPerPath):
#             constr1 = ( quicksum(self.TruckArcs[self.generateArcsIndexFromNode('T'+str(p+1),i,h)[0][0]]\
#                         for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotSIndex)) if i!=h)\
#                    -quicksum(self.TruckArcs[self.generateArcsIndexFromNode('T'+str(p+1),h,j)[0][0]]\
#                         for j in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)) if j!=h)==0\
#                             for h in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex)) )
#             self.model.addConstrs(constr1, name = 'conservation_truck'+str(p+1))

#         ## TRUCK Cumulative weight & Sub-Tour Elimination
#             for a_idx in self.TruckArcsIndexDict[p+1]:
#                 [node_i_idx,node_j_idx] = self.convertArcsIndex2Node([a_idx])
#                 constr2_1 = ( self.TruckCumulativeDemand[p+1][node_j_idx] >= \
#                             self.TruckCumulativeDemand[p+1][node_i_idx]+self.CustomerDemand[node_j_idx]*self.TruckArcs[a_idx]\
#                            -self.M*(1-self.TruckArcs[a_idx]))
#                 self.model.addConstr(constr2_1,name='cumulative_demand_truck'+str(p+1))

#                 arc_ij = self.mergeDepotCusArcsVar([','.join([self.PathConstructSeries[node_i_idx],\
#                                                               self.PathConstructSeries[node_j_idx]]+['T'+str(p+1)])])
#                 constr2_2 = ( self.TruckCumulativeDistance[p+1][node_j_idx] >= \
#                             self.TruckCumulativeDistance[p+1][node_i_idx]+\
#                             self.TruckDistance[arc_ij[0]]*self.TruckArcs[a_idx]\
#                            -self.M*(1-self.TruckArcs[a_idx]))
#                 self.model.addConstr(constr2_2,name='cumulative_distance_truck'+str(p+1))  
# #                 if node_j_idx not in np.concatenate((self.DepotTIndex,self.DepotSIndex)):
# #                 if node_j_idx in self.DockingIndex:
#                 #To force g=>0
#                 extra_const_2 = ( self.TruckCumulativeDistance[p+1][node_j_idx] <= \
#                             self.TruckCumulativeDistance[p+1][node_i_idx]+\
#                             self.TruckDistance[arc_ij[0]]*self.TruckArcs[a_idx])
                                 
# #                 extra_const = ( self.TruckCumulativeDistance[p+1][node_j_idx] <= \
# #                            self.M*(self.TruckArcs[a_idx]))
# #                 self.model.addConstr(extra_const_2,name='force_not_visit_to_zero'+str(p+1))
                
                
# #             ## TRUCK distance limit
# #             constr2_3 = (self.TruckCumulativeDistance[p+1][i]<=self.Dt for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)))
# #             self.model.addConstrs(constr2_3,name='truck_distance_limit'+str(p+1))
#             ## TRUCK capacity limit
#             constr2_3 = (self.TruckCumulativeDemand[p+1][i]<=self.Lt for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotTIndex)))
#             self.model.addConstrs(constr2_3,name='truck_capacity_limit'+str(p+1))
            
#             ## Truck passed count
#             for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex,self.DepotSIndex)):
#                 all_poss_p_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type ='T'+str(p+1))
#                 if len(all_poss_p_truck_arc_idx_start)!=0:
#                     constr4 = (self.TruckPassedNode[p+1][i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                                                for a_idx in all_poss_p_truck_arc_idx_start))
#                     self.model.addConstr(constr4,name='alpha'+str(p+1)+'_'+str(i))
        
#         ## ARC SPLIT COUNT
#         for i in np.concatenate((self.DepotSIndex,self.DepotTIndex,self.TruckCusIndex,self.DroneCusIndex,self.DockingIndex)):
#             if i!=self.DepotTIndex[0]:   
#                 all_poss_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'T')
# #                 print(i,all_poss_truck_arc_idx_start)
#                 all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                 constr3 = self.NodesDelta[i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                                     for a_idx in all_poss_truck_arc_idx_start)\
#                                                 +quicksum(self.DroneArcs[a_idx[0]]\
#                                                     for a_idx in all_poss_drone_arc_idx_start)
#                 self.model.addConstr(constr3,name='arc_split_count'+str(i))
#             if i!=self.DepotSIndex[0]:
#                 all_poss_truck_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=i,vech_type = 'T')
#                 all_poss_drone_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=i,vech_type = 'D')
#                 constr3 = self.NodesBeta[i] == quicksum(self.TruckArcs[a_idx[0]]\
#                                     for a_idx in all_poss_truck_arc_idx_end)\
#                                 +quicksum(self.DroneArcs[a_idx[0]]\
#                                     for a_idx in all_poss_drone_arc_idx_end)
#                 self.model.addConstr(constr3,name='arc_incoming_count'+str(i))

#         ## TRUCK RELEASE
#         all_poss_truck_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=self.DepotSIndex[0],vech_type = 'T')
#         constr5 = quicksum(self.TruckArcs[a_idx[0]] for a_idx in all_poss_truck_arc_idx_start)>=1
#         self.model.addConstr(constr5,name='truck_release')
        
#         #####################################################################
#         #####################################################################
#         # DRONE CONDITION 
#         ## DRONE FLOW'S CONSERVATION (ONLY CUS&DOCK)   
#         constr6 = ( quicksum(self.DroneArcs[self.generateArcsIndexFromNode('D',i,h)[0][0]]\
#                             for i in np.concatenate((self.TruckCusIndex,self.DroneCusIndex,self.DockingIndex,self.DepotSIndex))\
#                                 if (i!=h and len(self.generateArcsIndexFromNode('D',i,h))!=0))\
#                    -quicksum(self.DroneArcs[self.generateArcsIndexFromNode('D',h,j)[0][0]] \
#                             for j in np.concatenate((self.DroneCusIndex,self.DockingIndex,self.DepotTIndex))\
#                                 if (j!=h and len(self.generateArcsIndexFromNode('D',h,j))!=0)) == 0\
#                                     for h in self.DroneCusIndex)
#         self.model.addConstrs(constr6, name = 'conservation_drone')
        
#         ## DRONE Cumulation weight & Sub-Tour Elimination
#         for a_idx in self.DroneArcsIndex:
#             [node_i_idx,node_j_idx] = self.convertArcsIndex2Node([a_idx])
#             constr7_1 = ( self.DroneCumulativeDemand[node_j_idx] >= \
#                         self.DroneCumulativeDemand[node_i_idx]+\
#                         self.CustomerDemand[node_j_idx]*self.DroneArcs[a_idx]\
#                        -self.M*(1-self.DroneArcs[a_idx]))
#             self.model.addConstr(constr7_1,name='cumulative_demand_drone')
            
#             arc_ij = self.mergeDepotCusArcsVar([','.join([self.PathConstructSeries[node_i_idx],\
#                                                           self.PathConstructSeries[node_j_idx]]+['D'])])
#             constr7_2 = ( self.DroneCumulativeDistance[node_j_idx] >= \
#                         self.DroneCumulativeDistance[node_i_idx]+\
#                         self.DroneDistance[arc_ij[0]]*self.DroneArcs[a_idx]\
#                        -self.M*(1-self.DroneArcs[a_idx]))
#             self.model.addConstr(constr7_2,name='cumulative_distance_drone')
            
#         ## DRONE capacity limit
#         constr7_3 = (self.DroneCumulativeDemand[i]<=self.Ld for i in self.DroneCusIndex)
#         self.model.addConstrs(constr7_3,name='drone_capacity')
        
#         ## DRONE flying range limit
#         constr7_4 = (self.DroneCumulativeDistance[i]<=self.Dd for i in np.concatenate((self.DroneCusIndex,self.DockingIndex,self.DepotTIndex)))
#         self.model.addConstrs(constr7_4,name='drone_flying_range')
        
#         ## Prevent DRONE from spliting new drone path at drone node
#         constr8 = (self.NodesDelta[i]<=1 for i in self.DroneCusIndex)
#         self.model.addConstrs(constr8,name='limit_no_drone_landing')
        
#         for i in self.DockingIndex:
#             all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#             constr10_2 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                     (self.M*quicksum(self.TruckPassedNode[p+1][i] for p in range(self.NoTruckPerPath) )) )
#             self.model.addConstr(constr10_2,name='drone_takeoff_nodes_docking'+'_'+str(i))

#         ## Limit number of DRONE taking-off from truck
#         for p in range(self.NoTruckPerPath):
#             constr9_1 = quicksum(self.NodesDelta[i]-self.TruckPassedNode[p+1][i] for i in self.TruckCusIndexDict[p+1])<=self.Lr
#             self.model.addConstr(constr9_1,name='limit_no_drone_takeoff_from_truck'+str(p+1))
            
#             constr9_2 = quicksum(self.NodesBeta[i]-self.TruckPassedNode[p+1][i] for i in self.DockingIndex) <=self.Lr
#             self.model.addConstr(constr9_2,name='limit_no_drone_incoming2dock_on_each_truck'+str(p+1))
        
#             ## DRONE can take-off from customer nodes or docking that truck visited
#             for i in np.concatenate((self.TruckCusIndexDict[p+1],self.DockingIndex)):
#                 if i not in self.DockingIndex: 
#                     all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                     constr10_1 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                             (self.M*self.TruckPassedNode[p+1][i]))
#                     self.model.addConstr(constr10_1,name='drone_takeoff_nodes'+str(p+1)+'_'+str(i))
                
#                 ## EXTRA Eliminate drone reverse path
#                 all_poss_drone_arc_idx_start = self.generateArcsIndexFromNode(node_i_idx=i,vech_type = 'D')
#                 for dock_idx in self.DockingIndex:
#                     if i!= dock_idx:
#                         constr_ext1 = (self.TruckCumulativeDistance[p+1][i]-self.TruckCumulativeDistance[p+1][dock_idx])<=\
#                                         self.M*self.DecisionIfThen[p+1][dock_idx][i]
#                         self.model.addConstr(constr_ext1,name='if_then_1'+str(p+1)+str(i))

#                         constr_ext2 = quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_start)<=\
#                                         self.M*(1-self.DecisionIfThen[p+1][dock_idx][i])
#                         self.model.addConstr(constr_ext2,name='if_then_2'+str(p+1)+str(i))


#         ## Prevent DRONE from visiting customer nodes that truck already visited (treat i as destination node)
#         for i in self.TruckCusIndexDict[1]:
#             drone_cus_node = self.getBothTypeCustomerIndex(i)[1]
#             all_truck_cus_node = self.getBothTypeCustomerIndex(i)[0]
#             all_poss_drone_arc_idx_end = self.generateArcsIndexFromNode(node_j_idx=drone_cus_node,vech_type = 'D')
#             constr11 = (quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_idx_end)<=\
#                         self.M*(1-quicksum(self.TruckPassedNode[p+1][all_truck_cus_node[p]] for p in range(self.NoTruckPerPath) )))
#             self.model.addConstr(constr11,name='drone_landing_nodes'+str(i))
        
#         ## DRONE can land on docking only if truck visited it
#         for d_idx in self.DockingIndex:
#             all_poss_drone_arc_dock_end = self.generateArcsIndexFromNode(node_j_idx=d_idx,vech_type = 'D')
#             constr12 = quicksum(self.DroneArcs[a_idx[0]] for a_idx in all_poss_drone_arc_dock_end) <=\
#                     self.M*quicksum(self.TruckPassedNode[p+1][d_idx] for p in range(self.NoTruckPerPath))
#             self.model.addConstr(constr12,name='drone_landing_docking'+str(d_idx))

#     def convertDroneArc2TruckArcIndex(self,d_a_idx):
#         arc_name_split = self.PathConstructSeries[d_a_idx].split(',')
#         t_arc_name = ','.join(arc_name_split[:2]+['T'])
#         t_a_idx = self.PathConstructSeries[self.PathConstructSeries==t_arc_name].index.values
#         return t_a_idx[0]

#     def mergeDepotCusArcsVar(self,a_var):
#         '''INPUT: ['depot_s,customer_1_T1,T1','customer_2_T1,depot_t,T1']
#         OUTPUT: ['depot,customer_1,T1','customer_2,depot,T1']'''
#         new_a_var =[]
#         for a in a_var:
#             v = a.split(',')
#             if len(v)>1:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0]=self.depot[0]
#                 elif v[0] in self.TruckCusNodes+self.DroneCusNodes: v[0]='_'.join(v[0].split('_')[:-1])
#                 if v[1]==self.depot_s[0] or v[1]==self.depot_t[0]:v[1]=self.depot[0]
#                 elif (v[1] in self.TruckCusNodes+self.DroneCusNodes): v[1]='_'.join(v[1].split('_')[:-1])   
#             else:
#                 if v[0]==self.depot_s[0] or v[0]==self.depot_t[0]:v[0] = self.depot[0]
#             new_a_var.append(','.join(v))
#         return new_a_var   
    
#     def generateCost(self,a_idx):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if 'T' in arcs_name.split(',')[2]:
#             # Convert all depot to depot, calculating distance
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.TruckDistance[a_name]*self.transportation_truck_cost*60/self.truck_speed 
# #             print(arcs_name,cost)
#         elif arcs_name.split(',')[2]=='D' :
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = (self.DroneDistance[a_name]*self.transportation_drone_cost*60/self.drone_speed) 
# #         print(arcs_name)
#         return cost
    
#     def getBothTypeCustomerIndex(self, index):
#         node_name = self.PathConstructSeries[index]
#         customer_name = '_'.join(node_name.split('_')[:-1])+'_'
# #         print(node_name,customer_name)
#         truck_node_index = self.PathConstructSeries[self.PathConstructSeries.isin(self.TruckCusNodes)&self.PathConstructSeries.str.contains(customer_name)].index
# #         print(truck_node_index)
#         drone_node_index = self.PathConstructSeries[self.PathConstructSeries.isin(self.DroneCusNodes)&self.PathConstructSeries.str.contains(node_name[:-2])].index
#         return [list(truck_node_index),drone_node_index[0]]
    
#     def generateObjective(self,_duals):
#         OBJ = quicksum(self.TruckArcs[idx]*self.generateCost(idx) for idx in self.TruckArcsIndex)\
#             + quicksum(self.DroneArcs[idx]*self.generateCost(idx) for idx in self.DroneArcsIndex)\
#             + self.truck_fix_cost*quicksum(self.TruckPassedNode[p+1][self.DepotSIndex[0]] for p in range(self.NoTruckPerPath))\
#             - quicksum( ( self.NodesDelta[self.getBothTypeCustomerIndex(i)[1]] +\
#                         quicksum(self.TruckPassedNode[p+1][self.getBothTypeCustomerIndex(i)[0][p]] for p in range(self.NoTruckPerPath)) )\
#                         *_duals[i]\
#                       for i in self.TruckCusIndexDict[1]) - _duals[-1]*quicksum(self.TruckPassedNode[p+1][self.DepotSIndex[0]] for p in range(self.NoTruckPerPath))
        
#         self.model.setObjective(OBJ,sense = GRB.MINIMIZE)
#         return OBJ
    
#     def getTruckNumberFromRoute(self,path_name,show_value=False):
#         '''path_name: name of the var'''
#         truck_no = quicksum(self.Path.loc[path_name][0][a_idx[0]] for a_idx in self.generateArcsIndexFromNode('T',node_i_idx=self.DepotSIndex[0]) )
#         if show_value: print('Truck no. of path',path_name,':', np.round(truck_no.getValue()))
#         return  np.round(truck_no.getValue())
#     #In case found the new sol that give better reduce cost, 
#     #we will output it and add to new col in master problem
#     def getNewPath(self):
#         return self.model.getAttr('X',self.model.getVars())
    
#     def generateCostOfVars_implicit(self,a_idx,new_vars):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if 'T' in arcs_name.split(',')[2]:
#             # Convert all depot to depot, calculating distance
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.TruckDistance[a_name]*self.transportation_truck_cost*60/self.truck_speed 
#         elif arcs_name.split(',')[2]=='D' :
#             a_name = self.mergeDepotCusArcsVar([arcs_name])[0]
#             cost = self.DroneDistance[a_name]*self.transportation_drone_cost*60/self.drone_speed
# #         print(arcs_name,cost,cost*new_vars[a_idx])
#         return cost
    
#     def getNewPathCost(self,var_name_df):
#         new_vars = var_name_df.values
#         cost = quicksum(new_vars[idx]*self.generateCostOfVars_implicit(idx,new_vars) for idx in self.ArcsIndex)\
#                + self.truck_fix_cost*np.sum([var_name_df[var_name_df.index=='alpha%s[%s]'%(p+1,self.DepotSIndex[0])] for p in range(self.NoTruckPerPath)])
# #         quicksum(self.TruckPassedNode[p+1][self.DepotSIndex[0]] for p in range(self.NoTruckPerPath))
#         return cost
    
#     def solveModel(self, timeLimit = None,GAP=None):
#         self.model.setParam('PoolGap',10)
#         self.model.setParam(GRB.Param.PoolSearchMode, 2)
#         self.model.setParam('TImeLimit', timeLimit)
#         self.model.setParam('MIPGap',GAP)
#         self.model.optimize()
# # FOR EXPLICIT COLUMN GENERATION
#     def generateCostOfVars(self,var,a_idx):
#         arcs_name = self.PathConstructSeries[a_idx]
#         if arcs_name.split(',')[2]=='T':
#             cost = self.TruckDistance[arcs_name]*self.transportation_truck_cost*60/self.truck_speed 
#         elif arcs_name.split(',')[2]=='D' :
#             truck_arc_index = self.convertDroneArc2TruckArcIndex(a_idx)
#             cost = (1-var[truck_arc_index])*(self.DroneDistance[arcs_name]*self.transportation_drone_cost*60/self.drone_speed) 
            
#         return cost

#     def calculateReduceCost(self,var_idx):
#         var = x[x.columns[var_idx]]
#         Cost = quicksum(var[idx]*self.generateCostOfVars(var,idx) for idx in self.ArcsIndex)+truck_fix_cost
#         Pricing = quicksum(var[i]*self.Duals[i] for i in self.CustomerIndex)+self.Duals[-1]
#         return Cost,Pricing

#     #In case found the new sol that give better reduce cost, we will output it and add to new col in master problem
#     def getNewVar(self):
#         return self.added_indices
        
#     def update_processed_indices(self):
#         return self.processed_indices

#     def generatePathWithNegReduceCost(self,):
#         for var in self.unexplored_indices:
#             self.processed_indices.append(var)
#             Cost,Pricing = self.calculateReduceCost(var)
#             reduce_cost = Cost-Pricing
#             print(reduce_cost)
#             if reduce_cost.getValue()<0:
#                 print('REDCUCE COST',reduce_cost,var)
#                 self.added_indices.append([var,Cost.getValue()])