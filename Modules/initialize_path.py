import numpy as np
import random 
import pandas as pd
from itertools import combinations,permutations 
import nltk
import sys
sys.path.insert(0,'/Users/ravitpichayavet/Documents/GaTechOR/GraduateResearch/CTC_CVRP/Modules')
import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
import time
import binpacking
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib import pyplot as plt
import copy



class InitialRouteGenerator:
    def __init__(self,_no_layer, _labeling_dict, customer_demand, constant_dict, distance_matrix):
        # CONSTANTs
        self.no_truck = _no_layer
        self.no_dock = len(_labeling_dict['docking'])
        self.no_customer = len(_labeling_dict['customers'])
        self.customer_demand = customer_demand
        self.distance_matrix = distance_matrix
        self.truck_capacity = constant_dict['truck_capacity']
        self.fixed_setup_time = constant_dict['fixed_setup_time']
        self.truck_speed = constant_dict['truck_speed']        
        
        self.depot = _labeling_dict['depot']
        self.depot_s = _labeling_dict['depot_s']
        self.depot_t = _labeling_dict['depot_t']
        self.all_depot = _labeling_dict['all_depot']
        self.nodes = _labeling_dict['nodes']
        self.customers = _labeling_dict['customers']
        self.arcs = _labeling_dict['arcs']

#     def generateNodeSet(self):
#         self.docking,self.customers,self.depot,self.depot_s,self.depot_t,self.all_depot,self.nodes,self.arcs=vis_sol.create_nodes(self.no_dock,self.no_customer)
#         self.arcs = [','.join(list(l)) for l in self.arcs]
#         self.arcs = self.splitDepotArcsVar(self.arcs,self.depot,self.depot_s,self.depot_t)
#         return self.docking,self.customers,self.depot,self.depot_s,self.depot_t,self.all_depot,self.nodes,self.arcs  
    
    def generateArcs(self):
        #Not nesessary as arcs already generated with nodeSet
        nodes = set()
        nodes = list(nodes.union(set(self.docking),set(self.customers),set(self.depot)))
        no_node = len(nodes)
        node_combi = list(combinations(nodes,2))
        arc_permute = [list(permutations(list(c))) for c in node_combi ]
        sh = np.shape(arc_permute)
        arc_permute = np.reshape(arc_permute,(sh[0]*sh[1],sh[2])).tolist()
        arc_permute = [','.join(list(l)) for l in arc_permute]
#         print(arc_permute)
        arc_permute = self.splitDepotArcsVar(arc_permute,self.depot,self.depot_s,self.depot_t)
        print("Finished creating truck_arcs:",len(arc_permute))
        return arc_permute
    
    def splitDepotArcsVar(self,a_var,depot,depot_s,depot_t):
        '''INPUT: ['depot,customer_1,T','customer_2,depot,T']
        OUTPUT:['depot_s,customer_1,T','customer_2,depot_t,T'] '''
        new_a_var =[]
        for a in a_var:
            v = a.split(',')
            if len(v)>1:
                if v[0]==depot[0]: v[0]=depot_s[0]
                if v[1]==depot[0]: v[1]=depot_t[0]
            new_a_var.append(','.join(v))
        return new_a_var
    
    def mergeDepotArcsVar(self,a_var,depot,depot_s,depot_t):
        '''INPUT: ['depot_s,customer_1,T','customer_2,depot_t,T']
        OUTPUT: ['depot,customer_1,T','customer_2,depot,T']'''
        new_a_var =[]
        for a in a_var:
            v = a.split(',')
            if len(v)>1:
                if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0]=depot[0]
                if v[1]==depot_s[0] or v[1]==depot_t[0]:v[1]=depot[0]
            else:
                if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0] = depot[0]
            new_a_var.append(','.join(v))
        return new_a_var    
    
    def generateAllCombiNodes(self,max_visited_nodes=None,str_node=None,end_node=None):
        '''THIS FUNCTION WILL GENERATE ALL POSSIBLE COMBINATIONS OF INPUT NODE LIST'''
        all_combi_nodes = []
        for i in range(1,max_visited_nodes+1):
            combination_set = list(combinations(self.customers,i))
            permutation_set = [list(permutations(list(c))) for c in combination_set ]
        #     print(permutation_set,np.shape(permutation_set))
            sh = np.shape(permutation_set)
        #     print(np.reshape(permutation_set,(sh[0]*sh[1],sh[2])))
            re_perm_set = np.reshape(permutation_set,(sh[0]*sh[1],sh[2])).tolist()
            all_combi_nodes = all_combi_nodes+re_perm_set
        # ADD depot 
        if (str_node is not None) and (end_node is not None):
            all_combi_nodes = [str_node+p+end_node for p in all_combi_nodes]
        else:all_combi_nodes = [p for p in all_combi_nodes]
        return all_combi_nodes
    
    def generateSetOfArcsFromNodesCombi(self,all_combi_nodes,added_type=''):
        '''THIS FUNCTION WILL GENERATE ARCS LIST BY CREATING BIGRAMS OF INPUT COMBI NODES''' 
        node2arcs = []
        for r in all_combi_nodes:    
            if len(r)==1: arc_list = [r]
            else: arc_list = list(nltk.bigrams(r))
            if added_type!='':arc_list = [','.join(list(a)+[added_type]) for a in arc_list]
            else:arc_list = [tuple(a) for a in arc_list]
            node2arcs.append(arc_list)
        return node2arcs
    
    def generateRoutes(self,initRouteDf,truck_cap_limit=None,
                       max_visited_nodes=None, max_vehicles_per_route=None, 
                       clustered=None, nbInitRoute=None,
                       mode=None,drone_cap_limit=None):
        if max_visited_nodes is None: max_visited_nodes=len(self.customers)
        if max_vehicles_per_route is None: max_vehicles_per_route=len(self.customers)
        if truck_cap_limit is None: truck_cap_limit = self.truck_capacity
        self.all_combi_nodes=list()
        self.routes_arcs=list()
        t1 = time.time()
        
        _all_combi_nodes = self.generateAllCombiNodes(max_visited_nodes,self.depot,self.depot)
        self.all_combi_nodes+= _all_combi_nodes
        self.routes_arcs += self.generateSetOfArcsFromNodesCombi(_all_combi_nodes,'')
        
        if nbInitRoute is None: nbInitRoute = len(self.all_combi_nodes)
        print('nbInitRoute is set to (#UniqueSequences) * (#MaxVehiclesPerRoute) = {}*{} = {}'.format(nbInitRoute,max_vehicles_per_route,nbInitRoute*max_vehicles_per_route))
        
        previous_cols = initRouteDf.columns[initRouteDf.columns.str.contains('r')].shape[0]
        ## ADD NEW COL TO DATAFRAME
        counter = 0
        for idx in range(len(self.all_combi_nodes)):
            if ((idx/len(self.all_combi_nodes))*100 % 10)==0: 
                print('progress:',idx*max_vehicles_per_route,'/',len(self.all_combi_nodes)*max_vehicles_per_route)
#             if idx>nbInitRoute: break
            lr_route = self.calculateLr(self.routes_arcs[idx])
            qr_route = pd.Series(self.all_combi_nodes[idx]).apply(lambda x: self.customer_demand[x]).sum()
            veh_min = int(np.ceil(qr_route*lr_route/truck_cap_limit))
#             print(veh_min)
            if (veh_min > max_vehicles_per_route): continue
            else:
                for veh_no in range(veh_min,max_vehicles_per_route+1):
                    if (veh_no < veh_min)  and (mode!='all'): continue
                    else:
                        self.addNewCol(initRouteDf, lr_route, veh_no,self.all_combi_nodes[idx],self.routes_arcs[idx],'route['+str(counter)+']')
                        counter += 1; #print(veh_no,veh_min)
#         initRouteDf['labels']=init_path.splitDepotArcsVar(initRouteDf.labels,self.depot,[self.all_depot[0]],[self.all_depot[1]])
        self.init_routes_df = initRouteDf.copy()
#         self.init_routes_df = self.init_routes_df.set_index('labels')
        print('#Feasible Cols:', len(initRouteDf.columns)-1)
        print('Elapsed-time:',time.time()-t1)
    
    def generateBasicInitialPatterns(self,nbInitRoute,initRouteDf):
        columns = ['PathCoeff']
        path = pd.DataFrame(index = range(nbInitRoute), columns = columns)
        path['PathCoeff']=[initRouteDf[initRouteDf.columns[idx]].values for idx in range(nbInitRoute)]
        return path
    
    
    def addNewCol(self, df, col_cost, veh_no, nodes, arcs, var_name=None):
        # SET DEFAULT AS INDEXING
        coeff = nodes+arcs
        if var_name is None:var_name = df.columns.shape[0]
            #df[var_name]= np.in1d(df.labels,coeff).astype(int)
        column = df.labels.isin(coeff).astype(int)
        df[var_name]=column
#         print(column)
        df.loc[df['labels']=='m',var_name] = veh_no
        df.loc[df['labels']=='lr',var_name] = col_cost
    
    
    def calculateLr(self, route_arcs):
#         formatted_routes = self.mergeDepotArcsVar(route_arcs,self.depot,self.depot_s,self.depot_t)
        lr = self.fixed_setup_time+(pd.Series(route_arcs).apply(lambda x:self.distance_matrix[x]).sum()/self.truck_speed)
#         print(route_arcs,lr)
        return lr

    def generateInitDF(self, _row_labels,_constant_dict):
        x = pd.DataFrame(data =_row_labels,columns=['labels'])
        self.generateRoutes(initRouteDf=x,
                                   truck_cap_limit=_constant_dict['truck_capacity'],
                                   max_visited_nodes=_constant_dict['max_nodes_proute'], 
                                   max_vehicles_per_route=_constant_dict['max_vehicles_proute'],mode='all')
        x = x.set_index('labels')
        # Reformatting
        feasibleCols = x.shape[1]
        init_route = self.generateBasicInitialPatterns(feasibleCols,initRouteDf=x)
        init_route.rename(index=lambda x:'route[%d]'%x,inplace=True)
        return init_route
    
    def generateInitDFV2(self, _row_labels,_constant_dict):
        n = len(self.customers)
        init_max_npr = _constant_dict['init_max_nodes_proute']
        init_max_mpr = _constant_dict['init_max_vehicles_proute']
        U = [[0]]
        P = []
        init_route_df = pd.DataFrame(data =_row_labels,columns=['labels'])
        counter = 0
        terminate = False
        t1 = time.time()
        while not terminate:
            cS = U.pop(0)
            for j in range(1,n+1):
                if (j not in cS) and (len(cS)-1) < init_max_npr:
                    nS = cS + [j]
                    U.append(nS)
                    nS_lr = 0
                    coeff = []
                    for idx in range(len(nS)):
                        if idx==0: i='O'
                        else: i = 'c_%s'%(nS[idx])
                        if idx==(len(nS)-1): j = 'O'
                        else: j = 'c_%s'%(nS[idx+1])
                        coeff += [i,(i,j)]
                        nS_lr+=self.distance_matrix[(i,j)]/self.truck_speed
        #                     print(coeff)
                    nS_lr+=self.fixed_setup_time
                    nS_qr = 0
                    for c_i in range(1,len(nS)): nS_qr+=self.customer_demand['c_%s'%(nS[c_i])]
                    veh_min = int(np.ceil(nS_qr*nS_lr/self.truck_capacity))
                    if (veh_min > init_max_mpr): continue # not feasible even for p veh
                    else:
                        column = init_route_df.labels.isin(coeff).astype(int)
                        column.iloc[0] = nS_lr
                        for veh_no in range(veh_min,init_max_mpr+1):
                            if ((np.log10(counter+1)) % 0.5)==0: 
                                print('processed route:',counter+1)
                            var_name = 'route['+str(counter)+']'
                            column.iloc[1] = veh_no
                            init_route_df[var_name]=column.values
                            counter+=1
                        
            if len(U)==0: terminate=True
        self.initColsTe = time.time()-t1
        print('Total: %d routes'%(counter-1))
        print('Elapsed Time:',self.initColsTe)
        self.init_routes_df = init_route_df
        init_route = self.generateBasicInitialPatterns(init_route_df.shape[1]-1,initRouteDf=init_route_df.set_index('labels'))
        init_route.rename(index=lambda x:'route[%d]'%x,inplace=True)
        return init_route
    
    def generateInitDFV3wTimeWindow(self, _row_labels,_constant_dict):
        print("Generate Init cols with time window:",_constant_dict['time_window'])
        TW_factor = _constant_dict['tw_avg_factor']
        TW = _constant_dict['time_window']
        n = len(self.customers)
        p = _constant_dict['max_nodes_proute']
        U = [[0]]
        P = []
        init_route_df = pd.DataFrame(data =_row_labels,columns=['labels'])
        counter = 0
        terminate = False
        t1 = time.time()
        while not terminate:
            cS = U.pop(0)
            for j in range(1,n+1):
                if (j not in cS) and (len(cS)-1) < p:
                    nS = cS + [j]
                    U.append(nS)
                    nS_lr = 0
                    coeff = []
                    for idx in range(len(nS)):
                        if idx==0: i='O'
                        else: i = 'c_%s'%(nS[idx])
                        if idx==(len(nS)-1): j = 'O'
                        else: j = 'c_%s'%(nS[idx+1])
                        coeff += [i,(i,j)]
                        nS_lr+=self.distance_matrix[(i,j)]/self.truck_speed
        #                     print(coeff)
                    nS_lr+=self.fixed_setup_time
                    nS_qr = 0
                    for c_i in range(1,len(nS)): nS_qr+=self.customer_demand['c_%s'%(nS[c_i])]
                    veh_min = int(np.ceil(nS_qr*nS_lr/self.truck_capacity))
                    if (veh_min > p): continue # not feasible even for p veh
                    else:
                        column = init_route_df.labels.isin(coeff).astype(int)
                        column.iloc[0] = nS_lr
                        for veh_no in range(veh_min,p+1):
                            #Check time-window feasibility
                            j = 'c_%s'%(nS[-1])
                            _wait_t = TW_factor*nS_lr/veh_no
                            _travel_t = nS_lr-(self.distance_matrix[(j,'O')]/self.truck_speed)
                            if (_wait_t+_travel_t > TW) : continue
                            else:
                                if ((np.log10(counter)) % 0.5)==0: 
                                    print('progress:',counter)
                                var_name = 'route['+str(counter)+']'
                                column.iloc[1] = veh_no
                                init_route_df[var_name]=column.values
                                counter+=1   
            if len(U)==0: terminate=True
        self.initColsTe = time.time()-t1
        print('Elapsed Time:',self.initColsTe)
        self.init_routes_df = init_route_df
        init_route = self.generateBasicInitialPatterns(init_route_df.shape[1]-1,initRouteDf=init_route_df.set_index('labels'))
        init_route.rename(index=lambda x:'route[%d]'%x,inplace=True)
        return init_route

    # def generateInitDFV4wTimeWindow(self, _row_labels,_constant_dict):
    #     print("Generate Init cols with time window:",_constant_dict['time_window'])
    #     TW_factor = _constant_dict['tw_avg_factor']
    #     TW = _constant_dict['time_window']
    #     n = len(self.customers)
    #     p = _constant_dict['init_max_nodes_proute']
    #     U = [[0]]
    #     P = []
    #     init_route_df = pd.DataFrame(data =_row_labels,columns=['labels'])
    #     counter = 0
    #     terminate = False
    #     t1 = time.time()
    #     while not terminate:
    #         cS = U.pop(0)
    #         for j in range(1,n+1):
    #             if (j not in cS) and (len(cS)-1) < p:
    #                 nS = cS + [j]
    #                 U.append(nS)
    #                 nS_lr = 0
    #                 coeff = []
    #                 for idx in range(len(nS)):
    #                     if idx==0: i='O'
    #                     else: i = 'c_%s'%(nS[idx])
    #                     if idx==(len(nS)-1): j = 'O'
    #                     else: j = 'c_%s'%(nS[idx+1])
    #                     coeff += [i,(i,j)]
    #                     nS_lr+=self.distance_matrix[(i,j)]/self.truck_speed
    #     #                     print(coeff)
    #                 nS_lr+=self.fixed_setup_time
    #                 nS_qr = 0
    #                 for c_i in range(1,len(nS)): nS_qr+=self.customer_demand['c_%s'%(nS[c_i])] 
    #                 column = init_route_df.labels.isin(coeff).astype(float)
    #                 column.iloc[0] = nS_lr
    #                 j = 'c_%s'%(nS[-1])
    #                 _travel_t = nS_lr-(self.distance_matrix[(j,'O')]/self.truck_speed)
    #                 veh_min_feas = int(np.ceil(nS_qr*nS_lr/self.truck_capacity))
    #                 veh_min_tw = int(np.ceil((TW_factor*nS_lr)/(TW-_travel_t)))
    #                 if (TW-_travel_t)<0: continue
    #                 else:
    #                     veh_no = max([veh_min_tw,veh_min_feas])
    #                     if ((np.log10(counter+1)) % 0.5)==0: 
    #                         print('progress:',counter)
    #                     var_name = 'route['+str(counter)+']'
    #                     column.iloc[1] = veh_no
    #                     init_route_df[var_name]=column.values
    #                     counter+=1    
    #         if len(U)==0: terminate=True
    #     self.initColsTe = time.time()-t1
    #     print('Elapsed Time:',self.initColsTe)
    #     self.init_routes_df = init_route_df
    #     init_route = self.generateBasicInitialPatterns(init_route_df.shape[1]-1,initRouteDf=init_route_df.set_index('labels'))
    #     init_route.rename(index=lambda x:'route[%d]'%x,inplace=True)
    #     return init_route
    
    def generateInitDFV4wTimeWindow(self, _row_labels, _constant_dict):
        """
        Generates initial feasible routes and returns them as a DataFrame.

        :param _row_labels: List of row labels for the DataFrame.
        :param _constant_dict: Dictionary of constants from ExperimentConfig.
        :return: A DataFrame containing the initial routes.
        """
        print("Generate Init cols with time window:", _constant_dict['time_window'])
        
        tw_factor = _constant_dict['tw_avg_factor']
        tw = _constant_dict['time_window']
        n = len(self.customers)
        p = _constant_dict['init_max_nodes_proute']
        
        route_data = []
        # Use a more efficient deque for the BFS
        from collections import deque
        queue = deque([[0]]) # Use deque for efficient queue operations
        
        t1 = time.time()
        
        while queue:
            current_path = queue.popleft()
            
            for j in range(1, n + 1):
                if (j not in current_path) and (len(current_path) - 1) < p:
                    new_path = current_path + [j]
                    queue.append(new_path)
                    
                    # Original logic for route length and travel time
                    travel_time = 0
                    coefficients = []
                    
                    # Iterate through the path to calculate travel time
                    for idx in range(len(new_path)):
                        if idx == 0:
                            i = 'O'
                        else:
                            i = f'c_{new_path[idx]}'
                            
                        if idx == (len(new_path) - 1):
                            j = 'O'
                        else:
                            j = f'c_{new_path[idx + 1]}'
                        
                        coefficients.extend([i, (i, j)])
                        travel_time += self.distance_matrix[(i, j)] / self.truck_speed
                    
                    # The original logic included the full loop. Let's fix it by subtracting the last leg.
                    last_leg_dist = self.distance_matrix[(f'c_{new_path[-1]}', 'O')]
                    travel_time_without_return = travel_time - (last_leg_dist / self.truck_speed)
                    
                    # The original code's variable `nS_lr` was actually the total time, so we re-add the depot leg.
                    total_route_time = travel_time # This includes the return to depot
                    
                    total_demand = sum(self.customer_demand[f'c_{node}'] for node in new_path[1:])
                    
                    # Original logic for `veh_min_feas` and `veh_min_tw`
                    veh_min_feas = np.ceil(total_demand / _constant_dict['truck_capacity'])
                    
                    # Check for time window feasibility based on travel_time_without_return
                    if (tw - travel_time_without_return) < 0:
                        continue
                        
                    veh_min_tw = np.ceil((tw_factor * (travel_time_without_return+(last_leg_dist / self.truck_speed))) / (tw - travel_time_without_return))
                    
                    veh_no = max(veh_min_tw, veh_min_feas)

                    # Now, build the column data for the DataFrame in one go
                    column_data = {'labels': _row_labels, 'values': [0.0] * len(_row_labels)}
                    
                    # `nS_lr` in the original code seems to have been the total travel time + setup time
                    column_data['values'][0] = total_route_time + self.fixed_setup_time
                    column_data['values'][1] = veh_no
                    
                    for i_idx, i_label in enumerate(coefficients):
                        try:
                            # Handle both nodes and arcs
                            idx = _row_labels.index(i_label)
                            column_data['values'][idx] = 1.0
                        except ValueError:
                            pass
                    
                    # Append the series to the list
                    route_data.append(pd.Series(column_data['values'], index=column_data['labels'], name=f'route[{len(route_data)}]'))

        # Create the DataFrame in a single, efficient operation
        init_route_df = pd.DataFrame(route_data).T.copy()
        # init_route_df.insert(0, 'labels', _row_labels)
        init_route_df = init_route_df.reset_index().rename(columns={'index': 'labels'})
        self.initColsTe = time.time() - t1
        print('Elapsed Time:', self.initColsTe)
        # init_route = init_route_df.set_index('labels')
        self.init_routes_df = init_route_df.copy()
        init_route = self.generateBasicInitialPatterns(init_route_df.shape[1]-1,initRouteDf=init_route_df.set_index('labels'))
        init_route.rename(index=lambda x: f'route[{x}]', inplace=True)        
        return init_route










#################################
########Not yet edited###########

def classifyCustomerNodes(_customer_demand_df,_constant_dict,_drone_distance):
    Nt = _customer_demand_df[(_customer_demand_df>_constant_dict['max_weight_drone'])&(_customer_demand_df.index.str.contains('_T1'))]
    Nt_customer_name = Nt.index.to_series().apply(lambda x: '_'.join(x.split('_')[:-1]))
    print('Nt:\n',Nt)
    
    drone_capable_nodes = _customer_demand_df[(_customer_demand_df<=_constant_dict['max_weight_drone'])&(_customer_demand_df.index.str.contains('_D'))]
    drone_cap_customer_name = drone_capable_nodes.index.to_series().apply(lambda x: '_'.join(x.split('_')[:-1]))
    drone_indy_route = drone_cap_customer_name.apply(lambda x:getDroneDistanceFromDepot(x,_drone_distance)*2<=_constant_dict['max_distance_drone'] )
    Nd = drone_capable_nodes[drone_indy_route]
    print('Nd:\n',Nd)
    
    Nd_complement = drone_capable_nodes[~drone_indy_route].index.to_series().apply(lambda x:'_'.join(x.split('_')[:-1]))
    print('debugged_ndcomplement:\n',Nd_complement)
    
#     remove Nd' from Nt_customer_name 
#     Nt_customer_name = Nt_customer_name[~Nt_customer_name.isin(Nd_complement.values)]
#     [n for n in Nt_customer_name if n not in Nd_complement.values]
    print('Nt_customer_name:\n',Nt_customer_name)
    
    Nd_complement = Nd_complement.apply(lambda x: getTwoNearestNodes(x,_drone_distance,Nt_customer_name))
    Nd_complement_distance_used = Nd_complement.apply(lambda x: np.sum([list(y.values()) for y in x]))

    Nd_tilde = Nd_complement[Nd_complement_distance_used[Nd_complement_distance_used<=_constant_dict['max_distance_drone']].index]
    Nt_tilde = Nd_complement_distance_used[Nd_complement_distance_used>_constant_dict['max_distance_drone']]

    print('Nd_tilde:\n',Nd_tilde)
    print('Nt_tilde:\n',Nt_tilde)
    print('--------------------------------------')
    print('Nd_complement:\n',Nd_complement)
    return Nt,Nd,Nt_tilde,Nd_tilde,Nd_complement
    
def assignDroneRouteQueue(_Nd_tilde):
    drone_route_queue = []
    for d in _Nd_tilde.to_dict().items():
    #     print(d) 
        drone_insertion_dict = dict()
        drone_insertion_dict['drone_served_node'] = [d[0]]#[util.getFormatNodeName(d[0])]
        drone_insertion_dict['insert_aft_depot_s'] = []
        drone_insertion_dict['insert_bef_depot_t'] = []
        drone_insertion_dict['depart_cus'] = []
        depart_node = list(d[1][0].keys())[0]
        land_node = list(d[1][1].keys())[0]
        drone_insertion_dict['depart_node'] = [util.getFormatNodeName(depart_node,affli='s')]
        drone_insertion_dict['landing_node'] = [util.getFormatNodeName(land_node,affli='t')]
    #     print(depart_node,land_node)
        if 'dock' in depart_node:
            drone_insertion_dict['insert_aft_depot_s'].append(depart_node)
        if 'dock' in land_node:
            drone_insertion_dict['insert_bef_depot_t'].append(land_node)
        if 'cus' in depart_node:
            drone_insertion_dict['depart_cus'].append(depart_node)
            drone_route_queue=[drone_insertion_dict]+drone_route_queue
        else: drone_route_queue=drone_route_queue+[drone_insertion_dict]
#     print(drone_route_queue)
    drone_route_pd = pd.Series(drone_route_queue)
    return drone_route_pd

def generateIntegratedTourQueue(_customer_demand_df,_constant_dict,_drone_distance):
    
    Nt,Nd,Nt_tilde,Nd_tilde,Nd_complement = init_path.classifyCustomerNodes(_customer_demand_df,_constant_dict,_drone_distance)
    _drone_route_pd = init_path.assignDroneRouteQueue(Nd_tilde)
    print('\nDrone_Route_pd:\n',_drone_route_pd.to_dict())
    Nt_merged_cus_name = pd.concat((Nt_tilde,Nt)).index.to_series().apply(util.getFormatNodeName).values+'_'
    Nt_merged = _customer_demand_df[(_customer_demand_df.index.str.contains('_T1'))&_customer_demand_df.index.str.contains('|'.join(Nt_merged_cus_name))]
    _truck_bins = binpacking.to_constant_volume(Nt_merged.to_dict(),_constant_dict['max_capacity_truck'])
    _drone_bins = binpacking.to_constant_volume(Nd.to_dict(),_constant_dict['max_weight_drone'])
    print('TruckBins:\n',_truck_bins)
    #print('DroneBins:',_drone_bins)
    integrated_tour_queue = []
    for t_b in _truck_bins:
        tour_dict = dict()
        tour_dict['t_b'] = t_b
        tour_dict['d_attachment']=[]
        tour_dict['insert_aft_depot_s'] = []
        tour_dict['insert_bef_depot_t'] = []

        drone_depart_cus_pd = _drone_route_pd.apply(lambda x:x['depart_cus'])
        mask_truck_visted = drone_depart_cus_pd.apply(lambda x:''.join(x+['_T1']) in t_b.keys())
        mask_not_depart_from_cus = _drone_route_pd.apply(lambda x:len(x['depart_cus'])==0)
        #print(mask_truck_visted)
#         print(drone_depart_cus_pd[mask_truck_visted].values)
        if drone_depart_cus_pd[mask_truck_visted].shape[0]!=0:
            for idx in range(len(_drone_route_pd[mask_truck_visted])):
                added_drone_route = _drone_route_pd[mask_truck_visted].values[idx]
                tour_dict['insert_aft_depot_s']+=added_drone_route['insert_aft_depot_s']
                tour_dict['insert_bef_depot_t']+=added_drone_route['insert_bef_depot_t']
                tour_dict['d_attachment'].append([added_drone_route['depart_node'][0]+'_T1',\
                                             added_drone_route['drone_served_node'][0],\
                                             added_drone_route['landing_node'][0]])
#             print('Depart From Cus:',added_drone_route['depart_node'][0]+'_T1')
#             print(tour_dict)
        if _drone_route_pd[mask_not_depart_from_cus].shape[0]!=0 & len(integrated_tour_queue)==0:
            for other_drone_route in _drone_route_pd[mask_not_depart_from_cus].to_dict().items():
                added_drone_route = other_drone_route[1]
                tour_dict['insert_aft_depot_s']+=added_drone_route['insert_aft_depot_s']
                tour_dict['insert_bef_depot_t']+=added_drone_route['insert_bef_depot_t']
                tour_dict['d_attachment'].append([added_drone_route['depart_node'][0],\
                                             added_drone_route['drone_served_node'][0],\
                                             added_drone_route['landing_node'][0]])
#             print('Depart From Other:',added_drone_route['depart_node'][0])
#             print(tour_dict)
        integrated_tour_queue.append(tour_dict)
    print("\nIntegratedTourQueue:\n",integrated_tour_queue)
    return integrated_tour_queue

def getDroneDistanceFromDepot(_customer_node_name,_drone_distance):
    return _drone_distance['%s,depot,D'%_customer_node_name]

def getTwoNearestNodes(_customer_node_name,_drone_distance,_Nt_customer_name):
    drone_distance_pd = pd.Series(_drone_distance)
    possible_depart_node = drone_distance_pd[drone_distance_pd.index.str.contains(','+_customer_node_name)&(drone_distance_pd!=0)].sort_values()
    _Nt_customer_name = _Nt_customer_name+','
    all_depart_node = _Nt_customer_name.values.tolist()+['depot']+['dock']
    print("all_depart_node",all_depart_node)
    print("possible_depart_node_before_filter:",possible_depart_node)
    possible_depart_node = possible_depart_node[possible_depart_node.index.str.contains('|'.join(all_depart_node))][0:1]
    print("possible_depart_node_aft_filter:",possible_depart_node)
    nearest_depart_node = possible_depart_node.index[0].split(',')[0]+','
    print('Nearest_depart_node:', nearest_depart_node)
    
    land_dock_node = drone_distance_pd[(~drone_distance_pd.index.str.contains(nearest_depart_node))\
                                       &(drone_distance_pd.index.str.contains('%s,dock|%s,depot'%(_customer_node_name,_customer_node_name)))].sort_values()[0:1]
    nearest_dock_node = land_dock_node.index[0].split(',')[1]
    print('Nearest_dock_node:', nearest_dock_node)
    return dict([(nearest_depart_node.strip(','),possible_depart_node[0])]),dict([(nearest_dock_node.strip(','),land_dock_node[0])])


def getDroneCustomer(_customer_demand_df,_constant_dict,_drone_distance):
    result = _customer_demand_df[(_customer_demand_df<=_constant_dict['max_weight_drone'])&(_customer_demand_df.index.str.contains('_D'))]
    customer_name = result.index.to_series().apply(lambda x: '_'.join(x.split('_')[:-1]))
    distance_exceed_mask = customer_name.apply(lambda x:getDroneDistanceFromDepot(x,_drone_distance)*2<_constant_dict['max_distance_drone'] )
    return result[distance_exceed_mask]
# For truck, just compliment of drone set
# def getTruckCustomer(_customer_demand_df):
#     result = _customer_demand_df[(_customer_demand_df>constant_dict['max_weight_drone'])&(_customer_demand_df.index.str.contains('_T1'))]
#     return result

def getComplimentDroneSet(_customer_demand_df,_drone_served_cus):
    customer_name = _drone_served_cus.index.to_series().apply(lambda x: '_'.join(x.split('_')[:-1]))
    result = _customer_demand_df[(~_customer_demand_df.index.str.contains('|'.join(customer_name)))&_customer_demand_df.index.str.contains('T1')]
    return result

# route assignment using binpacking lib
def genInitRouteWithBinPacking(_truck_served_cus,_drone_served_cus):
    truck_routes = binpacking.to_constant_volume(_truck_served_cus.to_dict(),constant_dict['max_capacity_truck'])
    drone_routes = binpacking.to_constant_volume(_drone_served_cus.to_dict(),constant_dict['max_weight_drone'])



class initSetWithKMean:
    def __init__(self, _nodes_position, _customer_demand_df, _no_dock, _truck_cap_limit):
        self.nodes_position = _nodes_position.copy()
        del self.nodes_position['depot']
        for d in range(_no_dock): del self.nodes_position['dock_%s'%(d+1)]
        self.customer_demand_df=_customer_demand_df
        self.color_code = 'bgrcmyk'
        self.truck_cap_limit =_truck_cap_limit
    
    def kmeanClustering(self, xy_coordinate, point_name, no_clusters=None):
        if no_clusters is None: no_clusters=2
        # data = np.array(list(self.nodes_position.values()))
        kmeans = KMeans(n_clusters=no_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(xy_coordinate)
        
        self.clustered_pd_coor = pd.Series(xy_coordinate.tolist(),index=pred_y)
        self.clustered_pd = pd.Series(pred_y, index=point_name)
        
        nodes_groups = []
        for i in range(no_clusters):
            plt_data = self.clustered_pd_coor[self.clustered_pd_coor.index==i].values.tolist()
            plt_data = np.reshape(plt_data, (len(plt_data),2))
            plt.scatter(plt_data[:,0], plt_data[:,1],c=self.color_code[i])
            nodes_groups.append(self.clustered_pd[self.clustered_pd==i].index.tolist())
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.show()
        return nodes_groups
    
    def generateInitialSet(self):
#         data = np.array(list(self.nodes_position.values()))
        point_name = list(self.nodes_position.keys())
        to_be_clustered = [point_name]
        clustered = []
        one_point_cluster = []
        while len(to_be_clustered)>0:
            chosen_nodes = to_be_clustered[0].copy()
            print('CHOSEN_NODES:',chosen_nodes)
            del to_be_clustered[0]
            data = np.transpose(pd.DataFrame(self.nodes_position)[chosen_nodes].values)
        
            result = self.kmeanClustering(data, chosen_nodes)
            print(result)
            for r in result:
                truck1_r = [x+'_T1' for x in r]
                print(r,self.customer_demand_df[truck1_r].sum())
                if (len(r)>6) or (self.customer_demand_df[truck1_r].sum()>self.truck_cap_limit): to_be_clustered.append(r)
                elif len(r)==1: one_point_cluster.append(r)
                else: clustered.append(r)
        if len(one_point_cluster)!=0 :
            for one in one_point_cluster:
                if len(clustered)>1:
                    added=False
                    for i_clus in range(len(clustered)):
                        truck1_r = [x+'_T1' for x in clustered[i_clus]+one]
                        if (self.customer_demand_df[truck1_r].sum()<self.truck_cap_limit):
                            clustered[i_clus]+=one
                            added=True
                            break
                    if not added: clustered.append(one)
                else: clustered.append(one)
        return clustered





def generate_all_combi_nodes(nodes_list,str_node=None,end_node=None):
    '''THIS FUNCTION WILL GENERATE ALL POSSIBLE COMBINATIONS OF INPUT NODE LIST'''
    possible_t_node = nodes_list
    # possible_t_node = customers
    all_combi_nodes = []
    for i in range(1,len(possible_t_node)+1):
        combination_set = list(combinations(possible_t_node,i))
        permutation_set = [list(permutations(list(c))) for c in combination_set ]
    #     print(permutation_set,np.shape(permutation_set))
        sh = np.shape(permutation_set)
    #     print(np.reshape(permutation_set,(sh[0]*sh[1],sh[2])))
        re_perm_set = np.reshape(permutation_set,(sh[0]*sh[1],sh[2])).tolist()
        all_combi_nodes = all_combi_nodes+re_perm_set
    # ADD depot 
    if (str_node is not None) and (end_node is not None):
        all_combi_nodes = [str_node+p+end_node for p in all_combi_nodes]
    else:all_combi_nodes = [p for p in all_combi_nodes]
    return all_combi_nodes

def generate_arcs_from_node_combi(all_combi_nodes,added_type=''):
    '''THIS FUNCTION WILL GENERATE ARCS LIST BY CREATING BIGRAMS OF INPUT COMBI NODES''' 
    node2arcs = []
    for r in all_combi_nodes:    
        if len(r)==1: arc_list = [r]
        else: arc_list = list(nltk.bigrams(r))
        if added_type!='':arc_list = [','.join(list(a)+[added_type]) for a in arc_list]
        else:arc_list = [','.join(a) for a in arc_list]
        node2arcs.append(arc_list)
    return node2arcs

def generate_all_permutation_of_nodes_list(clusters_node,str_node=None,end_node=None):
    '''THIS FUNCTION WILL GENERATE ALL POSSIBLE COMBINATIONS OF INPUT NODE LIST
    c_i,c_f: no of nodes to combi; must not exceed len(nodes_list)'''
    re_perm_set=[]
    for clus in clusters_node:
        possible_t_node = clus
        permutation_set = list(permutations(clus))
        sh = np.shape(permutation_set)
        print('Cluster:',clus,sh)
        re_perm_set+=np.reshape(permutation_set,(sh[0],sh[1])).tolist()
    if (str_node is not None) and (end_node is not None):
        all_combi_nodes = [str_node+p+end_node for p in re_perm_set]
    else:all_combi_nodes = [p for p in re_perm_set]
    return all_combi_nodes

def generate_all_combi_nodes_custo(nodes_list,c_i,c_f,str_node=None,end_node=None):
    '''THIS FUNCTION WILL GENERATE ALL POSSIBLE COMBINATIONS OF INPUT NODE LIST
    c_i,c_f: no of nodes to combi; must not exceed len(nodes_list)'''
    possible_t_node = nodes_list
    # possible_t_node = customers
    all_combi_nodes = []
#     for i in range(1,len(possible_t_node)+1):
    for i in range(c_i,c_f+1):
        combination_set = list(combinations(possible_t_node,i))
        permutation_set = [list(permutations(list(c))) for c in combination_set ]
    #     print(permutation_set,np.shape(permutation_set))
        sh = np.shape(permutation_set)
    #     print(np.reshape(permutation_set,(sh[0]*sh[1],sh[2])))
        re_perm_set = np.reshape(permutation_set,(sh[0]*sh[1],sh[2])).tolist()
        all_combi_nodes = all_combi_nodes+re_perm_set
        print(i,':%d'%len(all_combi_nodes))
    # ADD depot 
    if (str_node is not None) and (end_node is not None):
        all_combi_nodes = [str_node+p+end_node for p in all_combi_nodes]
    else:all_combi_nodes = [p for p in all_combi_nodes]
    return all_combi_nodes

def generate_arcs_from_node_combi_custo(all_combi_nodes,added_type=''):
    '''THIS FUNCTION WILL GENERATE ARCS LIST BY CREATING BIGRAMS OF INPUT COMBI NODES''' 
    node2arcs = []
    for r in all_combi_nodes:    
        if len(r)==1: arc_list = [r]
        else: arc_list = list(nltk.bigrams(r))
        if added_type!='':arc_list = [','.join(list(a)+[added_type]) for a in arc_list]
        else:arc_list = [','.join(a) for a in arc_list]
        node2arcs.append(arc_list)
    return node2arcs

def add_new_col(df, nodes, arcs, var_name=None):
    # SET DEFAULT AS INDEXING
    coeff = nodes+arcs
    if var_name is None:
        var_name = df.columns.shape[0]
    df[var_name]= np.in1d(df.labels,coeff).astype(int)

def add_new_col_model2(df, nodes, arcs, var_name=None):
    # SET DEFAULT AS INDEXING
    # delta don't care which type of nodes, it count only the split out arcs (this case only truck arcs is cosidered, count only!)
#     nodes = [n[:-2] for n in nodes if (n in drone_cus_nodes+truck_cus_nodes)]
    coeff = nodes+arcs
#     print(coeff)
#     print(df.labels)
    if var_name is None:
        var_name = df.columns.shape[0]
    new_col = []
    for i in df.labels:
        if( i in coeff):new_col.append(np.in1d(coeff,i).sum())
        else: new_col.append(0)
    df[var_name] = new_col    
    
    
def splitDepotArcsVar(a_var,depot,depot_s,depot_t):
    '''INPUT: ['depot,customer_1,T','customer_2,depot,T']
    OUTPUT:['depot_s,customer_1,T','customer_2,depot_t,T'] '''
    new_a_var =[]
    for a in a_var:
        v = a.split(',')
        if len(v)>1:
            if v[0]==depot[0]: v[0]=depot_s[0]
            if v[1]==depot[0]: v[1]=depot_t[0]
        new_a_var.append(','.join(v))
    return new_a_var

def mergeDepotArcsVar(a_var,depot,depot_s,depot_t):
    '''INPUT: ['depot_s,customer_1,T','customer_2,depot_t,T']
    OUTPUT: ['depot,customer_1,T','customer_2,depot,T']'''
    new_a_var =[]
    for a in a_var:
        v = a.split(',')
        if len(v)>1:
            if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0]=depot[0]
            if v[1]==depot_s[0] or v[1]==depot_t[0]:v[1]=depot[0]
        else:
            if v[0]==depot_s[0] or v[0]==depot_t[0]:v[0] = depot[0]
        new_a_var.append(','.join(v))
    return new_a_var    
    

def create_arcs_truck(no_dock,no_customer,truck_cus_nodes,docking):
    nodes = set()
    docking = ['dock_'+str(i+1) for i in range(no_dock)]
    customers = ['customer_'+str(i+1) for i in range(no_customer)]
    depot = ['depot']
    depot_s = ['depot_s']
    depot_t = ['depot_t']
    nodes = list(nodes.union(set(docking),set(customers),set(depot)))
    no_node = len(nodes)
    # print(all_cus_nodes_types)
    # TRUCK ARCS IS THE ARCS THAT ASSOCIATE WITH TRUCK NODE ONLY
    tag = [truck_cus_nodes[0].split('_')[-1]]
    truck_node_combi = list(combinations(truck_cus_nodes+depot+docking,2))
    arc_truck_permute = [list(permutations(list(c))) for c in truck_node_combi ]
    sh = np.shape(arc_truck_permute)
    arcs_truck = np.reshape(arc_truck_permute,(sh[0]*sh[1],sh[2])).tolist()
    arcs_truck = [','.join(list(l)+tag) for l in arcs_truck]
    arcs_truck = splitDepotArcsVar(arcs_truck,depot,depot_s,depot_t)
    print("Finished creating truck_arcs:",len(arcs_truck))
    return arcs_truck

def create_arcs_drone(no_dock,no_customer,truck_cus_nodes,drone_cus_nodes,docking):
    depot = ['depot']
    depot_s = ['depot_s']
    depot_t = ['depot_t']
    drone_node_combi = list(combinations(drone_cus_nodes+truck_cus_nodes+depot+docking,2))
    ## i or j isin DroneNodes, j isin DroneNodes U Docking, cus i != cus j
    # eliminate !(i or j isin DroneNodes) == (i and j isin TruckNodes)
    # eliminate node_name(i)==node_name(j)
    temp_d_combi =[]
    print("Creating drone arcs from drone_node_combi..",len(drone_node_combi))
    for n_combi in drone_node_combi:
        # help to reduce the set that going to permute in the next step!
        if (n_combi[0] in truck_cus_nodes): continue
        if ('_'.join(n_combi[0].split('_')[:-1])=='_'.join(n_combi[1].split('_')[:-1])): continue
            
            
        temp_d_combi.append(n_combi)
    # print('====')
    # print(temp_d_combi)
    arc_drone_permute = [list(permutations(list(c))) for c in temp_d_combi ]
    sh = np.shape(arc_drone_permute)
    arcs_drone = np.reshape(arc_drone_permute,(sh[0]*sh[1],sh[2])).tolist()
    # Only arcs that j isin DroneNodes U Docking
    arcs_drone = [','.join(list(l)+['D']) for l in arcs_drone if (l[1] in drone_cus_nodes+depot+docking)]
    arcs_drone = splitDepotArcsVar(arcs_drone,depot,depot_s,depot_t)
    # print('\n',arcs_drone)
    print("Finished creating drone_arcs:",len(arcs_drone))
    return arcs_drone
# arcs_truck = create_arcs_truck(no_dock,no_customer)
# arcs_drone = create_arcs_drone(no_dock,no_customer)
