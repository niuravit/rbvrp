import sys
sys.path.insert(0,'/Users/ravitpichayavet/Documents/GaTechIE/GraduateResearch/CTC_CVRP/Modules')
import os
os.environ['GRB_LICENSE_FILE'] = '/Users/ravitpichayavet/gurobi.lic'
import numpy as np
from itertools import combinations,permutations 
import random 
import pandas as pd
import pickle as pk
import pybnb
import gurobipy as gp
from copy import deepcopy
from math import ceil,floor
from datetime import datetime
epsilon = 1e-6
import time
import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
import model as md




class MinimumFleetSizeWithTimeWindowBnP(pybnb.Problem):
    def __init__(self, _dist_mat, _initializer, _init_route, _const_dict, _chDom = True):
        self.inst_dist_mat = _dist_mat
        self.initializer = _initializer
        self.constant_dict = _const_dict
        self.b_cond_log = []; self.del_pats = []
        self.lp_sol = None
        self.loc_bound = None
        # some formatting
        df_lab = _initializer.init_routes_df.set_index("labels");
        scc_cols = df_lab.loc[df_lab.index.isin(_initializer.arcs),:]
        self.route_pats = {r_name: {k:v for k,v in r_dict.items() if v>0} for r_name, r_dict in scc_cols.to_dict().items()} 
        # Root node rmp
        self.rmp_initializer_model = md.timeWindowModel(_init_route, _initializer,
             _dist_mat,_const_dict, _relax_route=True)
        self.rmp_initializer_model.buildModel();
        self.rmp_initializer_model.model.setParam('OutputFlag',False)
        self.rmp_model = self.rmp_initializer_model.model.copy()
        self.rmp_init_df = deepcopy(_initializer.init_routes_df)
        
        # Network
        self.label = _initializer.init_routes_df["labels"];
        self.arcs = self.rmp_initializer_model.arcs
        self.arcs_drp_org = [a for a in self.arcs if 'O' not in a]
        self.arcs_index = self.rmp_initializer_model.arcs_index
        self.nodes = self.rmp_initializer_model.nodes
        
        self.is_root_node = True
        self.ch_dom = _chDom
        self.node_count = 0
        
        # [ip_obj, route_pats, model, init_df, node_count]
        self.best_node = [1e10, self.route_pats, self.rmp_model, self.rmp_init_df, self.node_count ]

    def sense(self):
        return pybnb.minimize
    
    def objective(self):
        if (self.loc_bound < 1e10) and (self.loc_bound > -1e10):
            lp_obj = sum([p.Obj*self.lp_sol[p.varName] for p in self.rmp_model.getVars()] )
            temp_m_ip_bound = self.rmp_initializer_model
            temp_m_ip_bound.model = self.rmp_model.copy()
            temp_m_ip_bound.shortCuttingColumns()
            temp_m_ip_bound.model.update()
            temp_m_ip_bound.solveModel()
#             ip_obj = sum([p.Obj for p in temp_m_ip_bound.relaxedBoundedModel.getVars() if p.X>0] )
            ip_obj = temp_m_ip_bound.model.ObjVal
            print("LP/IP:",lp_obj,ip_obj)
            p_vars = temp_m_ip_bound.model.getVars()
            _route_pats_shortcut = dict( [ (p.varName ,  
                                       dict( [(self.label[a_idx], 
                                               temp_m_ip_bound.init_routes_df[p.varName][a_idx])
                                                for a_idx in self.arcs_index 
                                                    if temp_m_ip_bound.init_routes_df[p.varName][a_idx]>0] )
                                                      ) for p in p_vars ] )
            print("IP SOLUTIONS:",[[p.varName, p.x, _route_pats_shortcut[p.varName]] for p in p_vars if p.x >0])
            # update best node:
            if ip_obj<self.best_node[0]:
                print("Best Node found!")
                self.best_node = [ip_obj, _route_pats_shortcut,temp_m_ip_bound.model, temp_m_ip_bound.init_routes_df, self.node_count ]
            return ip_obj
        else:
            return 1e10
    def bound(self):
        # update bound from column generation
        self.loc_bound, self.route_pats, self.lp_sol, self.rmp_model, self.rmp_init_df, self.node_count = SolveMinFleetWithTimeWindowNode(self)
        if (self.loc_bound-np.floor(self.loc_bound))>1e-6:
            new_lb = np.ceil(self.loc_bound)
        else: new_lb = np.floor(self.loc_bound)
        print("==Called bound(), Update LOCAL BOUND:",self.loc_bound, "ceil to", new_lb) 
        return new_lb

    # 1st-called
    def save_state(self, node): 
        # node structure: 
        #. |---- route_pats:[r1,r2,..], r1 = {<cus_idx>: # of visits}
        #. |____ b_cond_log:[((c1,c2),0),((c1,c3),1),..], (c1,c2) = <arc begin branched on>
        node.state = (self.route_pats, self.b_cond_log, self.del_pats, self.rmp_model, self.rmp_init_df)

    def load_state(self, node):
        (self.route_pats, self.b_cond_log, self.del_pats, self.rmp_model, self.rmp_init_df) = node.state
        
    def branch(self):
        _n = self.initializer.no_customer #no. node
        _nodes = self.nodes
        # From Node
        _route_pats = self.route_pats
        _b_cond_log = self.b_cond_log
            # _del_pats = self.del_pats #don't need will create new here
        _cur_rmp_model = self.rmp_initializer_model
        _cur_rmp_model.model = self.rmp_model.copy() # Import node model
        
        # for braching
        _temp_model = self.rmp_initializer_model
        _temp_model.model = self.rmp_model.copy()
#         _temp_model.shortCutColumnsBnp()
        _temp_model.solveRelaxedBoundedModel()
        _m_constrs = _temp_model.model.getConstrs()
        p_vars = _temp_model.relaxedBoundedModel.getVars() #Load current node's model
        _objVal = _temp_model.relaxedBoundedModel.ObjVal

#         print(_route_pats)
#         print("Temp model Sol:",[[p.varName, p.x,_route_pats[p.varName]] for p in p_vars if p.x >0])
        
        # get the pair vars
        # find the first fractional variable and branch
        frac = None
        val = None
        gap = None
        print("==Called branch, branching with cond:",_b_cond_log)
        _not_bch_arcs =[]
        for a in _b_cond_log:
            _not_bch_arcs+=[a[0]]
            if a[0][0]!='O' and a[0][1]!='O': 
                if a[1]==1: _not_bch_arcs+=[(a[0][1],a[0][0])]
#                 _not_bch_arcs+=[(a[0][0],j) for j in _nodes if j!=a[0][0]]
            
#         print("Len frac_sol:",len(frac_sol),"Len route_pats:", len(route_pats))
        # Infeasibility condition trickered when penalized route being selected
        if (_cur_rmp_model.relaxedBoundedModel.status != 2):
            print("Fathomed by infeasibility")
            return []
        else:
            _lp_sol = dict([(p.varName,(p.x)) for p in p_vars])
#             _route_pats = dict( [ (p.varName, dict( [(c_idx+1,_temp_model.model.getCoeff(_m_constrs[c_idx],p))
#                                                 for c_idx in range(_n)] )
#                                                       ) for p in p_vars ] )
            print("===Len lp_sol:",len(_lp_sol),"Len route_pats:", len(_route_pats)) 
#             print(_lp_sol, _route_pats)
            route_dict = dict(); arc_score_dict = dict();
            frac_dict = dict(); route_incident_dict = dict();
            r_w_cycle = []
#             for arc in self.arcs_drp_org:
            for arc in self.arcs:
                val = 0
#                 if arc not in _not_bch_arcs:
                for r_name, r_dict in _route_pats.items():     
                    if (arc in r_dict.keys()) and (_lp_sol[r_name]>0) and (arc not in _not_bch_arcs):
#                             print(r_dict[arc],_lp_sol[r_name])
                        if (r_dict[arc]>0):
#                             print(r_name,arc,_lp_sol[r_name])
                            val += _lp_sol[r_name]*r_dict[arc]
                            # route contains cycle visiting arc > 1
                            if (r_dict[arc]>1) and (r_name not in r_w_cycle): 
                                r_w_cycle.append(r_name)
                        arc_score_dict[arc] = val 
                if (abs(val - np.floor(val)) > epsilon) and (val!=1):
#                 if val!=1:
#                     gap = abs(val - np.floor(val))
                    if "O" in arc: frac_dict[arc] = val
                    else: frac_dict[arc] = val

            bch_arcs_dict = dict(); bch_arcs=[]
            print("r_w_cycle",r_w_cycle)
            print("frac_dict",frac_dict)
            if (len(r_w_cycle) > 0):
                # Build node incident dict of route
                for r_name in r_w_cycle:
                    incident_dict = dict(zip([x for x in _nodes],[0]*(_n+1)))
                    for arc in _route_pats[r_name].keys():
                        i = arc[0]; j = arc[1];
                        incident_dict[i]+=1*_route_pats[r_name][arc]
                        incident_dict[j]+=1*_route_pats[r_name][arc]
#                         print(arc, incident_dict)
                    for arc in _route_pats[r_name].keys():
                        if (arc not in bch_arcs) and (arc not in _not_bch_arcs):
                            if (i=="O") and (incident_dict[j]>1): bch_arcs += [(arc)]
                            elif (j=="O") and (incident_dict[i]>1): bch_arcs += [(arc)]
                            elif (incident_dict[i]>1 or incident_dict[j]>1): bch_arcs += [(arc)]
                    print("r_name",r_name, "Incident-dict",incident_dict)
#                 for r_name in r_w_cycle:
#                     bch_arcs += _route_pats[r_name].keys()
                print("bch_arcs",bch_arcs)
                for a in bch_arcs: 
                    if ((a in self.arcs) and (a not in _not_bch_arcs)):
                        # Avoid branch on arc with O to help speed up
                        if (("O" in a) and (arc_score_dict[a]!=1)): bch_arcs_dict[a] = arc_score_dict[a]
                        else: bch_arcs_dict[a] = arc_score_dict[a] 
#                 bch_arcs_dict = dict([(a,arc_score_dict[a] ) for a in bch_arcs if ((a in self.arcs) and (a not in _not_bch_arcs))])
                print("bch_arcs_dict",bch_arcs_dict)
                if len(bch_arcs_dict.keys())>0:
                    frac = max(bch_arcs_dict, key = bch_arcs_dict.get)
#                     frac = random.choice(list(bch_arcs_dict.keys()))
                else:
                    frac = max(frac_dict, key = frac_dict.get)
#                     frac = random.choice(list(frac_dict.keys()))
            elif (len(frac_dict.keys())>0):
                frac = max(frac_dict, key = frac_dict.get)
#                 frac = random.choice(list(frac_dict.keys()))
#             print("_route_pats",_route_pats)
            
#             print("arc_score_dict",arc_score_dict)
            
            print("frac",frac)
            if frac!=None:
                print("===branching on arc/val", frac)
                # from parent
                r_b_cond_log = deepcopy(_b_cond_log)
                r_b_cond_log.append([frac,1])
                r_route_pats = deepcopy(_route_pats)
                if "O" not in frac: r_b_cond_log.append([(frac[1],frac[0]),0])

                l_b_cond_log = deepcopy(_b_cond_log)
                l_b_cond_log.append([frac,0])
                l_route_pats = deepcopy(_route_pats)
    #             print(route_pats.keys())
                # yikes look at that iterable
                r_del = []
                l_del = []
                for indx in list(reversed(_route_pats.keys())):
#                     i,j = frac[0], frac[1]
                    route_pat = _route_pats[indx]
#                     if len(route_pat.keys())==2: continue
                    if frac in route_pat.keys(): 
                        # pat contains arc, delete it from (0) l-branch 
#                         print("0-LEFT-BCHING:",indx, route_pat)
                        l_del.append(indx)
#                         l_route_pats.pop(indx) #remove from 0 branch
                    else: 
                        # pat does not contains arc, delete it from (1) r-branch 
#                         print("1-RIGHT-BCHING:",indx, route_pat)
                        route_pat_nodes = [a[1] for a in list(route_pat.keys())]
                        
                        if (frac[0] == "O"): # and (frac[1] in route_pat_nodes)):
                            if (frac[1] in route_pat_nodes):
                            # Not having (0,j) but have some (i,j) i!=0
                                r_del.append(indx) # penalize from 1 branch
                        elif (frac[1] == "O"):# and (frac[0] in route_pat_nodes)):
                            if (frac[0] in route_pat_nodes):  
                            # Not having (i,0) but have some (i,j) j!=0
                                r_del.append(indx) # penalize from 1 branch
                        elif ( ((frac[0] in route_pat_nodes)) or ((frac[1] in route_pat_nodes)) ):
                            # Not having (i,j) but have some (i,k) k!=j or (k,j) k!=i
                            r_del.append(indx) # penalize from 1 branch
                l_child = pybnb.Node()
                l_child.state = (l_route_pats, l_b_cond_log, l_del,_cur_rmp_model.model.copy(),deepcopy(self.rmp_init_df))
                r_child = pybnb.Node()
                r_child.state = (r_route_pats, r_b_cond_log, r_del,_cur_rmp_model.model.copy(),deepcopy(self.rmp_init_df))
#                 print("Len l-branch vars:", len(l_route_pats),"Len r-branch vars:", len(r_route_pats))
                return l_child,r_child
            else: 
                return []
        
        
def SolveMinFleetWithTimeWindowNode(cTCCVRP_mt):
    # Instance 
    _n = cTCCVRP_mt.initializer.no_customer #no. node
    _dist_mat = cTCCVRP_mt.inst_dist_mat
    _initializer = cTCCVRP_mt.initializer
    _const_dict = cTCCVRP_mt.constant_dict
    _node_count = cTCCVRP_mt.node_count
    _chDom = cTCCVRP_mt.ch_dom
    # From Node
    _route_pats = cTCCVRP_mt.route_pats
    _b_cond_log = cTCCVRP_mt.b_cond_log
    _del_pats = cTCCVRP_mt.del_pats
    _tWLP_node = cTCCVRP_mt.rmp_initializer_model
    _tWLP_node.init_routes_df = deepcopy(cTCCVRP_mt.rmp_init_df)
    _tWLP_node.model = cTCCVRP_mt.rmp_model.copy() # Start with parent's node model
    
    print("==Branching Condition:", _b_cond_log)

    # apply branching to parent's column pool: filter out del_pats
#     print(_route_pats)
    
    # DELETING 
#     _tWLP_node.init_routes_df.drop(columns=_del_pats,inplace=True)
#     for v in _tWLP_node.model.getVars():
#         if v.varName in _del_pats: 
#             _tWLP_node.model.remove(v)
    # INCIDENT DICT
    necess_link = [bh[0] for bh in _b_cond_log if (bh[1]==1)] # 1-branch
    incident_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    
    _bch_conflict = False
    _inflow_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    _outflow_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    for arc in necess_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
#         incident_dict[i]+=1; incident_dict[j]+=1
        if i!=0 and j!=0: _inflow_dict[j]+=1; _outflow_dict[i]+=1;
        if (_outflow_dict[i]>1 and i!=0) or (_inflow_dict[j]>1 and j!=0): 
            _bch_conflict = True; break;
    print("necess_link",necess_link)
    print("_inflow_dict",_inflow_dict)
    print("_outflow_dict",_outflow_dict)
    
    # PENALIZING
#     _tWLP_node.init_routes_df.drop(columns=_del_pats,inplace=True)
    m_lab_idx = cTCCVRP_mt.label.loc[cTCCVRP_mt.label=='m'].index[0]
    for v in _tWLP_node.model.getVars():
        if v.varName in _del_pats: 
            _tWLP_node.init_routes_df.loc[m_lab_idx,v.varName] = 1e10
            v.Obj = 1e10
#             _route_pats.pop(v.varName)
#     print("==Solving node, with braching conds list:",_b_cond_log)
#     print("==Len Vars:",len(_tWLP_node.model.getVars()))
#     print("==Del Pats:",_del_pats)
    _tWLP_node.model.update()
#     print("DFAFTERDROP:",_tWLP_node.init_routes_df.columns)
#     print("MODELVARS:",_tWLP_node.model.getVars())
    
    # check if model is infeasible
    # if so, return inf
    _tWLP_node.solveRelaxedBoundedModel()
    mrelax_obj = _tWLP_node.relaxedBoundedModel.ObjVal
    print("==Model's status:",_tWLP_node.relaxedBoundedModel.status)
    if (_tWLP_node.relaxedBoundedModel.status != 2) or (mrelax_obj >= 1e9) or (_bch_conflict):
        print("==Prunned by INFEASIBILITY, mrx_obj, bch_conflict:",mrelax_obj,_bch_conflict )
        return 1e10, None, None, None, None,_node_count
    
    
    # COLUMN GENERATION!: build pricing model
    # add branching constraint to pricing sp model
    
    ########Pricing###########
    t1 = time.time()
    _tWLP_node.runColumnsGeneration(None,_pricing_status=False,
            _check_dominance=_chDom,_dominance_rule=4,_DP_ver="SIMUL_M",
            _time_limit=_const_dict['dp_time_limit'],_update_m_ub=False,_filtering_mode="BestRwdPerI",
            _bch_cond = _b_cond_log,_node_count_lab = str(_node_count))
    colGen_te = time.time()-t1
    # update _route_pats & objval
    _tWLP_node.solveRelaxedBoundedModel()
    mrelax_obj = _tWLP_node.relaxedBoundedModel.ObjVal
    p_vars = _tWLP_node.relaxedBoundedModel.getVars()
    if cTCCVRP_mt.is_root_node:
        print("==THIS IS ROOT NODE!:", cTCCVRP_mt.is_root_node); cTCCVRP_mt.is_root_node = False;
        cTCCVRP_mt.root_node = [mrelax_obj, _tWLP_node.model, _tWLP_node.init_routes_df, colGen_te,_tWLP_node.colgenLogs]
    else: 
        print("==THIS IS NODE:",_node_count)
    
    
    print("LP-ColGen OBJ:",_tWLP_node.relaxedBoundedModel.ObjVal)
    print("ModelVars, DataFrameVars :", len(p_vars),len(_tWLP_node.init_routes_df.columns)-1)
    
    _route_pats = dict( [ (p.varName ,   dict( [(cTCCVRP_mt.label[a_idx], _tWLP_node.init_routes_df[p.varName][a_idx])
                                                    for a_idx in cTCCVRP_mt.arcs_index 
                                                        if _tWLP_node.init_routes_df[p.varName][a_idx]>0] )
                              ) for p in p_vars ] )
#         mrelax_obj = _tWLP_node.relaxedBoundedModel.ObjVal
#         scc_cols = _tWLP_node.init_routes_df.loc[_tWLP_node.customer_index,:].drop(columns = "labels")
#         scc_cols['new_index'] = range(1,_n+1)
#         _route_pats = scc_cols.set_index('new_index').to_dict()
#         p_vars = _tWLP_node.relaxedBoundedModel.getVars()
    print("==Obj-val colgen:", mrelax_obj)
#     print("==Del Pats:",_del_pats)
    print("==Branching Condition:", _b_cond_log)
    print([[p.varName, p.x, _route_pats[p.varName]] for p in p_vars if p.x >0])
    lp_sol = dict([(p.varName,(p.x)) for p in p_vars])
    _node_count+=1
    return mrelax_obj, _route_pats, lp_sol, _tWLP_node.model.copy(), deepcopy(_tWLP_node.init_routes_df) ,_node_count
        



        
class MinimumAverageTimeSpentBnP(pybnb.Problem):
    def __init__(self, _dist_mat, _initializer, _init_route, _const_dict, _chDom = True, _acc_flag = None, _dom_rule=None):
        self.inst_dist_mat = _dist_mat
        self.initializer = _initializer
        self.constant_dict = _const_dict
        self.b_cond_log = []; self.del_pats = []
        self.lp_sol = None
        self.loc_bound = None
        # some formatting
        df_lab = _initializer.init_routes_df.set_index("labels");
        scc_cols = df_lab.loc[df_lab.index.isin(_initializer.arcs),:]
        self.route_pats = {r_name: {k:v for k,v in r_dict.items() if v>0} for r_name, r_dict in scc_cols.to_dict().items()} 
        # Root node rmp
        self.rmp_initializer_model = md.phaseIIModel(_init_route, _initializer,
             _dist_mat,_const_dict, _relax_route=False)
        self.rmp_initializer_model.buildModel();
        self.rmp_initializer_model.model.setParam('OutputFlag',False)
        self.rmp_model = self.rmp_initializer_model.model.copy()
        self.rmp_init_df = deepcopy(_initializer.init_routes_df)
        
        # Network
        self.label = _initializer.init_routes_df["labels"];
        self.arcs = self.rmp_initializer_model.arcs
        self.arcs_drp_org = [a for a in self.arcs if 'O' not in a]
        self.arcs_index = self.rmp_initializer_model.arcs_index
        self.nodes = self.rmp_initializer_model.nodes
        
        self.is_root_node = True
        self.ch_dom = _chDom
        self.dom_rule = _dom_rule
        self.node_count = 0
        
        # [ip_obj, route_pats, model, init_df, node_count]
        self.best_node = [1e10, self.route_pats, self.rmp_model, self.rmp_init_df, self.node_count ]
        
        
        # acc flag
        self.acc_flag = _acc_flag
        
    
    def sense(self):
        return pybnb.minimize
    
    def objective(self):
        if (self.loc_bound < 1e10) and (self.loc_bound > -1e10):
            lp_obj = sum([p.Obj*self.lp_sol[p.varName] for p in self.rmp_model.getVars()] )
            temp_m_ip_bound = self.rmp_initializer_model
            temp_m_ip_bound.model = self.rmp_model.copy()
            temp_m_ip_bound.shortCuttingColumns()
            temp_m_ip_bound.model.update()
            temp_m_ip_bound.solveModel()
#             ip_obj = sum([p.Obj for p in temp_m_ip_bound.relaxedBoundedModel.getVars() if p.X>0] )
            ip_obj = temp_m_ip_bound.model.ObjVal
            print("LP/IP:",lp_obj,ip_obj)
            p_vars = temp_m_ip_bound.model.getVars()
            _route_pats_shortcut = dict( [ (p.varName ,  
                                       dict( [(self.label[a_idx], 
                                               temp_m_ip_bound.init_routes_df[p.varName][a_idx])
                                                for a_idx in self.arcs_index 
                                                    if temp_m_ip_bound.init_routes_df[p.varName][a_idx]>0] )
                                                      ) for p in p_vars ] )
            print("IP SOLUTIONS:",[[p.varName, p.x, _route_pats_shortcut[p.varName]] for p in p_vars if p.x >0])
            # update best node:
            if ip_obj<self.best_node[0]:
                print("Best Node found!")
                self.best_node = [ip_obj, _route_pats_shortcut,temp_m_ip_bound.model, temp_m_ip_bound.init_routes_df, self.node_count ]
            return ip_obj
        else:
            return 1e10
        
    def bound(self):
        # update bound from column generation
        self.loc_bound, self.route_pats, self.lp_sol, self.rmp_model, self.rmp_init_df, self.node_count = SolveMinAverageTimeSpentNode(self)
        if (self.loc_bound-np.floor(self.loc_bound))>1e-6:
            new_lb = np.round(self.loc_bound,6)
        else: new_lb = np.floor(self.loc_bound)
        print("==Called bound(), Update LOCAL BOUND:", new_lb) 
        return new_lb

    # 1st-called
    def save_state(self, node): 
        # node structure: 
        #. |---- route_pats:[r1,r2,..], r1 = {<cus_idx>: # of visits}
        #. |____ b_cond_log:[((c1,c2),0),((c1,c3),1),..], (c1,c2) = <arc begin branched on>
        node.state = (self.route_pats, self.b_cond_log, self.del_pats, self.rmp_model, self.rmp_init_df)

    def load_state(self, node):
        (self.route_pats, self.b_cond_log, self.del_pats, self.rmp_model, self.rmp_init_df) = node.state
    
        
    def branch(self):
        _n = self.initializer.no_customer #no. node
        _nodes = self.nodes
        # From Node
        _route_pats = self.route_pats
        _b_cond_log = self.b_cond_log
            # _del_pats = self.del_pats #don't need will create new here
        _cur_rmp_model = self.rmp_initializer_model
        _cur_rmp_model.model = self.rmp_model.copy() # Import node model
        
        # for braching
        _temp_model = self.rmp_initializer_model
        _temp_model.model = self.rmp_model.copy()
#         _temp_model.shortCutColumnsBnp()
        _temp_model.solveRelaxedBoundedModel()
        _m_constrs = _temp_model.model.getConstrs()
        p_vars = _temp_model.relaxedBoundedModel.getVars() #Load current node's model
        _objVal = _temp_model.relaxedBoundedModel.ObjVal

#         print(_route_pats)
#         print("Temp model Sol:",[[p.varName, p.x,_route_pats[p.varName]] for p in p_vars if p.x >0])
        
        # get the pair vars
        # find the first fractional variable and branch
        frac = None
        val = None
        gap = None
        print("==Called branch, branching with cond:",_b_cond_log)
        _not_bch_arcs =[]
        for a in _b_cond_log:
            _not_bch_arcs+=[a[0]]
            if a[0][0]!='O' and a[0][1]!='O': 
                if a[1]==1: _not_bch_arcs+=[(a[0][1],a[0][0])]
#                 _not_bch_arcs+=[(a[0][0],j) for j in _nodes if j!=a[0][0]]
            
#         print("Len frac_sol:",len(frac_sol),"Len route_pats:", len(route_pats))
        # Infeasibility condition trickered when penalized route being selected
        if (_cur_rmp_model.relaxedBoundedModel.status != 2):
            print("Fathomed by infeasibility")
            return []
        else:
            _lp_sol = dict([(p.varName,(p.x)) for p in p_vars])
            print("===Len lp_sol:",len(_lp_sol),"Len route_pats:", len(_route_pats)) 
#             print(_lp_sol, _route_pats)
            route_dict = dict(); arc_score_dict = dict();
            frac_dict = dict(); route_incident_dict = dict();
            r_w_cycle = []
#             for arc in self.arcs_drp_org:
            for arc in self.arcs:
                val = 0
#                 if arc not in _not_bch_arcs:
                for r_name, r_dict in _route_pats.items():     
                    if (arc in r_dict.keys()) and (_lp_sol[r_name]>0) and (arc not in _not_bch_arcs):
#                             print(r_dict[arc],_lp_sol[r_name])
                        if (r_dict[arc]>0):
#                             print(r_name,arc,_lp_sol[r_name])
                            val += _lp_sol[r_name]*r_dict[arc]
                            # route contains cycle visiting arc > 1
                            if (r_dict[arc]>1) and (r_name not in r_w_cycle): 
                                r_w_cycle.append(r_name)
#                         arc_score_dict[arc] = val
                        arc_score_dict[arc] = np.abs(0.5-val)
                if (abs(val - np.floor(val)) > epsilon) and (val!=1):
                    frac_dict[arc] = np.abs(0.5-val)
#                     frac_dict[arc] = val
#                 if val!=1:
#                     gap = abs(val - np.floor(val))
#                     if "O" in arc: frac_dict[arc] = val
#                     else: frac_dict[arc] = val

            bch_arcs_dict = dict(); bch_arcs=[]
#             print("r_w_cycle",r_w_cycle)
#             print("frac_dict",frac_dict)
            if (len(r_w_cycle) > 0):
                # Build node incident dict of route
                for r_name in r_w_cycle:
                    incident_dict = dict(zip([x for x in _nodes],[0]*(_n+1)))
                    for arc in _route_pats[r_name].keys():
                        i = arc[0]; j = arc[1];
                        incident_dict[i]+=1*_route_pats[r_name][arc]
                        incident_dict[j]+=1*_route_pats[r_name][arc]
#                         print(arc, incident_dict)
                    for arc in _route_pats[r_name].keys():
                        if (arc not in bch_arcs) and (arc not in _not_bch_arcs):
                            if (i=="O") and (incident_dict[j]>1): bch_arcs += [(arc)]
                            elif (j=="O") and (incident_dict[i]>1): bch_arcs += [(arc)]
                            elif (incident_dict[i]>1 or incident_dict[j]>1): bch_arcs += [(arc)]
#                     print("r_name",r_name, "Incident-dict",incident_dict)
#                 for r_name in r_w_cycle:
#                     bch_arcs += _route_pats[r_name].keys()
#                 print("bch_arcs",bch_arcs)
                for a in bch_arcs: 
                    if ((a in self.arcs) and (a not in _not_bch_arcs)):
                        # Avoid branch on arc with O to help speed up
                        if (("O" in a) and (arc_score_dict[a]!=1)): bch_arcs_dict[a] = arc_score_dict[a]
                        else: bch_arcs_dict[a] = arc_score_dict[a] 
#                 bch_arcs_dict = dict([(a,arc_score_dict[a] ) for a in bch_arcs if ((a in self.arcs) and (a not in _not_bch_arcs))])
                print("bch_arcs_dict",bch_arcs_dict)
                if len(bch_arcs_dict.keys())>0:
                    frac = max(bch_arcs_dict, key = bch_arcs_dict.get)
#                     frac = random.choice(list(bch_arcs_dict.keys()))
                else:
                    frac = max(frac_dict, key = frac_dict.get)
#                     frac = random.choice(list(frac_dict.keys()))
            elif (len(frac_dict.keys())>0):
                frac = max(frac_dict, key = frac_dict.get)
#                 frac = random.choice(list(frac_dict.keys()))
#             print("_route_pats",_route_pats)
            
#             print("arc_score_dict",arc_score_dict)
            
            print("frac",frac)
            if frac!=None:
                print("===branching on arc/val", frac)
                # from parent
                r_b_cond_log = deepcopy(_b_cond_log)
                r_b_cond_log.append([frac,1])
                r_route_pats = deepcopy(_route_pats)
                if "O" not in frac: r_b_cond_log.append([(frac[1],frac[0]),0])

                l_b_cond_log = deepcopy(_b_cond_log)
                l_b_cond_log.append([frac,0])
                l_route_pats = deepcopy(_route_pats)
    #             print(route_pats.keys())
                # yikes look at that iterable
                r_del = []
                l_del = []
                for indx in list(reversed(_route_pats.keys())):
#                     i,j = frac[0], frac[1]
                    route_pat = _route_pats[indx]
#                     if len(route_pat.keys())==2: continue
                    if frac in route_pat.keys(): 
                        # pat contains arc, delete it from (0) l-branch 
#                         print("0-LEFT-BCHING:",indx, route_pat)
                        l_del.append(indx)
#                         l_route_pats.pop(indx) #remove from 0 branch
                    else: 
                        # pat does not contains arc, delete it from (1) r-branch 
#                         print("1-RIGHT-BCHING:",indx, route_pat)
                        route_pat_nodes = [a[1] for a in list(route_pat.keys())]
                        
                        if (frac[0] == "O"): # and (frac[1] in route_pat_nodes)):
                            if (frac[1] in route_pat_nodes):
                            # Not having (0,j) but have some (i,j) i!=0
                                r_del.append(indx) # penalize from 1 branch
                        elif (frac[1] == "O"):# and (frac[0] in route_pat_nodes)):
                            if (frac[0] in route_pat_nodes):  
                            # Not having (i,0) but have some (i,j) j!=0
                                r_del.append(indx) # penalize from 1 branch
                        elif ( ((frac[0] in route_pat_nodes)) or ((frac[1] in route_pat_nodes)) ):
                            # Not having (i,j) but have some (i,k) k!=j or (k,j) k!=i
                            r_del.append(indx) # penalize from 1 branch
                l_child = pybnb.Node()
                l_child.state = (l_route_pats, l_b_cond_log, l_del,_cur_rmp_model.model.copy(),deepcopy(self.rmp_init_df))
                r_child = pybnb.Node()
                r_child.state = (r_route_pats, r_b_cond_log, r_del,_cur_rmp_model.model.copy(),deepcopy(self.rmp_init_df))
#                 print("Len l-branch vars:", len(l_route_pats),"Len r-branch vars:", len(r_route_pats))
                return l_child,r_child
            else: 
                return []

        
        
def SolveMinAverageTimeSpentNode(cTCCVRP_mt):
    # Instance 
    _n = cTCCVRP_mt.initializer.no_customer #no. node
    _dist_mat = cTCCVRP_mt.inst_dist_mat
    _initializer = cTCCVRP_mt.initializer
    _const_dict = cTCCVRP_mt.constant_dict
    _node_count = cTCCVRP_mt.node_count
    _chDom = cTCCVRP_mt.ch_dom
    _domRule = cTCCVRP_mt.dom_rule
    # From Node
    _route_pats = cTCCVRP_mt.route_pats
    _b_cond_log = cTCCVRP_mt.b_cond_log
    _del_pats = cTCCVRP_mt.del_pats
    _minAverageTimeLP_node = cTCCVRP_mt.rmp_initializer_model
    _minAverageTimeLP_node.init_routes_df = deepcopy(cTCCVRP_mt.rmp_init_df)
    _minAverageTimeLP_node.model = cTCCVRP_mt.rmp_model.copy() # Start with parent's node model
    
    print("==Branching Condition:", _b_cond_log)

    # apply branching to parent's column pool: Penalize del_pats!
    # INCIDENT DICT
    necess_link = [bh[0] for bh in _b_cond_log if (bh[1]==1)] # 1-branch
    incident_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    
    _bch_conflict = False
    _inflow_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    _outflow_dict = dict(zip([x for x in range(_n+1)],[0]*(_n+1)))
    for arc in necess_link:
        i = int(arc[0].split("_")[-1].replace("O","0")); j = int(arc[1].split("_")[-1].replace("O","0"))
#         incident_dict[i]+=1; incident_dict[j]+=1
        if i!=0 and j!=0: _inflow_dict[j]+=1; _outflow_dict[i]+=1;
        if (_outflow_dict[i]>1 and i!=0) or (_inflow_dict[j]>1 and j!=0): 
            _bch_conflict = True; break;
    print("necess_link",necess_link)
    print("_inflow_dict",_inflow_dict)
    print("_outflow_dict",_outflow_dict)
    
    # PENALIZING
#     m_lab_idx = cTCCVRP_mt.label.loc[cTCCVRP_mt.label=='m'].index[0]
    for v in _minAverageTimeLP_node.model.getVars():
        if v.varName in _del_pats: 
#             _minAverageTimeLP_node.init_routes_df.loc[m_lab_idx,v.varName] = 1e10
            v.Obj = 1e10
#             _route_pats.pop(v.varName)
#     print("==Solving node, with braching conds list:",_b_cond_log)
#     print("==Len Vars:",len(_minAverageTimeLP_node.model.getVars()))
#     print("==Del Pats:",_del_pats)
    _minAverageTimeLP_node.model.update()
#     print("DFAFTERDROP:",_minAverageTimeLP_node.init_routes_df.columns)
#     print("MODELVARS:",_minAverageTimeLP_node.model.getVars())
    
    # check if model is infeasible
    # if so, return inf
    _minAverageTimeLP_node.solveRelaxedBoundedModel()
    mrelax_obj = _minAverageTimeLP_node.relaxedBoundedModel.ObjVal
    print("==Model's status:",_minAverageTimeLP_node.relaxedBoundedModel.status)
    if (_minAverageTimeLP_node.relaxedBoundedModel.status != 2) or (mrelax_obj >= 1e9) or (_bch_conflict):
        print("==Prunned by INFEASIBILITY, mrx_obj, bch_conflict:",mrelax_obj,_bch_conflict )
        return 1e10, None, None, None, None,_node_count
    
    
    # COLUMN GENERATION!: build pricing model
    # add branching constraint to pricing sp model
    
    ########Pricing###########
    t1 = time.time()
    _minAverageTimeLP_node.runColumnsGeneration(None,_pricing_status=False,
            _check_dominance=_chDom,_dominance_rule=_domRule ,_DP_ver="SIMUL_M",
            _time_limit=_const_dict['dp_time_limit'],_filtering_mode="BestRwdPerI",
            _bch_cond = _b_cond_log,_node_count_lab = str(_node_count),
            _acc_flag =cTCCVRP_mt.acc_flag)
    colGen_te = time.time()-t1
    # update _route_pats & objval
    _minAverageTimeLP_node.solveRelaxedBoundedModel()
    mrelax_obj = _minAverageTimeLP_node.relaxedBoundedModel.ObjVal
    p_vars = _minAverageTimeLP_node.relaxedBoundedModel.getVars()
    if cTCCVRP_mt.is_root_node:
        print("==THIS IS ROOT NODE!:", cTCCVRP_mt.is_root_node); cTCCVRP_mt.is_root_node = False;
        cTCCVRP_mt.root_node = [mrelax_obj, _minAverageTimeLP_node.model, _minAverageTimeLP_node.init_routes_df, colGen_te,_minAverageTimeLP_node.colgenLogs]
        cTCCVRP_mt.is_root_node = False
    else: 
        print("==THIS IS NODE:",_node_count)
    
    
    print("LP-ColGen OBJ:",_minAverageTimeLP_node.relaxedBoundedModel.ObjVal)
    print("ModelVars, DataFrameVars :", len(p_vars),len(_minAverageTimeLP_node.init_routes_df.columns)-1)
    
    _route_pats = dict( [ (p.varName ,   dict( [(cTCCVRP_mt.label[a_idx], _minAverageTimeLP_node.init_routes_df[p.varName][a_idx])
                                                    for a_idx in cTCCVRP_mt.arcs_index 
                                                        if _minAverageTimeLP_node.init_routes_df[p.varName][a_idx]>0] )
                              ) for p in p_vars ] )
#         mrelax_obj = _minAverageTimeLP_node.relaxedBoundedModel.ObjVal
#         scc_cols = _minAverageTimeLP_node.init_routes_df.loc[_minAverageTimeLP_node.customer_index,:].drop(columns = "labels")
#         scc_cols['new_index'] = range(1,_n+1)
#         _route_pats = scc_cols.set_index('new_index').to_dict()
#         p_vars = _minAverageTimeLP_node.relaxedBoundedModel.getVars()
    print("==Obj-val colgen:", mrelax_obj)
#     print("==Del Pats:",_del_pats)
    print("==Branching Condition:", _b_cond_log)
    print([[p.varName, p.x, _route_pats[p.varName]] for p in p_vars if p.x >0])
    lp_sol = dict([(p.varName,(p.x)) for p in p_vars])
    _node_count+=1
    return mrelax_obj, _route_pats, lp_sol, _minAverageTimeLP_node.model.copy(), deepcopy(_minAverageTimeLP_node.init_routes_df) ,_node_count



