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
import bnp as bnp


## EXPERIMENT 1 & EXPERIMENT 2 (by specifying fixed dem-rates)
# Minimum Fleet size model's Experiment
def runExperimentMinFleetSizeTWFromInstList(_inst_list, _const_dict, _edge_config,
                                            _inst_dir, _sol_dir, _result_dir, _exp_name,
                                            _dom_rule=3,_chDom=True,
                                            _nSize=7,_show_full_info=False,
                                            _dp_mode=None,
                                            _log_file_name=None,
                                            _bnp_time_limit=None, _bnp_sol_suff="",
                                           _queue_strategy='bound', _fixed_homo_dem=None,
                                           _root_node_only = False):
    _expLog = dict()
    _solLog = dict()
    # Instance info
    _inst_info_topics = ['n','total_dem']
    # Initial solution set info
    _init_nstops = _const_dict['init_max_nodes_proute']
    _init_sol_topics = ['init_npr','init_mpr','init_cols','%s-stopIP'%_init_nstops,'%s-stopIP_te'%_init_nstops]
    # Pricing DP parameters
    _dp_params = ['npr','mpr','tt_m' ]
    # Root node colgen & IPHeuristics solution set info
    _root_node_sol_topics = ['rtLPcolGenRelax','rtLPcolGenRelax_te','rtIPcolGen','rtIPcolGen_te','rtColAdded','rtIP_remSpace']
    # BnP solution set info
    _bnp_sol_topics = ['termCond', 'upb', 'lwb', 'gap', 'nodesExp', 'wallTime','bnpTimeLim','bnp_remSpace']
    
    _col = _inst_info_topics+_init_sol_topics+_dp_params+_root_node_sol_topics+_bnp_sol_topics
    
    for inst in _inst_list:
        if _bnp_time_limit is None:
            _suff1 = 'tl-inf'; _bnp_time_limit=np.inf
        else: _suff1='tl-%s'%(_bnp_time_limit)
            
        _suff1+="-tw{}-".format(_const_dict['time_window'])
        
    # Import Instance Data
        instSol = dict(zip(['%s-stopIP'%_init_nstops,'LPcolGenRelax','IPcolGen','bnpSol'],[None]))
        instLog = dict(zip(_col,[0]*len(_col)))
        _distMat,_nodeTce,_cusDem,_nodePos = rand_inst.import_instance(_inst_dir,inst)
        if _fixed_homo_dem is not None:
            _suff1 += "fixedD-{}".format(_fixed_homo_dem)
            _cusDem.loc[_cusDem.index.str.contains("c")] = _fixed_homo_dem
        print("_cusDem",_cusDem.to_list())    
        _nodeTce['marker']['size'] = _nSize
        instLog['n'] = len(_nodePos)-1
        instLog['total_dem'] = _cusDem.sum()
        instLog['init_npr'] = _const_dict['init_max_nodes_proute']
        instLog['init_mpr'] = _const_dict['init_max_vehicles_proute']
        
        labeling_dict = vis_sol.create_nodes(0,instLog['n'])
        docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()
    # Generate initial feasible set R+ for root node
        _initializer = init_path.InitialRouteGenerator(1,labeling_dict,
                                              _cusDem,_const_dict,
                                              _distMat)
        _row_labels = ['lr','m']+depot+customers+arcs
        _init_route = _initializer.generateInitDFV4wTimeWindow(_row_labels,_const_dict)
        instLog['init_cols'] = len(_init_route)
    # IP with R+   
        init_timeW_IP_model = md.timeWindowModel(_init_route, _initializer,
                                                 _distMat,_const_dict, _relax_route=False)
        init_timeW_IP_model.buildModel()
        init_timeW_IP_model.model.setParam('OutputFlag',False)
        t1 = time.time()
        init_timeW_IP_model.solveModel()
        instLog['%s-stopIP'%_init_nstops] = init_timeW_IP_model.model.ObjVal
        instLog['%s-stopIP_te'%_init_nstops] = time.time()-t1
        instSol['%s-stopIP'%_init_nstops] = init_timeW_IP_model.getRouteSolution(init_timeW_IP_model.model.getVars(),
                                             _edge_config,_nodeTce,
                                              init_timeW_IP_model.customer_demand)
        
    # DP Params
        _const_dict['max_vehicles_proute_DP'] = round(init_timeW_IP_model.model.ObjVal) # Use feasible sol as UB
        instLog['npr'] = _const_dict['max_nodes_proute_DP'] #
        instLog['mpr'] = _const_dict['max_vehicles_proute_DP']
        instLog['tt_m'] = _const_dict['max_vehicles']
       
    # Root Node ColGen and IPHeu
        # Put node limit to 1. Solve root node only
        if (_root_node_only): _node_limit = 1
        else: _node_limit = None
    
        # Call Branch and Price solver & solve!
        _problem = bnp.MinimumFleetSizeWithTimeWindowBnP(_distMat, _initializer, 
                         _init_route, _const_dict, _chDom = True)
        _results = pybnb.solver.solve(_problem, log_filename = _log_file_name, 
                                      comm=None, absolute_gap=1e-6,
                                      time_limit=_bnp_time_limit,
                                     queue_strategy=_queue_strategy, node_limit=_node_limit)
        
    # Retriev root node solution
        _m_rt = _problem.rmp_initializer_model 
        [_rt_relax_obj,_m_rt.model,_m_rt.init_routes_df,_rt_colGen_te, _rt_colGen_log ] = _problem.root_node
        _m_rt.shortCuttingColumns()
        _m_rt.model.update()
        
        instLog['rtLPcolGenRelax'] = _rt_relax_obj
        instLog['rtLPcolGenRelax_te'] = _rt_colGen_te
        instSol['rtLPcolGenRelax'] = _rt_colGen_log
        instLog['rtColAdded'] = np.array([x['cols_add'] for x in instSol['rtLPcolGenRelax'].values()]).sum()
        
        t1=time.time(); _m_rt.solveModel()
        instLog['rtIPcolGen'] = _m_rt.model.ObjVal
        instLog['rtIPcolGen_te'] = time.time()-t1
        instSol['rtIPcolGen'] = _m_rt.getRouteSolution(_m_rt.model.getVars(),
                                                     _edge_config,_nodeTce,
                                                      _m_rt.customer_demand)
        
        instLog['rtIP_remSpace'] = _m_rt.calculateAverageRemainingSpace(_m_rt.model.getVars())
        
    # Retriev the best node solution
        _obj,_route_pats,_rmp_model,_rmp_init_df,_node_count = _problem.best_node
        instLog['termCond'] = _results.termination_condition
        instLog['upb'] = _obj
        instLog['lwb'] = _results.bound
        instLog['gap'] = _results.absolute_gap
        instLog['nodesExp'] = _results.nodes
        instLog['wallTime'] = _results.wall_time
        instLog['bnpTimeLim'] = _bnp_time_limit
        
        _m_ip_bnp = _problem.rmp_initializer_model
        _m_ip_bnp.init_routes_df = _rmp_init_df
        _m_ip_bnp.model = _rmp_model.copy()
        _m_ip_bnp.shortCuttingColumns()
        _m_ip_bnp.model.update()
        _m_ip_bnp.solveModel()
        
        twBnP_sol = _m_ip_bnp.getRouteSolution(_m_ip_bnp.model.getVars(),
                                             _edge_config,_nodeTce,
                                              _initializer.customer_demand)
        instSol['bnpSol'] = twBnP_sol
        instLog['bnp_remSpace'] = _m_ip_bnp.calculateAverageRemainingSpace(_m_ip_bnp.model.getVars())
        _expLog[inst] = instLog
        _solLog[inst] = instSol
        
    # Init p stops solution plot 
        
        _init_ip_plt_title = 'SOL_Init{0}stops_{1}_{2}'.format(_init_nstops, 
                                                inst,str(round(instLog['%s-stopIP'%_init_nstops],2)) )
        vis_sol.plot_network(instSol['%s-stopIP'%_init_nstops],_nodeTce,
                         _display_cus_dem=True,_cus_dem=_cusDem,_title=_init_ip_plt_title,
                         _save_to_file=_sol_dir+'SOL_Init{0}stops_{1}.png'.format(_init_nstops,inst),
                         _display_plot=False,_display_info_table=True,_show_all_info=False)
        
    # Root node Heuristics solution plot    
        _rt_iph_plt_title = 'SOL_rtColGenIPHeu_{0}_{1}{2}'.format(
                                                inst,str(round(instLog['rtIPcolGen'],2)),"_"+_suff1)
        vis_sol.plot_network(instSol['rtIPcolGen'],_nodeTce,_display_cus_dem=True,
                         _cus_dem=_cusDem,_title=_rt_iph_plt_title,
                         _save_to_file=_sol_dir+'SOL_rtColGenIPHeu_{0}{1}.png'.format(inst,"_"+_suff1),
                         _display_plot=False,_display_info_table=True,_show_all_info=False)
        
    # Bnp best node solution plot    
        _bnp_plt_title = 'SOL_IPBnp_{0}_{1}_{2}'.format(inst,str(round(instLog['upb'],2)),_suff1+_bnp_sol_suff)
        vis_sol.plot_network(instSol['bnpSol'],_nodeTce,_display_cus_dem=True,
                _cus_dem=_cusDem,_title=_bnp_plt_title,
                _save_to_file=_sol_dir+'SOL_IPBnp_{0}{1}.png'.format(inst,"_"+_suff1+_bnp_sol_suff),
                _display_plot=True,_display_info_table=True,_show_all_info=False)
        
    # Save result
        _resultDF = pd.DataFrame(_expLog).transpose()
        _resultDF = _resultDF.reindex(columns=_col)
        _resultDF.to_csv(_result_dir+_exp_name+'.csv')
        
    # Final iteration
    time_stp = datetime.now().strftime("%H%M-%b_%d_%y")
    _resultDF.to_csv(_result_dir+_exp_name+'_%s.csv'%time_stp)
    return _expLog,_solLog,_col




## EXPERIMENT 2
# Minimum Fleet size model's Experiment with varied demand rates













## EXPERIMENT 3
# Minimum Fleet size model's Experiment with varied time window
def runSensitivityLeadTimeMinFleetSizeTWFromInstList(_inst_list, _const_dict, _edge_config,
                                            _inst_dir, _sol_dir, _result_dir, _exp_name,_lead_time_list,
                                            _dom_rule=3,_chDom=True,
                                            _nSize=7,_show_full_info=False,
                                            _dp_mode=None,
                                            _log_file_name=None,
                                            _bnp_time_limit=None, _bnp_sol_suff="",
                                           _queue_strategy='bound', _fixed_homo_dem=None,
                                           _root_node_only = True):
    
    _expLog = dict()
    _solLog = dict()
    # Instance info
    _inst_info_topics = ['n','total_dem']
    # Initial solution set info
    _init_nstops = _const_dict['init_max_nodes_proute']
    _init_sol_topics = ['init_npr','init_mpr','init_cols','%s-stopIP'%_init_nstops]
    # Pricing DP parameters
    _dp_params = ['npr','mpr','tt_m' ]
    # Root node colgen & IPHeuristics solution set info
    _root_node_sol_topics = list(np.array([['rtIPcolGen-lt{}'.format(lt),'rtIPcolGen_te-lt{}'.format(lt),'colsNo-{}'.format(lt)] for lt in _lead_time_list]).flat)
    # BnP solution set info
    _bnp_sol_topics = ['wallTime','bnpTimeLim']
    
    _col = _inst_info_topics+_init_sol_topics+_dp_params+_root_node_sol_topics+_bnp_sol_topics
    
    for inst in _inst_list:
        if _bnp_time_limit is None:
            _suff1 = 'tl-inf'; _bnp_time_limit=np.inf
        else: _suff1='tl-%s'%(_bnp_time_limit)
        
        # Start with smallest lead time (most restricted)
        _lead_time = _lead_time_list[0]
        _const_dict['time_window'] = _lead_time
        _main_suff = _suff1
        _suff1= _main_suff+"-tw{}-".format(_const_dict['time_window'])
        
        # Import Instance Data
        instSol = dict()
        instLog = dict(zip(_col,[0]*len(_col)))
        
        _distMat,_nodeTce,_cusDem,_nodePos = rand_inst.import_instance(_inst_dir,inst)
        if _fixed_homo_dem is not None:
            _suff1 += "fixedD-{}".format(_fixed_homo_dem)
            _cusDem.loc[_cusDem.index.str.contains("c")] = _fixed_homo_dem
        print("_cusDem",_cusDem.to_list())    
        _nodeTce['marker']['size'] = _nSize
        instLog['n'] = len(_nodePos)-1
        instLog['total_dem'] = _cusDem.sum()
        instLog['init_npr'] = _const_dict['init_max_nodes_proute']
        instLog['init_mpr'] = _const_dict['init_max_vehicles_proute']
        
        labeling_dict = vis_sol.create_nodes(0,instLog['n'])
        docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()
        
        # Generate initial feasible set R+ for root node
        _initializer = init_path.InitialRouteGenerator(1,labeling_dict,
                                              _cusDem,_const_dict,
                                              _distMat)
        _row_labels = ['lr','m']+depot+customers+arcs
        _init_route = _initializer.generateInitDFV4wTimeWindow(_row_labels,_const_dict)
        instLog['init_cols'] = len(_init_route)
        
        print("_lead_time:{}".format(_lead_time) ,"init_col:{}".format(len(_init_route)))
        
        # IP with R+   
        init_timeW_IP_model = md.timeWindowModel(_init_route, _initializer,
                                                 _distMat,_const_dict, _relax_route=False)
        init_timeW_IP_model.buildModel()
        init_timeW_IP_model.model.setParam('OutputFlag',False)
        t1 = time.time()
        init_timeW_IP_model.solveModel()
#         instLog['%s-stopIP'%_init_nstops] = init_timeW_IP_model.model.ObjVal
#         instLog['%s-stopIP_te'%_init_nstops] = time.time()-t1
#         instSol['%s-stopIP'%_init_nstops] = init_timeW_IP_model.getRouteSolution(init_timeW_IP_model.model.getVars(),
#                                              _edge_config,_nodeTce,
#                                               init_timeW_IP_model.customer_demand)
        
        # DP Params
        try:
            instLog['%s-stopIP'%_init_nstops] = init_timeW_IP_model.model.ObjVal
            _const_dict['max_vehicles_proute_DP'] = round(init_timeW_IP_model.model.ObjVal) # Use feasible sol as UB
            instLog['npr'] = _const_dict['max_nodes_proute_DP'] #
            instLog['mpr'] = _const_dict['max_vehicles_proute_DP']
            instLog['tt_m'] = _const_dict['max_vehicles']
        except:
            instLog['%s-stopIP'%_init_nstops] = 'infeas'
            instSol['%s-stopIP'%_init_nstops] = 'infeas'
            _expLog[inst] = instLog
            _solLog[inst] = instSol

            _resultDF = pd.DataFrame(_expLog).transpose()
            _resultDF = _resultDF.reindex(columns=_col)
            _resultDF.to_csv(_result_dir+_exp_name+'.csv')
            continue
        
       
        
        
        # Root Node ColGen and IPHeu
        # Put node limit to 1. Solve root node only
        if (_root_node_only): _node_limit = 1
        else: _node_limit = None
        
        # Call Branch and Price solver & solve!
        _problem = bnp.MinimumFleetSizeWithTimeWindowBnP(_distMat, _initializer, 
                         _init_route, _const_dict, _chDom = True)
        t1=time.time()
        _results = pybnb.solver.solve(_problem, log_filename = _log_file_name, 
                                      comm=None, absolute_gap=1e-6,
                                      time_limit=_bnp_time_limit,
                                     queue_strategy=_queue_strategy, node_limit=_node_limit)
        # Retriev root node solution
        _m_rt = _problem.rmp_initializer_model 
        [_rt_relax_obj,_m_rt.model,_m_rt.init_routes_df,_rt_colGen_te, _rt_colGen_log ] = _problem.root_node
        _m_rt.shortCuttingColumns()
        _m_rt.model.update()
        print("col after colgen:", _m_rt.init_routes_df.shape[1]-1)
        
        _m_rt.solveModel()
        instLog['colsNo-{}'.format(_lead_time)] = _m_rt.init_routes_df.shape[1]-1
        instLog['rtIPcolGen-lt{}'.format(_lead_time)] = _m_rt.model.ObjVal
        instLog['rtIPcolGen_te-lt{}'.format(_lead_time)] = time.time()-t1
        instSol['rtIPcolGen-lt{}'.format(_lead_time)] = _m_rt.getRouteSolution(_m_rt.model.getVars(),
                                                     _edge_config,_nodeTce,
                                                      _m_rt.customer_demand) 
        instLog['wallTime'] += instLog['rtIPcolGen_te-lt{}'.format(_lead_time)]
        # Root node Heuristics solution plot    
        _rt_iph_plt_title = 'SOL_rtColGenIPHeu_{0}_{1}{2}'.format(
                                                inst,str(round(instLog['rtIPcolGen-lt{}'.format(_lead_time)],2)),"_"+_suff1)
        vis_sol.plot_network(instSol['rtIPcolGen-lt{}'.format(_lead_time)],_nodeTce,_display_cus_dem=True,
                         _cus_dem=_cusDem,_title=_rt_iph_plt_title,
                         _save_to_file=_sol_dir+'SOL_rtColGenIPHeu_{0}{1}.png'.format(inst,"_"+_suff1),
                         _display_plot=False,_display_info_table=True,_show_all_info=False)
        
        
        # Iterate over lead time list
        for _lead_time in _lead_time_list[1:]:
            _const_dict['time_window'] = _lead_time
            _suff1= _main_suff+"-tw{}-".format(_const_dict['time_window'])
            
            # Use prev col pool as init route
            # Rename col as new
            _initializer.init_routes_df = deepcopy(_m_rt.init_routes_df)
            _col_names = dict([(_initializer.init_routes_df.columns[c],'route[%d]'%(c-1) ) for c in range(len(_initializer.init_routes_df.columns))])
            _initializer.init_routes_df.rename(columns = _col_names,inplace = True)
            _initializer.init_routes_df.rename(columns ={'route[-1]':'labels'},inplace = True)
#             
            print(_initializer.init_routes_df.columns)
            
            _init_route = _initializer.generateBasicInitialPatterns(_initializer.init_routes_df.shape[1]-1,initRouteDf=_initializer.init_routes_df.set_index('labels'))
            _init_route.rename(index=lambda x:'route[%d]'%x ,inplace=True)
            
            print("_lead_time:{}".format(_lead_time) ,"init_col:{}".format(len(_init_route)))
            _problem = bnp.MinimumFleetSizeWithTimeWindowBnP(_distMat, _initializer, 
                         _init_route, _const_dict, _chDom = True)
            t1=time.time()
            _results = pybnb.solver.solve(_problem, log_filename = _log_file_name, 
                                          comm=None, absolute_gap=1e-6,
                                          time_limit=_bnp_time_limit,
                                         queue_strategy=_queue_strategy, node_limit=_node_limit)
            # Retriev root node solution
            _m_rt = _problem.rmp_initializer_model 
            [_rt_relax_obj,_m_rt.model,_m_rt.init_routes_df,_rt_colGen_te, _rt_colGen_log ] = _problem.root_node
            _m_rt.shortCuttingColumns()
            _m_rt.model.update()
            print("col after colgen:", _m_rt.init_routes_df.shape[1]-1)

            _m_rt.solveModel()
            instLog['colsNo-{}'.format(_lead_time)] = _m_rt.init_routes_df.shape[1]-1
            instLog['rtIPcolGen-lt{}'.format(_lead_time)] = _m_rt.model.ObjVal
            instLog['rtIPcolGen_te-lt{}'.format(_lead_time)] = time.time()-t1
            instSol['rtIPcolGen-lt{}'.format(_lead_time)] = _m_rt.getRouteSolution(_m_rt.model.getVars(),
                                                         _edge_config,_nodeTce,
                                                          _m_rt.customer_demand)
            instLog['wallTime'] += instLog['rtIPcolGen_te-lt{}'.format(_lead_time)]

            # Root node Heuristics solution plot    
            _rt_iph_plt_title = 'SOL_rtColGenIPHeu_{0}_{1}{2}'.format(
                                                    inst,str(round(instLog['rtIPcolGen-lt{}'.format(_lead_time)],2)),"_"+_suff1)
            vis_sol.plot_network(instSol['rtIPcolGen-lt{}'.format(_lead_time)],_nodeTce,_display_cus_dem=True,
                             _cus_dem=_cusDem,_title=_rt_iph_plt_title,
                             _save_to_file=_sol_dir+'SOL_rtColGenIPHeu_{0}{1}.png'.format(inst,"_"+_suff1),
                             _display_plot=False,_display_info_table=True,_show_all_info=False)

        # Save each instance result
        _expLog[inst] = instLog
        _solLog[inst] = instSol
        
        _resultDF = pd.DataFrame(_expLog).transpose()
        _resultDF = _resultDF.reindex(columns=_col)
        _resultDF.to_csv(_result_dir+_exp_name+'.csv')

    # Save total wall time
    instLog['bnpTimeLim'] = _bnp_time_limit 
    # Final iteration
    time_stp = datetime.now().strftime("%H%M-%b_%d_%y")
    _resultDF.to_csv(_result_dir+_exp_name+'_%s.csv'%time_stp)












## EXPERIMENT 4
# Minimum Average time spent model's Experiment
def runExperimentMinAveTimeSpentFromInstList(_inst_list, _const_dict, _edge_config,
                                            _inst_dir, _sol_dir, _result_dir, _exp_name,
                                            _dom_rule=4,_chDom=True,
                                            _nSize=7,_show_full_info=False,
                                            _dp_mode=None,
                                            _log_file_name=None,
                                            _bnp_time_limit=None, _bnp_sol_suff="",
                                            _acc_flag=None,
                                           _queue_strategy='bound'):
    
    _expLog = dict()
    _solLog = dict()
    if _bnp_time_limit is None:
        _suff1 = 'tl-inf'; _bnp_time_limit=np.inf
    else: _suff1='tl-%s'%(_bnp_time_limit)
    # Instance info
    _inst_info_topics = ['n','total_dem']
    # Initial solution set info
    _init_pstops = _const_dict['init_max_nodes_proute'] 
    _init_mvehs = _const_dict['init_max_vehicles_proute'] # update again with Phase 1 solution
    _init_sol_topics = ['init_npr','init_mpr','init_cols','initIP','initIP_te']
    # Pricing DP parameters
    _dp_params = ['npr','mpr','tt_m' ]
    # Root node colgen & IPHeuristics solution set info
    _root_node_sol_topics = ['rtLPcolGenRelax','rtLPcolGenRelax_te','rtIPcolGen','rtIPcolGen_te','rtColAdded','rtIP_remSpace']
    # BnP solution set info
    _bnp_sol_topics = ['termCond', 'upb', 'lwb', 'gap', 'nodesExp', 'wallTime','bnpTimeLim','bnp_remSpace']
    
    _col = _inst_info_topics+_init_sol_topics+_dp_params+_root_node_sol_topics+_bnp_sol_topics
    
    for inst in _inst_list:
    # Import Instance Data
        instSol = dict(zip(['initIP','LPcolGenRelax','IPcolGen','bnpSol'],[None]))
        instLog = dict(zip(_col,[0]*len(_col)))
        _distMat,_nodeTce,_cusDem,_nodePos = rand_inst.import_instance(_inst_dir,inst)
        _nodeTce['marker']['size'] = _nSize
        instLog['n'] = len(_nodePos)-1
        instLog['total_dem'] = _cusDem.sum()
        instLog['init_npr'] = _const_dict['init_max_nodes_proute']
        instLog['init_mpr'] = _const_dict['init_max_vehicles_proute']
        
        labeling_dict = vis_sol.create_nodes(0,instLog['n'])
        docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()
    # Generate initial feasible set R+ for root node
        _initializer = init_path.InitialRouteGenerator(1,labeling_dict,
                                              _cusDem,_const_dict,
                                              _distMat)
        _row_labels = ['lr','m']+depot+customers+arcs
        _init_route = _initializer.generateInitDFV2(_row_labels,_const_dict)
        instLog['init_cols'] = len(_init_route)
        
    # Phase 1 with R+
        phaseI_min_veh_model = md.phaseIModel(_init_route, _initializer,
                 _distMat,_const_dict)
        phaseI_min_veh_model.model.setParam('OutputFlag',False)
        phaseI_min_veh_model.buildModel()
        phaseI_min_veh_model.solveModel()
        _min_veh = phaseI_min_veh_model.model.ObjVal;print('Min M:',_min_veh)
        # UPDATE PHASE 1 SOL AS MAX VEHICLE
        _const_dict['max_vehicles'] = _min_veh
        
    # IP with R+   
        init_phaseII_IP_model = md.phaseIIModel(_init_route, _initializer,
                                                 _distMat,_const_dict, _relax_route=False)
        init_phaseII_IP_model.buildModel()
        init_phaseII_IP_model.model.setParam('OutputFlag',False)
        t1 = time.time()
        init_phaseII_IP_model.solveModel()
        instLog['initIP'] = init_phaseII_IP_model.model.ObjVal*60/_cusDem.sum()
        instLog['initIP_te'] = time.time()-t1
        instSol['initIP'] = init_phaseII_IP_model.getRouteSolution(init_phaseII_IP_model.model.getVars(),
                                             _edge_config,_nodeTce,
                                              init_phaseII_IP_model.customer_demand)
        
    # DP Params       
        if (_const_dict['max_vehicles_proute_DP'] is None) or (_const_dict['max_vehicles_proute_DP']>_const_dict['max_vehicles']):
            _const_dict['max_vehicles_proute_DP'] = _const_dict['max_vehicles'] # Use Phase1 sol as UB
        instLog['npr'] = int(_const_dict['max_nodes_proute_DP']) #
        instLog['mpr'] = int(_const_dict['max_vehicles_proute_DP'])
        instLog['tt_m'] = _const_dict['max_vehicles']
    
    # Root Node ColGen and IPHeu 
        # Call Branch and Price solver & solve!
        _problem = bnp.MinimumAverageTimeSpentBnP(_distMat, _initializer, 
                         _init_route, _const_dict, _chDom = True, _acc_flag = _acc_flag, _dom_rule = _dom_rule)
        _results = pybnb.solver.solve(_problem, log_filename = _log_file_name, 
                                      comm=None, absolute_gap=1e-6,
                                      time_limit=_bnp_time_limit,
                                     queue_strategy=_queue_strategy)
        
    # Retriev root node solution
        _m_rt = _problem.rmp_initializer_model 
        [_rt_relax_obj,_m_rt.model,_m_rt.init_routes_df,_rt_colGen_te, _rt_colGen_log ] = _problem.root_node
        _m_rt.shortCuttingColumns()
        _m_rt.model.update()
        
        instLog['rtLPcolGenRelax'] = _rt_relax_obj*60/_cusDem.sum()
        instLog['rtLPcolGenRelax_te'] = _rt_colGen_te
        instSol['rtLPcolGenRelax'] = _rt_colGen_log
        instLog['rtColAdded'] = np.array([x['cols_add'] for x in instSol['rtLPcolGenRelax'].values()]).sum()
        
        t1=time.time(); _m_rt.solveModel()
        instLog['rtIPcolGen'] = _m_rt.model.ObjVal*60/_cusDem.sum()
        instLog['rtIPcolGen_te'] = time.time()-t1
        instSol['rtIPcolGen'] = _m_rt.getRouteSolution(_m_rt.model.getVars(),
                                                     _edge_config,_nodeTce,
                                                      _m_rt.customer_demand)
        
        instLog['rtIP_remSpace'] = _m_rt.calculateAverageRemainingSpace(_m_rt.model.getVars())
        
    # Retriev the best node solution
        _obj,_route_pats,_rmp_model,_rmp_init_df,_node_count = _problem.best_node
        instLog['termCond'] = _results.termination_condition
        instLog['upb'] = _obj*60/_cusDem.sum()
        instLog['lwb'] = _results.bound*60/_cusDem.sum()
        if _results.absolute_gap is None : _results.absolute_gap = 0
        instLog['gap'] = round(_results.absolute_gap*60/_cusDem.sum(),3)
        instLog['nodesExp'] = _results.nodes
        instLog['wallTime'] = _results.wall_time
        instLog['bnpTimeLim'] = _bnp_time_limit
        
        _m_ip_bnp = _problem.rmp_initializer_model
        _m_ip_bnp.init_routes_df = _rmp_init_df
        _m_ip_bnp.model = _rmp_model.copy()
        _m_ip_bnp.shortCuttingColumns()
        _m_ip_bnp.model.update()
        _m_ip_bnp.solveModel()
        # Min time bnp solution
        mtBnP_sol = _m_ip_bnp.getRouteSolution(_m_ip_bnp.model.getVars(),
                                             _edge_config,_nodeTce,
                                              _initializer.customer_demand)
        instSol['bnpSol'] = mtBnP_sol
        instLog['bnp_remSpace'] = _m_ip_bnp.calculateAverageRemainingSpace(_m_ip_bnp.model.getVars())
        _expLog[inst] = instLog
        _solLog[inst] = instSol
        
        
    # Init p stops solution plot 
        _init_ip_plt_title = 'SOL_Init{}stops{}vehs_{}_{}'.format(_init_pstops,_init_mvehs,
                                                            inst,
                                                            str(round(instLog['initIP'],2)) )
        vis_sol.plot_network(instSol['initIP'],_nodeTce,
                         _display_cus_dem=True,_cus_dem=_cusDem,_title=_init_ip_plt_title,
                         _save_to_file=_sol_dir+'SOL_Init{}stops{}vehs_{}.png'.format(_init_pstops,
                                                                                      _init_mvehs,
                                                                                      inst),
                         _display_plot=False,_display_info_table=True,_show_all_info=False)
        
    # Root node Heuristics solution plot    
        _rt_iph_plt_title = 'SOL_rtColGenIPHeu{}stops{}vehs_{}_{}{}'.format(instLog['npr'], instLog['mpr'], inst,
                                                                str(round(instLog['rtIPcolGen'],2)),
                                                                "_"+_suff1)
        vis_sol.plot_network(instSol['rtIPcolGen'],_nodeTce,_display_cus_dem=True,
                         _cus_dem=_cusDem,_title=_rt_iph_plt_title,
                         _save_to_file=_sol_dir+'SOL_rtColGenIPHeu{}stops{}vehs_{}{}.png'.format(instLog['npr'], instLog['mpr'],
                                                                                                 inst,"_"+_suff1),
                         _display_plot=False,_display_info_table=True,_show_all_info=False)
        
    # Bnp best node solution plot    
        _bnp_plt_title = 'SOL_IPBnp{}stops{}vehs_{}_{}_{}'.format(instLog['npr'], instLog['mpr'], 
                                                                  inst,str(round(instLog['upb'],2)),
                                                     _suff1+_bnp_sol_suff)
        vis_sol.plot_network(instSol['bnpSol'],_nodeTce,_display_cus_dem=True,
                _cus_dem=_cusDem,_title=_bnp_plt_title,
                _save_to_file=_sol_dir+'SOL_IPBnp{}stops{}vehs_{}{}.png'.format(instLog['npr'], instLog['mpr'],
                                                                                inst,"_"+_suff1+_bnp_sol_suff),
                _display_plot=False,_display_info_table=True,_show_all_info=False)
        
    # Save result
        _resultDF = pd.DataFrame(_expLog).transpose()
        _resultDF = _resultDF.reindex(columns=_col)
        _resultDF.to_csv(_result_dir+_exp_name+'.csv')
        
    # Final iteration
    time_stp = datetime.now().strftime("%H%M-%b_%d_%y")
    _resultDF.to_csv(_result_dir+_exp_name+'_%s.csv'%time_stp)
    
    
    
    
## EXPERIMENT 4.1 - Check LP bound of colgen root node only
# Minimum Average time spent model's Experiment
def runExperimentMinAveTimeSpentLPColgenFromInstList(_inst_list, _const_dict, _edge_config,
                                            _inst_dir, _sol_dir, _result_dir, _exp_name,
                                            _dom_rule=4,_chDom=True,
                                            _nSize=7,_show_full_info=False,
                                            _dp_mode=None,
                                            _log_file_name=None,
                                            _bnp_time_limit=None, _bnp_sol_suff="",
                                            _acc_flag=None,
                                           _queue_strategy='bound'):
    
    _expLog = dict()
    _solLog = dict()
    if _bnp_time_limit is None:
        _suff1 = 'tl-inf'; _bnp_time_limit=np.inf
    else: _suff1='tl-%s'%(_bnp_time_limit)
    # Instance info
    _inst_info_topics = ['n','total_dem']
    # Initial solution set info
    _init_pstops = _const_dict['init_max_nodes_proute'] 
    _init_mvehs = _const_dict['init_max_vehicles_proute'] # update again with Phase 1 solution
    _init_sol_topics = ['init_npr','init_mpr','init_cols','initIP','initIP_te']
    # Pricing DP parameters
    _dp_params = ['npr','mpr','tt_m' ]
    # Root node colgen & IPHeuristics solution set info
    _root_node_sol_topics = ['rtLPcolGenRelax','rtLPcolGenRelax_te','rtIPcolGen','rtIPcolGen_te','rtColAdded','rtIP_remSpace']
    # BnP solution set info
    _bnp_sol_topics = ['termCond', 'upb', 'lwb', 'gap', 'nodesExp', 'wallTime','bnpTimeLim','bnp_remSpace']
    
    _col = _inst_info_topics+_init_sol_topics+_dp_params+_root_node_sol_topics+_bnp_sol_topics
    
    for inst in _inst_list:
    # Import Instance Data
        instSol = dict(zip(['initIP','LPcolGenRelax','IPcolGen','bnpSol'],[None]))
        instLog = dict(zip(_col,[0]*len(_col)))
        _distMat,_nodeTce,_cusDem,_nodePos = rand_inst.import_instance(_inst_dir,inst)
        _nodeTce['marker']['size'] = _nSize
        instLog['n'] = len(_nodePos)-1
        instLog['total_dem'] = _cusDem.sum()
        instLog['init_npr'] = _const_dict['init_max_nodes_proute']
        instLog['init_mpr'] = _const_dict['init_max_vehicles_proute']
        
        labeling_dict = vis_sol.create_nodes(0,instLog['n'])
        docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs = labeling_dict.values()
    # Generate initial feasible set R+ for root node
        _initializer = init_path.InitialRouteGenerator(1,labeling_dict,
                                              _cusDem,_const_dict,
                                              _distMat)
        _row_labels = ['lr','m']+depot+customers+arcs
        _init_route = _initializer.generateInitDFV2(_row_labels,_const_dict)
        instLog['init_cols'] = len(_init_route)
        
    # Phase 1 with R+
        phaseI_min_veh_model = md.phaseIModel(_init_route, _initializer,
                 _distMat,_const_dict)
        phaseI_min_veh_model.model.setParam('OutputFlag',False)
        phaseI_min_veh_model.buildModel()
        phaseI_min_veh_model.solveModel()
        _min_veh = phaseI_min_veh_model.model.ObjVal;print('Min M:',_min_veh)
        # UPDATE PHASE 1 SOL AS MAX VEHICLE
        _const_dict['max_vehicles'] = _min_veh
        
    # IP with R+   
        init_phaseII_IP_model = md.phaseIIModel(_init_route, _initializer,
                                                 _distMat,_const_dict, _relax_route=False)
        init_phaseII_IP_model.buildModel()
        init_phaseII_IP_model.model.setParam('OutputFlag',False)
        t1 = time.time()
        init_phaseII_IP_model.solveModel()
        instLog['initIP'] = init_phaseII_IP_model.model.ObjVal*60/_cusDem.sum()
        instLog['initIP_te'] = time.time()-t1
        instSol['initIP'] = init_phaseII_IP_model.getRouteSolution(init_phaseII_IP_model.model.getVars(),
                                             _edge_config,_nodeTce,
                                              init_phaseII_IP_model.customer_demand)
        
    # DP Params       
        if (_const_dict['max_vehicles_proute_DP'] is None) or (_const_dict['max_vehicles_proute_DP']>_const_dict['max_vehicles']):
            _const_dict['max_vehicles_proute_DP'] = _const_dict['max_vehicles'] # Use Phase1 sol as UB
        instLog['npr'] = int(_const_dict['max_nodes_proute_DP']) #
        instLog['mpr'] = int(_const_dict['max_vehicles_proute_DP'])
        instLog['tt_m'] = _const_dict['max_vehicles']
    
    # Root Node ColGen LP
        init_phaseII_IP_model.model.update()
        t1 = time.time()
        init_phaseII_IP_model.runColumnsGeneration(None,_pricing_status=False,
                _check_dominance=True,_dominance_rule=_dom_rule ,_DP_ver="SIMUL_M",
                _time_limit=_const_dict['dp_time_limit'],_filtering_mode="BestRwdPerI",
                _bch_cond = None,_node_count_lab = None,
                _acc_flag =_acc_flag)
        colGen_te = time.time()-t1
        # update _route_pats & objval
        init_phaseII_IP_model.solveRelaxedBoundedModel()
        mrelax_obj = init_phaseII_IP_model.relaxedBoundedModel.ObjVal
    
    # Root Node ColGen and IPHeu 
#         # Call Branch and Price solver & solve!
#         _problem = bnp.MinimumAverageTimeSpentBnP(_distMat, _initializer, 
#                          _init_route, _const_dict, _chDom = True, _acc_flag = _acc_flag, _dom_rule = _dom_rule)
#         _results = pybnb.solver.solve(_problem, log_filename = _log_file_name, 
#                                       comm=None, absolute_gap=1e-6,
#                                       time_limit=_bnp_time_limit,
#                                      queue_strategy=_queue_strategy)
        
    # Retriev root node solution
        _m_rt = init_phaseII_IP_model
        _rt_relax_obj = mrelax_obj
        _rt_colGen_te = colGen_te
        _rt_colGen_log = _m_rt.colgenLogs
#         [_rt_relax_obj,_m_rt.model,_m_rt.init_routes_df,_rt_colGen_te, _rt_colGen_log ] = _problem.root_node
    
        _m_rt.shortCuttingColumns()
        _m_rt.model.update()
        
        instLog['rtLPcolGenRelax'] = _rt_relax_obj*60/_cusDem.sum()
        instLog['rtLPcolGenRelax_te'] = _rt_colGen_te
        instSol['rtLPcolGenRelax'] = _rt_colGen_log
        instLog['rtColAdded'] = np.array([x['cols_add'] for x in instSol['rtLPcolGenRelax'].values()]).sum()
        
        t1=time.time(); _m_rt.solveModel()
        instLog['rtIPcolGen'] = _m_rt.model.ObjVal*60/_cusDem.sum()
        instLog['rtIPcolGen_te'] = time.time()-t1
        instSol['rtIPcolGen'] = _m_rt.getRouteSolution(_m_rt.model.getVars(),
                                                     _edge_config,_nodeTce,
                                                      _m_rt.customer_demand)
        
        instLog['rtIP_remSpace'] = _m_rt.calculateAverageRemainingSpace(_m_rt.model.getVars())
        
    # Retriev the best node solution
        _obj,_route_pats,_rmp_model,_rmp_init_df,_node_count = [1e10, [], [], _initializer.init_routes_df, 1 ]
#         _problem.best_node
        instLog['termCond'] = "root only" #_results.termination_condition
        instLog['upb'] = _obj*60/_cusDem.sum()
        instLog['lwb'] = _rt_relax_obj*60/_cusDem.sum()
#         if _results.absolute_gap is None : _results.absolute_gap = 0
        instLog['gap'] = None #round(_results.absolute_gap*60/_cusDem.sum(),3)
        instLog['nodesExp'] = 1#_results.nodes
        instLog['wallTime'] =  _rt_colGen_te#_results.wall_time
        instLog['bnpTimeLim'] = None#_bnp_time_limit
        
#         _m_ip_bnp = _problem.rmp_initializer_model
#         _m_ip_bnp.init_routes_df = _rmp_init_df
#         _m_ip_bnp.model = _rmp_model.copy()
#         _m_ip_bnp.shortCuttingColumns()
#         _m_ip_bnp.model.update()
#         _m_ip_bnp.solveModel()
#         # Min time bnp solution
#         mtBnP_sol = _m_ip_bnp.getRouteSolution(_m_ip_bnp.model.getVars(),
#                                              _edge_config,_nodeTce,
#                                               _initializer.customer_demand)
#         instSol['bnpSol'] = mtBnP_sol
        instLog['bnp_remSpace'] = 0#_m_ip_bnp.calculateAverageRemainingSpace(_m_ip_bnp.model.getVars())
        _expLog[inst] = instLog
        _solLog[inst] = instSol
  
        
    # Save result
        _resultDF = pd.DataFrame(_expLog).transpose()
        _resultDF = _resultDF.reindex(columns=_col)
        _resultDF.to_csv(_result_dir+_exp_name+'.csv')
        
    # Final iteration
    time_stp = datetime.now().strftime("%H%M-%b_%d_%y")
    _resultDF.to_csv(_result_dir+_exp_name+'_%s.csv'%time_stp)



