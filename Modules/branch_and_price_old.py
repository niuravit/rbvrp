import numpy as np
import random 
import pandas as pd
import pickle as pk
import sys
sys.path.insert(0,'/Users/admin/Desktop/EE_Year4_Term2/OR/VRP+d/Combine_variables/Modules')
from gurobipy import *
from datetime import datetime
EXP_PATH = '/Users/admin/Desktop/EE_Year4_Term2/OR/VRP+d/Combine_variables/Experiment'
import os
os.environ['GRB_LICENSE_FILE'] = '/Users/admin/Gurobi_lic_key/gurobi.lic'
import time
import visualize_sol as vis_sol
import initialize_path as init_path
import random_instance as rand_inst
import utility as util
import branch_and_price as bnp
import model as md
from itertools import combinations,permutations 
epsilon = 0.000001

class Experiment:
    def __init__(self,_gen_mode,_no_truck,\
                 _no_customer,_no_dock,\
                 _constant_dict,_vis_plot=False,_Ld=None):
        self.gen_mode = _gen_mode
        self.no_truck = _no_truck
        self.no_customer = _no_customer
        self.no_dock = _no_dock
        self.vis_plot = _vis_plot
        self.Ld = _Ld
        self.constant_dict = _constant_dict
        self.x = None

    def InitializeNodeArc(self,):
        self.initializer = init_path.InitialRouteGenerator(self.no_truck,self.no_dock,self.no_customer)
        ## GENERATE NODES SET
        self.docking,self.customers,\
        self.depot,self.depot_s,\
        self.depot_t,self.all_depot,\
        self.nodes,self.arcs=self.initializer.generateNodeSet()
        ## GENERATE NODE LAYERS
        self.truck_cus_nodes_dict,self.truck_cus_nodes,\
        self.drone_cus_nodes,self.all_cus_nodes_types=self.initializer.generateTruckAndDroneNodes(self.customers)
        ## GENERATE ARCs
        self.arcs_truck_dict, self.arcs_truck, self.arcs_drone = self.initializer.generateTruckAndDroneArcs(self.truck_cus_nodes_dict,self.truck_cus_nodes,self.drone_cus_nodes)
    
    def ImportInstance(self,inst_type,inst_id,_map_plot=True):
        self.instance_name='InstanceType{0}Cus{1}Dock{2}_{3}'.format(inst_type,self.no_customer,self.no_dock,inst_id)
        with open(EXP_PATH+'/Instance%sForTesting/%s.pickle'%(inst_type,self.instance_name),'rb') as f1:
            sp_dismat = pk.load(f1)
        self.truck_distance= sp_dismat['truck_distance']
        self.drone_distance= sp_dismat['drone_distance']
        self.nodes_position= sp_dismat['nodes_position']
        self.node_trace= sp_dismat['nodes_trace']
        self.truck_distance.update(dict([(k[:-1]+'3',self.truck_distance[k]) for k in self.truck_distance.keys() if k.split(',')[-1]=='T1']))
        self.customer_demand_df=sp_dismat['customer_demand_df']
        if self.no_truck ==3:
            truck3 = self.customer_demand_df[self.customer_demand_df.index.str.contains('T1')].copy()
            truck3.index = [i[:-1]+'3' for i in truck3.index]
            self.customer_demand_df = pd.concat((self.customer_demand_df[:'customer_1_D'][:-1],truck3,self.customer_demand_df['customer_1_D':])) 
        elif self.no_truck ==1: self.customer_demand_df = self.customer_demand_df[~self.customer_demand_df.index.str.contains('T2')]
        self.customer_demand = self.customer_demand_df.values.tolist()
        if _map_plot:
            vis_sol.plot_solution([['depot,customer_2,T1'],['depot,customer_1,T2']],\
                              self.node_trace,_cus_dem=self.customer_demand_df,\
                               _title=self.instance_name )
    
    def runExperiment(self,_fix_cost_consideration,_sol_plot=True,_minimun_truck_count=False,_solve_trigger=True,_iter_limit=100):
        print('\nFixCost Consideration: ', _fix_cost_consideration)
        print('\n InitColGenMode:',self.gen_mode)
        if self.x is None:
            self.x = pd.DataFrame(data =self.all_depot+self.truck_cus_nodes+self.drone_cus_nodes+self.docking+self.arcs_truck+self.arcs_drone,columns=['labels'])
        if self.gen_mode in ['kmean','all','custom']:
            if self.gen_mode is not 'kmean':clustered=None
            else: clustered = init_path.initSetWithKMean(self.nodes_position,self.customer_demand_df,self.no_dock,self.constant_dict['max_capacity_truck']).generateInitialSet()
            self.initializer.generateTruckOnlyPath(self.truck_cus_nodes_dict,self.all_depot\
                            ,group_no=self.constant_dict['truck_count'] ,truck_cap_limit=self.constant_dict['max_capacity_truck']\
                            ,initRouteDf=self.x, clustered=clustered,\
                            customer_demand_df=self.customer_demand_df,mode=self.gen_mode)
        elif self.gen_mode in ['all_d']:
            self.integrated_tour_queue = init_path.generateIntegratedTourQueue(self.customer_demand_df,self.constant_dict,self.drone_distance)

            self.initializer.generateIntegratedTour(self.all_depot,self.customer_demand_df,self.constant_dict,self.x,\
                                                    self.drone_distance,self.integrated_tour_queue)
            self.initializer.generateIndependentRoute(self.all_depot,self.constant_dict['truck_count'],\
                                                      self.customer_demand_df,self.constant_dict,self.x,self.drone_distance)
            
        feasibleCols = len(self.x.columns)-1
#         if _minimun_truck_count:
#             print('\nAuto adjust truck count to be %d\n'%len(self.integrated_tour_queue))
#             self.constant_dict['truck_count'] = len(self.integrated_tour_queue)
#         self.bnp_dict = {'path':self.path,'all_depot':self.all_depot,'docking':self.docking,\
#             'truck_cus_nodes':self.truck_cus_nodes,'truck_cus_nodes_dict':self.truck_cus_nodes_dict,\
#             'drone_cus_nodes':self.drone_cus_nodes,'arcs_truck':self.arcs_truck,\
#             'arcs_drone':self.arcs_drone,'no_truck':self.no_truck,\
#             'truck_distance':self.truck_distance,'drone_distance':self.drone_distance,\
#             'constant_dict':self.constant_dict,'arcs_truck_dict':self.arcs_truck_dict,'customer_demand_df':self.customer_demand_df}
        
        if feasibleCols>=100000:
            obj_cost=None
            timeStatistics_1={}
            no_added_cols=None
            solvedNodePools_1=None
            nodesNumber = 0
            total_truck_no=0
            integerSolPools_1= None
            originalRmp = None
        elif not _solve_trigger: return None
        else:
            self.x = self.x.set_index('labels')
            self.path = self.initializer.generateBasicInitialPatterns(feasibleCols,initRouteDf=self.x)
            self.path.rename(index=lambda x:'path[%d]'%x,inplace=True)
            if _minimun_truck_count:
                print('\nAuto adjust truck count to be %d\n'%len(self.integrated_tour_queue))
                self.constant_dict['truck_count'] = len(self.integrated_tour_queue)
            self.bnp_dict = {'path':self.path,'all_depot':self.all_depot,'docking':self.docking,\
                'truck_cus_nodes':self.truck_cus_nodes,'truck_cus_nodes_dict':self.truck_cus_nodes_dict,\
                'drone_cus_nodes':self.drone_cus_nodes,'arcs_truck':self.arcs_truck,\
                'arcs_drone':self.arcs_drone,'no_truck':self.no_truck,\
                'truck_distance':self.truck_distance,'drone_distance':self.drone_distance,\
                'constant_dict':self.constant_dict,'arcs_truck_dict':self.arcs_truck_dict,\
                'customer_demand_df':self.customer_demand_df,'iter_limit':_iter_limit}
            # SOLVE!!
            if _fix_cost_consideration: self.solvedNodePools_1,self.integerSolPools_1,self.timeStatistics_1,self.originalRmp = bnp.BranchAndPriceWithFixCost(self.bnp_dict)
            else: self.solvedNodePools_1,self.integerSolPools_1,self.timeStatistics_1,self.originalRmp = bnp.BranchAndPrice(self.bnp_dict)
            self.nodesNumber = len(self.solvedNodePools_1)
            self.solvedNodePools_1['originalRmp']=self.originalRmp
            self.solvedNodePools_1['gen_mode']=self.gen_mode
            self.lastest_keys = list(self.integerSolPools_1.keys())[-1]

            self.optimal_node = self.integerSolPools_1[self.lastest_keys]
            self.no_added_cols = self.optimal_node.RmpModel.Path[self.optimal_node.RmpModel.Path.index.str.contains('ROOTcol')].shape[0]
        #     sol_x = pd.Series(optimal_node.RmpModel.relaxedModel.getAttr('X'),index=optimal_node.RmpModel.Path.index)
        #     idx = sol_x[sol_x>0].index[0]
            self.obj_cost, self.total_truck_no, self.total_drone_no, self.truck_serve_cus, self.drone_serve_cus,self.solutions = vis_sol.visualizeSolutions(self.optimal_node.RmpModel,0,relaxation=True,ignore_fix_cost=_fix_cost_consideration,vis_plot=_sol_plot,node_trace=self.node_trace)
        self.result = [self.instance_name,self.no_customer,self.no_dock,\
                  self.obj_cost,self.timeStatistics_1, self.gen_mode,\
                  len(self.path), self.no_added_cols, self.nodesNumber,\
                  self.total_truck_no, self.total_drone_no,self.Ld,\
                  self.truck_serve_cus,self.drone_serve_cus,self.solutions]
        
    def logging(self,_tag=""):
        self.timestmp = datetime.now().strftime('%m%d-%H%M')
        self.log_name = self.instance_name+'_'+self.gen_mode+self.timestmp+_tag
#         return solvedNodePools_1,integerSolPools_1,timeStatistics_1,result


## Branch and Price Framework
def BranchAndPrice(_bnp_dict):
    # GLOBAL: allcombiJ, unsolvedNodePools, UpperBound, allbranchingdecision
    _path = _bnp_dict['path'] 
    all_depot = _bnp_dict['all_depot'] 
    docking = _bnp_dict['docking'] 
    truck_cus_nodes = _bnp_dict['truck_cus_nodes'] 
    truck_cus_nodes_dict = _bnp_dict['truck_cus_nodes_dict'] 
    drone_cus_nodes = _bnp_dict['drone_cus_nodes'] 
    arcs_truck = _bnp_dict['arcs_truck'] 
    arcs_drone = _bnp_dict['arcs_drone'] 
    arcs_truck_dict = _bnp_dict['arcs_truck_dict']
    no_truck = _bnp_dict['no_truck'] 
    truck_distance = _bnp_dict['truck_distance'] 
    drone_distance = _bnp_dict['drone_distance'] 
    constant_dict = _bnp_dict['constant_dict'] 
    customer_demand_df = _bnp_dict['customer_demand_df']
    iter_limit = _bnp_dict['iter_limit']
    customer_demand = customer_demand_df.values.tolist()
    # define master 
    t_str = time.time()
    RmpModel = md.VRPdMasterProblem(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
    RmpModel.buildModel()
    RmpModel.model.setParam('OutputFlag',0)
    t_fin_build_rootrmp = time.time()
    RmpModel.solveRelaxedModel()
    duals = RmpModel.getDuals()
    t_fin_solved_root = time.time()
    #=================================
    TemplateRmpModel = md.VRPdMasterProblem(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
    TemplateRmpModel.model = RmpModel.model.copy()  
    TemplateRmpModel.model.setParam('OutputFlag',0)

    TemplateRmpModel.relaxedModel = TemplateRmpModel.model.relax()
    TemplateRmpModel.Path = RmpModel.Path.copy()
    #=================================

    all_constrs = RmpModel.model.getConstrs()
    customer_coverage_constrs = [c for c in all_constrs if 'customer_coverage' in c.ConstrName]
    all_branching_decisions =[x for x in combinations(customer_coverage_constrs,2)]
    pre_chosen_constrs = dict([(con,None) for con in all_branching_decisions])

    UpperBound = {np.inf:0}
    # define pricing problem
    PricingModel = md.VRPdPricingProblem(all_depot,truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_truck_dict,arcs_drone,\
                    duals,customer_demand,no_truck,truck_distance,drone_distance,constant_dict)
    PricingModel.buildModel()
    PricingModel.model.setParam('OutputFlag',0)
    t_fin_build_pricing = time.time()

    # define root node
    root_node = bnp.BnPNode('0_ROOT',RmpModel, PricingModel, pre_chosen_constrs)
    # add to unsolved node pool
    unsolvedPools = [root_node]

    solvedNodePools = dict()
    integerSolPools = dict()

    iteration = 0
    print("Iteration limit(Tolerance):",iter_limit)
    while len(unsolvedPools)>0:
        iteration+=1
        currentNode = unsolvedPools[0]
        print('ITERATION:',iteration)
        print('Unsolved NODES',[b.Name for b in unsolvedPools])
        print('GOING TO SOLVE',currentNode.Name,'\n')
        print('Branching Conditions:',currentNode.PreProcessedConstrs.values())
    #     print('Variables:', currentNode.RmpModel.model.getVars())
        del unsolvedPools[0]
        currentNode.RmpModel = TemplateRmpModel
        bnp.addColsFromColsPool(currentNode.RmpModel,RmpModel)

        currentNode.RmpModel.model.setParam('OutputFlag',0)
        
#         currentNode.RmpModel.relaxedModel = RmpModel.relaxedModel.copy()
        currentNode.RmpModel.relaxedModel = currentNode.RmpModel.model.relax()
#         currentNode.RmpModel.Path = RmpModel.Path.copy()
    
        all_constrs = currentNode.RmpModel.model.getConstrs()
        customer_coverage_constrs = [c for c in all_constrs if 'customer_coverage' in c.ConstrName]
        all_braching_decisions =[x for x in combinations(customer_coverage_constrs,2)]
        print('VarBefore(Original):',len(currentNode.RmpModel.model.getVars() ))
        currentNode.RmpModel = bnp.branchRmp(currentNode.RmpModel, currentNode.PreProcessedConstrs)
        print('VarAfterBranch:',len(currentNode.RmpModel.model.getVars() ))
        
        ColGen = bnp.ColumnGeneration(currentNode.RmpModel ,currentNode.PricingModel,\
                                  _all_branching_decisions=all_branching_decisions,\
                                  _PreProcessedConstrs=currentNode.PreProcessedConstrs,\
                                   _originalRmpModel=RmpModel,_affliation=currentNode.Name)
        ColGen.generateColumns()
        currentNode.RmpModel = ColGen.RmpModel
        
        print('VarAfterCol:',len(currentNode.RmpModel.model.getVars() ))
        if ColGen.StatusCode==3: 
            currentNode.NodeStatus = 'FathomedByColGenInfeasible'
            solvedNodePools[currentNode.Name] = (currentNode,ColGen)
            continue

        # Calculate summation xj of node
        currentNode.SummationXj = ColGen.getSummationXj()
        #Prune by bounding
        lpRmpObj = ColGen.RmpModel.relaxedModel.getObjective().getValue()
        if lpRmpObj>np.min(list(UpperBound.keys())):
            print('Prune by Bounding, %d'%lpRmpObj,'UpperBound:%d'%np.min(list(UpperBound.keys())))
            currentNode.NodeStatus = 'FathomedByBounding'
            solvedNodePools[currentNode.Name] = (currentNode,ColGen)
            #update node status: FathomedByBounding
            continue

        #Prune by optimality
        fractional_sol = bnp.getFractionalSol(ColGen.RmpModel)
#         print('fractional_sol:',fractional_sol)
        if bnp.getFractionalSol(ColGen.RmpModel).shape[0]==0: 
            print('Prune by Optimality')
            #update the UpperBound
            ans = ColGen.RmpModel.getRelaxSolution()
            UpperBound[lpRmpObj] = ans
            print('New bound:',lpRmpObj)
            
            integerNode = bnp.BnPNode(currentNode.Name,[], PricingModel, currentNode.PreProcessedConstrs.copy())
            integerNode.RmpModel = md.VRPdMasterProblem(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
            integerNode.RmpModel.Path = currentNode.RmpModel.Path.copy()
            integerNode.NodeStatus = 'FathomedOptimality'
            integerNode.ObjVal = currentNode.RmpModel.relaxedModel.ObjVal
            integerNode.OptimalSol = currentNode.RmpModel.relaxedModel.getAttr('X')
            currentNode.RmpModel.model.update()
            integerNode.RmpModel.model = currentNode.RmpModel.model.copy()
            integerNode.RmpModel.relaxedModel = integerNode.RmpModel.model.relax()
            integerSolPools[integerNode.Name] = integerNode
            solvedNodePools[integerNode.Name] = (integerNode,ColGen)
            #update node status: FathomedByOptimality
            continue
    #         break
        pred_constrs=[]
        for k,v in currentNode.PreProcessedConstrs.items():
            if v is not None:pred_constrs+=[k[0].ConstrName,k[1].ConstrName]
        f_j = currentNode.SummationXj[(currentNode.SummationXj>epsilon)&(currentNode.SummationXj<1-epsilon*10)]
        idx_pd = pd.Series(f_j.index)
#         not_included_constrs =f_j[idx_pd[idx_pd.apply(lambda x: (x[0].ConstrName not in pred_constrs)and(x[1].ConstrName not in pred_constrs))]]
        not_included_constrs =f_j[idx_pd[idx_pd.apply(lambda x: (x[0].ConstrName not in pred_constrs)or(x[1].ConstrName not in pred_constrs))]]
#         print(f_j,not_included_constrs)
        if not_included_constrs.shape[0]!=0:
            chosen_j = not_included_constrs[[not_included_constrs.idxmax()]].index[0]
        else:
            chosen_j = f_j[[f_j.idxmax()]].index[0]
        print('Chosen J:',chosen_j)
        # Branching
        zero_node, one_node = bnp.branchOnNode(currentNode,chosen_j,iteration)
        unsolvedPools+=[zero_node, one_node]
        currentNode.NodeStatus = 'Solved'
        solvedNodePools[currentNode.Name] = (currentNode,ColGen)
        print('Adding new branches to pool!\n')
#         print('==============================\n')

        if iteration>iter_limit: 
            print('NUMBER OF ITERATIONS EXCEED TOLERANCE')
            break
    t_fin_bnp = time.time()
    timeStatistic = {'Total Operation time':t_fin_bnp-t_str,\
                    'Root RMP Build-up':t_fin_build_rootrmp-t_str,\
                    'Solving Root RMP':t_fin_solved_root-t_fin_build_rootrmp,\
                    'Pricing Build-up':t_fin_build_pricing-t_fin_solved_root,\
                    'Solving MIP through BnP':t_fin_bnp-t_fin_build_pricing}
    print('Congratulation FINISH Branch and Price!@jd;adjfaderadf')
    print('Total Operation time:',timeStatistic['Total Operation time'])
    print('Root RMP Build-up:',timeStatistic['Root RMP Build-up'])
    print('Solving Root RMP:',timeStatistic['Solving Root RMP'])
    print('Pricing Build-up:',timeStatistic['Pricing Build-up'])
    print('Solving MIP through BnP:',timeStatistic['Solving MIP through BnP'])
    return solvedNodePools,integerSolPools,timeStatistic,RmpModel



class ColumnGeneration:
    def __init__(self, _rmp_model, _pricing_model,_all_branching_decisions, _PreProcessedConstrs,_originalRmpModel=None, _affliation=None):
        self.RmpModel = _rmp_model
        self.PricingModel = _pricing_model
        self.removeConstrBranching()
#         duals_dummy=[]
#         self.PricingModel = md.VRPdPricingProblem(all_depot,truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_truck_dict,arcs_drone,\
#                 duals_dummy,customer_demand,no_truck,truck_distance,drone_distance,constant_dict)
#         self.PricingModel.model = _pricing_model.model.copy()
#         self.PricingModel.generateVariables()
#         self.PricingModel.generateConstrains()
        self.PricingModel.model.setParam('OutputFlag',0)
        self.PricingModel.model.update()
        self.linkModelVaribles()
        if _originalRmpModel is not None: self.originalRmpModel = _originalRmpModel

        self.ModelImprovement = True
        self.PreProcessedConstrs = _PreProcessedConstrs
        # Create extra constraints for pricing
        self.branchingConstr = self.generateBranchingConstrains()
        self.all_branching_decisions = _all_branching_decisions
        self.DualsDict = dict()
        self.IterCount = 0
        if _affliation is None: self.Afflication = 'nan'
        else: self.Afflication = _affliation
            
            
        
    def generateColumns(self):
        print('STRART PROCESS COLUMN GENERATION FOR RMP:',self.Afflication)
        print('...')
        while(self.ModelImprovement):
            t1 = time.time()
            self.IterCount+=1
            if  self.Afflication=='0_ROOT' and self.IterCount==1:
                self.RmpModel.solveRelaxedModel()
                self.StatusCode = self.RmpModel.relaxedModel.Status
                print('Column Generation 1st iteration for ROOT-NODE')
            else:
#                 print('Column Generation iteration: %d'%self.IterCount)
                # Solve relaxed Master
                self.RmpModel.solveRelaxedModel()
                self.StatusCode = self.RmpModel.relaxedModel.Status
        
            if self.StatusCode==3:
                print('RMP Infeasible!')
                break
                
            duals = self.RmpModel.getDuals()
            self.DualsDict[self.IterCount] = duals
#             print("Finished solving RMP (%s) ..., iteration:"%self.Afflication,self.IterCount)

            # Assume that pricing problem is already built
            
            self.PricingModel.generateObjective(duals)
            self.PricingModel.model.update()
#             print("Start solving pricing...")
            self.PricingModel.solveModel(120,0.1) #timelimit and epsilon(GAP)
            if self.PricingModel.model.Status == 3:
                print("PRINCING INFEASIBLE!!!\n\n")
                self.ModelImprovement = False
            else:#Check if new pattern improves the Master Solution
                self.ModelImprovement = (self.PricingModel.model.ObjVal)<-epsilon
#                 print('ModelImprovement:',self.ModelImprovement)
            if not (self.ModelImprovement): break
                
            #Add new generated pattern to master problem and iterate
            newPatternCost = 1
            newPath =  self.PricingModel.getNewPath()
            list_var =  self.PricingModel.model.getVars()
            v_name = [v.VarName for v in list_var]
            var_name_df = pd.DataFrame(data=newPath,index= v_name)
            interested_var = var_name_df[var_name_df.index.str.contains('theta')|var_name_df.index.str.contains('arcs')]
#             print('Interested Var:',interested_var[0])
#             print('Interested VarValues:',interested_var.values)
#             print('Index:',self.PricingModel.mergeDepotCusArcsVar( self.PricingModel.PathConstructSeries.values))
            new_path = pd.DataFrame(data = interested_var.values,index = self.PricingModel.mergeDepotCusArcsVar( self.PricingModel.PathConstructSeries.values))
#             new_path = pd.Series(data = interested_var[0].values,index = self.PricingModel.mergeDepotCusArcsVar( self.PricingModel.PathConstructSeries.values))
            
            newPathCost =  self.PricingModel.getNewPathCost(new_path.values).getValue()
#             newPathCost =  self.PricingModel.getNewPathCost(var_name_df).getValue()
            print('DUALS',duals,'NEWPATHCOST',newPathCost)
            newPathName = 'node_'+self.Afflication+'colGen_'+str(self.IterCount)
            
            self.RmpModel.addColumn(newPathCost, new_path[0].values, self.PricingModel.model.getVarByName,newPathName)
            self.originalRmpModel.addColumn(newPathCost, new_path[0].values, self.PricingModel.model.getVarByName,newPathName)
#             print('Adding Path: %s to RMP'%newPathName)
            print('SubP OBJ(reduce_cost):',self.PricingModel.model.ObjVal)
#             temp = pd.Series()
            _new_path = pd.Series(data = interested_var[0].values,index = self.PricingModel.mergeDepotCusArcsVar( self.PricingModel.PathConstructSeries.values))
            print('newPatternCuts:',_new_path[_new_path>epsilon])
#             print('=================================')
#         self.RmpModel.solveRelaxedModel()
#         self.StatusCode = self.RmpModel.relaxedModel.Status

        t2 = time.time()
        if self.StatusCode!=3:
            self.RmpModel.solveRelaxedModel()
#             self.RmpModel.update()
            print('FINISH ADDING COLLUMNS:', self.Afflication)
            print('Elasp Time:',t2-t1)
            print('Status code%s'%self.StatusCode)
#             print('currentRMP:',self.RmpModel.Path.index, self.RmpModel.relaxedModel.getVars())
#             print('originalRMP:',self.originalRmpModel.Path, self.originalRmpModel.relaxedModel.getVars() )
            a = pd.Series(self.RmpModel.relaxedModel.getAttr('X'), index = self.RmpModel.Path.index)
            print(a[a>0])
        else: print('Pruned by Infeasible!')

    def generateBranchingConstrains(self):
        branch_constraint_list=[]
        
#         l = [(b_nodes, val) for (b_nodes, val) in zero.PreProcessedConstrs.items() if val is not None]
#         for (b_nodes, val) in [l[-1]]:
        for (b_nodes, val) in self.PreProcessedConstrs.items():
            if val is None: continue
            elif val=='One':
                chosen_truck_index = [int(b.ConstrName.split('[')[-1][:-1]) for b in b_nodes]
                print('OneConstraints',chosen_truck_index)
                i0,i1 = chosen_truck_index
                branch_constraint = self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i0)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i0)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath))==self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i1)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i1)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath))
                self.PricingModel.model.addConstr(branch_constraint,name='branching_one'+str(chosen_truck_index))
        #         one_branches.append([b.ConstrName.split('[')[-1][0] for b in b_nodes])
            elif val=='Zero':
                chosen_truck_index = [int(b.ConstrName.split('[')[-1][:-1]) for b in b_nodes]
                print('ZeroConstraints',chosen_truck_index)
                branch_constraint = (quicksum(( self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath)))\
                                  for i in chosen_truck_index)<=1)
                self.PricingModel.model.addConstr(branch_constraint,name='branching_zero'+str(chosen_truck_index))
            branch_constraint_list.append(branch_constraint)
        self.PricingModel.model.update()
#         print(self.PricingModel.model.getConstrs())
        return branch_constraint_list
    
    def linkModelVaribles(self):
        get_all_vars_pd = pd.Series(self.PricingModel.model.getVars())
        self.PricingModel.NodesDelta = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'theta' in x.VarName)].values))
        
        self.PricingModel.TruckArcs = dict(zip(self.PricingModel.TruckArcsIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'truck_arcs' in x.VarName)].values))

        self.PricingModel.DroneArcs = dict(zip(self.PricingModel.DroneArcsIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'drone_arcs' in x.VarName)].values))

        self.PricingModel.DroneCumulativeDemand = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'z' in x.VarName)].values))

        self.PricingModel.DroneCumulativeDistance = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'v' in x.VarName)].values))

        self.PricingModel.TruckCumulativeDemand = dict()
        self.PricingModel.TruckPassedNode = dict()
        self.PricingModel.TruckCumulativeDistance = dict()
        self.PricingModel.DecisionIfThen = dict()
        for i in range(self.PricingModel.NoTruckPerPath):
            self.PricingModel.TruckCumulativeDemand[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)),get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'y'+str(i+1) in x.VarName)].values))

            self.PricingModel.TruckPassedNode[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'alpha'+str(i+1) in x.VarName)].values))

            self.PricingModel.TruckCumulativeDistance[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'g'+str(i+1) in x.VarName)].values))

            self.PricingModel.DecisionIfThen[i+1] = dict(zip(np.concatenate((self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'gamma'+str(i+1) in x.VarName)].values))

        self.PricingModel.NodesBeta = dict(zip(self.PricingModel.NodesIndex, get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'beta' in x.VarName)].values))    
    
    def removeConstrBranching(self):
        all_constrs = pd.Series(self.PricingModel.model.getConstrs())
        bch_constrs = all_constrs[all_constrs.apply(lambda x: 'branching' in x.ConstrName)]
        self.PricingModel.model.remove(bch_constrs.values.tolist())
        self.PricingModel.model.update()
    
    def getSummationXj(self):
        summation = list()
        for b_con in self.all_branching_decisions:
            jSet = bnp.generateJSet(self.RmpModel.relaxedModel,b_con)
#             print(b_con,np.sum([x.X for x in jSet['r1r2']]))
            summation.append(np.sum([x.X for x in jSet['r1r2']]))
        summationXj = pd.Series(summation,index=self.all_branching_decisions)
        return summationXj

    ################Consider FiXCOST###################
class ColumnGenerationWithFixCost:
    def __init__(self, _rmp_model, _pricing_model,_all_branching_decisions, _PreProcessedConstrs,_originalRmpModel=None, _affliation=None):
        self.RmpModel = _rmp_model
        self.PricingModel = _pricing_model
        self.removeConstrBranching()
#         duals_dummy=[]
#         self.PricingModel = md.VRPdPricingProblem(all_depot,truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_truck_dict,arcs_drone,\
#                 duals_dummy,customer_demand,no_truck,truck_distance,drone_distance,constant_dict)
#         self.PricingModel.model = _pricing_model.model.copy()
#         self.PricingModel.generateVariables()
#         self.PricingModel.generateConstrains()
        self.PricingModel.model.setParam('OutputFlag',0)
        self.PricingModel.model.update()
        self.linkModelVaribles()
        if _originalRmpModel is not None: self.originalRmpModel = _originalRmpModel

        self.ModelImprovement = True
        self.PreProcessedConstrs = _PreProcessedConstrs
        # Create extra constraints for pricing
        self.branchingConstr = self.generateBranchingConstrains()
        self.all_branching_decisions = _all_branching_decisions
        self.DualsDict = dict()
        self.IterCount = 0
        if _affliation is None: self.Afflication = 'nan'
        else: self.Afflication = _affliation
            
            
        
    def generateColumns(self):
        print('STRART PROCESS COLUMN GENERATION FOR RMP:',self.Afflication)
        print('...')
        while(self.ModelImprovement):
            t1 = time.time()
            self.IterCount+=1
            if  self.Afflication=='0_ROOT' and self.IterCount==1:
                self.RmpModel.solveRelaxedModel()
                self.StatusCode = self.RmpModel.relaxedModel.Status
                print('Column Generation 1st iteration for ROOT-NODE')
            else:
#                 print('Column Generation iteration: %d'%self.IterCount)
                # Solve relaxed Master
                self.RmpModel.solveRelaxedModel()
                self.StatusCode = self.RmpModel.relaxedModel.Status
        
            if self.StatusCode==3:
                print('RMP Infeasible!')
                break
                
            duals = self.RmpModel.getDuals()
            self.DualsDict[self.IterCount] = duals
#             print("Finished solving RMP (%s) ..., iteration:"%self.Afflication,self.IterCount)

            # Assume that pricing problem is already built
            
            self.PricingModel.generateObjective(duals)
            self.PricingModel.model.update()
#             print("Start solving pricing...")
            self.PricingModel.solveModel(120,0.1) #timelimit and epsilon(GAP)
            if self.PricingModel.model.Status == 3:
                print("PRINCING INFEASIBLE!!!\n\n")
                self.ModelImprovement = False
            else:#Check if new pattern improves the Master Solution
                self.ModelImprovement = (self.PricingModel.model.ObjVal)<-0.000001
#                 print('ModelImprovement:',self.ModelImprovement)
            if not (self.ModelImprovement): break
                
            #Add new generated pattern to master problem and iterate
            newPatternCost = 1
            newPath =  self.PricingModel.getNewPath()
            list_var =  self.PricingModel.model.getVars()
            v_name = [v.VarName for v in list_var]
            var_name_df = pd.DataFrame(data=newPath,index= v_name)
            interested_var = var_name_df[var_name_df.index.str.contains('theta')|var_name_df.index.str.contains('arcs')]
            new_path = pd.DataFrame(data = interested_var.values,index = self.PricingModel.mergeDepotCusArcsVar( self.PricingModel.PathConstructSeries.values))

#             newPathCost =  self.PricingModel.getNewPathCost(new_path.values).getValue()
            newPathCost =  self.PricingModel.getNewPathCost(var_name_df).getValue()
            print('DUALS',duals,'NEWPATHCOST',newPathCost)
            newPathName = 'node_'+self.Afflication+'colGen_'+str(self.IterCount)
            
            self.RmpModel.addColumn(newPathCost, new_path[0].values, self.PricingModel.model.getVarByName,newPathName)
            self.originalRmpModel.addColumn(newPathCost, new_path[0].values, self.PricingModel.model.getVarByName,newPathName)
#             print('Adding Path: %s to RMP'%newPathName)
#             print('SubP OBJ(reduce_cost):',self.PricingModel.model.ObjVal)
        #     print('newPatternCuts:',newPath)
#             print('=================================')
#         self.RmpModel.solveRelaxedModel()
#         self.StatusCode = self.RmpModel.relaxedModel.Status

        t2 = time.time()
        if self.StatusCode!=3:
            self.RmpModel.solveRelaxedModel()
#             self.RmpModel.update()
            print('FINISH ADDING COLLUMNS:', self.Afflication)
            print('Elasp Time:',t2-t1)
            print('Status code%s'%self.StatusCode)
#             print('currentRMP:',self.RmpModel.Path.index, self.RmpModel.relaxedModel.getVars())
#             print('originalRMP:',self.originalRmpModel.Path, self.originalRmpModel.relaxedModel.getVars() )
            a = pd.Series(self.RmpModel.relaxedModel.getAttr('X'), index = self.RmpModel.Path.index)
            print(a[a>0])
        else: print('Pruned by Infeasible!')

    def generateBranchingConstrains(self):
        branch_constraint_list=[]
        
#         l = [(b_nodes, val) for (b_nodes, val) in zero.PreProcessedConstrs.items() if val is not None]
#         for (b_nodes, val) in [l[-1]]:
        for (b_nodes, val) in self.PreProcessedConstrs.items():
            if val is None: continue
            elif val=='One':
                chosen_truck_index = [int(b.ConstrName.split('[')[-1][:-1]) for b in b_nodes]
                print('OneConstraints',chosen_truck_index)
                i0,i1 = chosen_truck_index
                branch_constraint = self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i0)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i0)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath))==self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i1)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i1)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath))
                self.PricingModel.model.addConstr(branch_constraint,name='branching_one'+str(chosen_truck_index))
        #         one_branches.append([b.ConstrName.split('[')[-1][0] for b in b_nodes])
            elif val=='Zero':
                chosen_truck_index = [int(b.ConstrName.split('[')[-1][:-1]) for b in b_nodes]
                print('ZeroConstraints',chosen_truck_index)
                branch_constraint = (quicksum(( self.PricingModel.NodesDelta[self.PricingModel.getBothTypeCustomerIndex(i)[1]] +\
                        quicksum(self.PricingModel.TruckPassedNode[p+1][self.PricingModel.getBothTypeCustomerIndex(i)[0][p]]\
                                 for p in range(self.PricingModel.NoTruckPerPath)))\
                                  for i in chosen_truck_index)<=1)
                self.PricingModel.model.addConstr(branch_constraint,name='branching_zero'+str(chosen_truck_index))
            branch_constraint_list.append(branch_constraint)
        self.PricingModel.model.update()
#         print(self.PricingModel.model.getConstrs())
        return branch_constraint_list
    
    def linkModelVaribles(self):
        get_all_vars_pd = pd.Series(self.PricingModel.model.getVars())
        self.PricingModel.NodesDelta = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'theta' in x.VarName)].values))
        
        self.PricingModel.TruckArcs = dict(zip(self.PricingModel.TruckArcsIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'truck_arcs' in x.VarName)].values))

        self.PricingModel.DroneArcs = dict(zip(self.PricingModel.DroneArcsIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'drone_arcs' in x.VarName)].values))

        self.PricingModel.DroneCumulativeDemand = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'z' in x.VarName)].values))

        self.PricingModel.DroneCumulativeDistance = dict(zip(self.PricingModel.NodesIndex,get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'v' in x.VarName)].values))

        self.PricingModel.TruckCumulativeDemand = dict()
        self.PricingModel.TruckPassedNode = dict()
        self.PricingModel.TruckCumulativeDistance = dict()
        self.PricingModel.DecisionIfThen = dict()
        for i in range(self.PricingModel.NoTruckPerPath):
            self.PricingModel.TruckCumulativeDemand[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)),get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'y'+str(i+1) in x.VarName)].values))

            self.PricingModel.TruckPassedNode[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'alpha'+str(i+1) in x.VarName)].values))

            self.PricingModel.TruckCumulativeDistance[i+1] = dict(zip(np.concatenate((self.PricingModel.DepotAllIndex,self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'g'+str(i+1) in x.VarName)].values))

            self.PricingModel.DecisionIfThen[i+1] = dict(zip(np.concatenate((self.PricingModel.TruckCusIndexDict[i+1],self.PricingModel.DockingIndex)), get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'gamma'+str(i+1) in x.VarName)].values))

        self.PricingModel.NodesBeta = dict(zip(self.PricingModel.NodesIndex, get_all_vars_pd[get_all_vars_pd.apply(lambda x: 'beta' in x.VarName)].values))    
    
    def removeConstrBranching(self):
        all_constrs = pd.Series(self.PricingModel.model.getConstrs())
        bch_constrs = all_constrs[all_constrs.apply(lambda x: 'branching' in x.ConstrName)]
        self.PricingModel.model.remove(bch_constrs.values.tolist())
        self.PricingModel.model.update()
    
    def getSummationXj(self):
        summation = list()
        for b_con in self.all_branching_decisions:
            jSet = bnp.generateJSet(self.RmpModel.relaxedModel,b_con)
#             print(b_con,np.sum([x.X for x in jSet['r1r2']]))
            summation.append(np.sum([x.X for x in jSet['r1r2']]))
        summationXj = pd.Series(summation,index=self.all_branching_decisions)
        return summationXj

## Branch and Price Class

## Branch and Price Framework
def BranchAndPriceWithFixCost(_bnp_dict):
    # GLOBAL: allcombiJ, unsolvedNodePools, UpperBound, allbranchingdecision
    _path = _bnp_dict['path'] 
    all_depot = _bnp_dict['all_depot'] 
    docking = _bnp_dict['docking'] 
    truck_cus_nodes = _bnp_dict['truck_cus_nodes'] 
    truck_cus_nodes_dict = _bnp_dict['truck_cus_nodes_dict'] 
    drone_cus_nodes = _bnp_dict['drone_cus_nodes'] 
    arcs_truck = _bnp_dict['arcs_truck'] 
    arcs_drone = _bnp_dict['arcs_drone'] 
    arcs_truck_dict = _bnp_dict['arcs_truck_dict']
    no_truck = _bnp_dict['no_truck'] 
    truck_distance = _bnp_dict['truck_distance'] 
    drone_distance = _bnp_dict['drone_distance'] 
    constant_dict = _bnp_dict['constant_dict'] 
    customer_demand_df = _bnp_dict['customer_demand_df']
    customer_demand = customer_demand_df.values.tolist()
    # define master p
    t_str = time.time()
    RmpModel = md.VRPdMasterProblemDebugged(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
    RmpModel.buildModel()
    RmpModel.model.setParam('OutputFlag',0)
    t_fin_build_rootrmp = time.time()
    RmpModel.solveRelaxedModel()
    duals = RmpModel.getDuals()
    t_fin_solved_root = time.time()
    #=================================
    TemplateRmpModel = md.VRPdMasterProblemDebugged(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
    TemplateRmpModel.model = RmpModel.model.copy()  
    TemplateRmpModel.model.setParam('OutputFlag',0)

    TemplateRmpModel.relaxedModel = TemplateRmpModel.model.relax()
    TemplateRmpModel.Path = RmpModel.Path.copy()
    #=================================

    all_constrs = RmpModel.model.getConstrs()
    customer_coverage_constrs = [c for c in all_constrs if 'customer_coverage' in c.ConstrName]
    all_branching_decisions =[x for x in combinations(customer_coverage_constrs,2)]
    pre_chosen_constrs = dict([(con,None) for con in all_branching_decisions])

    UpperBound = {np.inf:0}
    # define pricing problem
    PricingModel = md.VRPdPricingProblemDebugged(all_depot,truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,docking,arcs_truck,arcs_truck_dict,arcs_drone,\
                    duals,customer_demand,no_truck,truck_distance,drone_distance,constant_dict)
    PricingModel.buildModel()
    PricingModel.model.setParam('OutputFlag',0)
    t_fin_build_pricing = time.time()

    # define root node
    root_node = bnp.BnPNode('0_ROOT',RmpModel, PricingModel, pre_chosen_constrs)
    # add to unsolved node pool
    unsolvedPools = [root_node]

    solvedNodePools = dict()
    integerSolPools = dict()

    iteration = 0
    while len(unsolvedPools)>0:
        iteration+=1
        currentNode = unsolvedPools[0]
        print('ITERATION:',iteration)
        print('Unsolved NODES',[b.Name for b in unsolvedPools])
        print('GOING TO SOLVE',currentNode.Name,'\n')
        print('Branching Conditions:',currentNode.PreProcessedConstrs.values())
    #     print('Variables:', currentNode.RmpModel.model.getVars())
        del unsolvedPools[0]
        currentNode.RmpModel = TemplateRmpModel
        bnp.addColsFromColsPool(currentNode.RmpModel,RmpModel)

        currentNode.RmpModel.model.setParam('OutputFlag',0)
        
#         currentNode.RmpModel.relaxedModel = RmpModel.relaxedModel.copy()
        currentNode.RmpModel.relaxedModel = currentNode.RmpModel.model.relax()
#         currentNode.RmpModel.Path = RmpModel.Path.copy()
    
        all_constrs = currentNode.RmpModel.model.getConstrs()
        customer_coverage_constrs = [c for c in all_constrs if 'customer_coverage' in c.ConstrName]
        all_braching_decisions =[x for x in combinations(customer_coverage_constrs,2)]
        print('VarBefore(Original):',len(currentNode.RmpModel.model.getVars() ))
        currentNode.RmpModel = bnp.branchRmp(currentNode.RmpModel, currentNode.PreProcessedConstrs)
        print('VarAfterBranch:',len(currentNode.RmpModel.model.getVars() ))
        
        ColGen = ColumnGenerationWithFixCost(currentNode.RmpModel ,currentNode.PricingModel,\
                                  _all_branching_decisions=all_branching_decisions,\
                                  _PreProcessedConstrs=currentNode.PreProcessedConstrs,\
                                   _originalRmpModel=RmpModel,_affliation=currentNode.Name)
        ColGen.generateColumns()
        currentNode.RmpModel = ColGen.RmpModel
        
        print('VarAfterCol:',len(currentNode.RmpModel.model.getVars() ))
        if ColGen.StatusCode==3: 
            currentNode.NodeStatus = 'FathomedByColGenInfeasible'
            solvedNodePools[currentNode.Name] = (currentNode,ColGen)
            continue

        # Calculate summation xj of node
        currentNode.SummationXj = ColGen.getSummationXj()
        #Prune by bounding
        lpRmpObj = ColGen.RmpModel.relaxedModel.getObjective().getValue()
        if lpRmpObj>np.min(list(UpperBound.keys())):
            print('Prune by Bounding, %d'%lpRmpObj,'UpperBound:%d'%np.min(list(UpperBound.keys())))
            currentNode.NodeStatus = 'FathomedByBounding'
            solvedNodePools[currentNode.Name] = (currentNode,ColGen)
            #update node status: FathomedByBounding
            continue

        #Prune by optimality
        fractional_sol = bnp.getFractionalSol(ColGen.RmpModel)
#         print('fractional_sol:',fractional_sol)
        if bnp.getFractionalSol(ColGen.RmpModel).shape[0]==0: 
            print('Prune by Optimality')
            #update the UpperBound
            ans = ColGen.RmpModel.getRelaxSolution()
            UpperBound[lpRmpObj] = ans
            print('New bound:',lpRmpObj)
            
            integerNode = bnp.BnPNode(currentNode.Name,[], PricingModel, currentNode.PreProcessedConstrs.copy())
            integerNode.RmpModel = md.VRPdMasterProblemDebugged(_path, all_depot, truck_cus_nodes,truck_cus_nodes_dict,drone_cus_nodes,\
                                    docking,arcs_truck,arcs_drone,no_truck,truck_distance,drone_distance,\
                                   constant_dict)
            integerNode.RmpModel.Path = currentNode.RmpModel.Path.copy()
            integerNode.NodeStatus = 'FathomedOptimality'
            integerNode.ObjVal = currentNode.RmpModel.relaxedModel.ObjVal
            integerNode.OptimalSol = currentNode.RmpModel.relaxedModel.getAttr('X')
            currentNode.RmpModel.model.update()
            integerNode.RmpModel.model = currentNode.RmpModel.model.copy()
            integerNode.RmpModel.relaxedModel = integerNode.RmpModel.model.relax()
            integerSolPools[integerNode.Name] = integerNode
            solvedNodePools[integerNode.Name] = (integerNode,ColGen)
            #update node status: FathomedByOptimality
            continue
    #         break
        pred_constrs=[]
        for k,v in currentNode.PreProcessedConstrs.items():
            if v is not None:pred_constrs+=[k[0].ConstrName,k[1].ConstrName]
        f_j = currentNode.SummationXj[(currentNode.SummationXj>epsilon)&(currentNode.SummationXj<1-epsilon)]
        idx_pd = pd.Series(f_j.index)
#         not_included_constrs =f_j[idx_pd[idx_pd.apply(lambda x: (x[0].ConstrName not in pred_constrs)and(x[1].ConstrName not in pred_constrs))]]
        not_included_constrs =f_j[idx_pd[idx_pd.apply(lambda x: (x[0].ConstrName not in pred_constrs)or(x[1].ConstrName not in pred_constrs))]]
#         print(f_j,not_included_constrs)
        if not_included_constrs.shape[0]!=0:
            chosen_j = not_included_constrs[[not_included_constrs.idxmax()]].index[0]
        else:
            chosen_j = f_j[[f_j.idxmax()]].index[0]
        print('Chosen J:',chosen_j)
        # Branching
        zero_node, one_node = bnp.branchOnNode(currentNode,chosen_j,iteration)
        unsolvedPools+=[zero_node, one_node]
        currentNode.NodeStatus = 'Solved'
        solvedNodePools[currentNode.Name] = (currentNode,ColGen)
        print('Adding new branches to pool!\n')
#         print('==============================\n')

        if iteration>100: 
            print('NUMBER OF ITERATIONS EXCEED TOLERANCE')
            break
    t_fin_bnp = time.time()
    timeStatistic = {'Total Operation time':t_fin_bnp-t_str,\
                    'Root RMP Build-up':t_fin_build_rootrmp-t_str,\
                    'Solving Root RMP':t_fin_solved_root-t_fin_build_rootrmp,\
                    'Pricing Build-up':t_fin_build_pricing-t_fin_solved_root,\
                    'Solving MIP through BnP':t_fin_bnp-t_fin_build_pricing}
    print('Congratulation FINISH Branch and Price!@jd;adjfaderadf')
    print('Total Operation time:',timeStatistic['Total Operation time'])
    print('Root RMP Build-up:',timeStatistic['Root RMP Build-up'])
    print('Solving Root RMP:',timeStatistic['Solving Root RMP'])
    print('Pricing Build-up:',timeStatistic['Pricing Build-up'])
    print('Solving MIP through BnP:',timeStatistic['Solving MIP through BnP'])
    return solvedNodePools,integerSolPools,timeStatistic,RmpModel
    
    
    
   




class BnPNode():
    def __init__(self, _name, _rmpModel, _pricingModel, _preProcessedConstrs , _nodeStatus='Unsolved', _summationXj=None ,_obj_val=None, _optimal_sol=None ):
        self.Name = _name
        self.RmpModel = _rmpModel
        self.PricingModel = _pricingModel
        self.PreProcessedConstrs = _preProcessedConstrs
        self.NodeStatus = _nodeStatus
        self.SummationXj = _summationXj
        self.ObjVal = _obj_val
        self.OptimalSol = _optimal_sol

        
def addColsFromColsPool(_rmp_model,_original_rmp_model):
    var_in_cols_pool = _original_rmp_model.Path
    var_in_current_rmp = _rmp_model.Path
    new_var_tobe_added = var_in_cols_pool.loc[~var_in_cols_pool.index.isin(var_in_current_rmp.index)]
    #Added to Path
    _rmp_model.Path=var_in_current_rmp.append(new_var_tobe_added)

    #Added to gurobi var
    name_var_tobe_added = new_var_tobe_added.index.tolist() #Just list of name
    get_added_var = _original_rmp_model.model.getVarByName #getVarByName module
    get_col_var = _original_rmp_model.model.getCol #getCol module: input=var

    col_pd = pd.Series(name_var_tobe_added).apply(lambda x: get_col_var(get_added_var(x)))
    col_obj = pd.Series(name_var_tobe_added).apply(lambda x: get_added_var(x).Obj)
    col_name = pd.Series(name_var_tobe_added)

    new_col_df = pd.DataFrame([col_pd,col_obj,pd.Series(name_var_tobe_added)])
    # ADDING NEW VARS TO MODEL BY APPLY FUNC
    new_col_df.apply(lambda x: _rmp_model.model.addVar(vtype=GRB.BINARY,lb=0,obj=x[1],column=x[0],name=x[2]))
    _rmp_model.model.update()
    
    
def branchOnNode(bnp_node, chosen_constrs,i):
    '''chosen_constrs: current selected constrains'''
    pre_chosen_constrs_zero = bnp_node.PreProcessedConstrs.copy()
    pre_chosen_constrs_one = bnp_node.PreProcessedConstrs.copy()
    
    pre_chosen_constrs_zero[chosen_constrs]='Zero'
    pre_chosen_constrs_one[chosen_constrs]='One'
    
#     newRmpZero = md.VRPdMasterProblem(bnp_node.RmpModel.Path, bnp_node.RmpModel.AllDepot, bnp_node.RmpModel.TruckCusNodes, bnp_node.RmpModel.TruckCusNodesDict, bnp_node.RmpModel.DroneCusNodes, bnp_node.RmpModel.Docking, bnp_node.RmpModel.TruckArcs, bnp_node.RmpModel.DroneArcs, bnp_node.RmpModel.NoTruckPerPath, bnp_node.RmpModel.TruckDistance, bnp_node.RmpModel.DroneDistance, bnp_node.RmpModel.constant_dict)
#     newRmpZero.model = bnp_node.RmpModel.model.copy()
    
#     newRmpOne = md.VRPdMasterProblem(bnp_node.RmpModel.Path, bnp_node.RmpModel.AllDepot, bnp_node.RmpModel.TruckCusNodes, bnp_node.RmpModel.TruckCusNodesDict, bnp_node.RmpModel.DroneCusNodes, bnp_node.RmpModel.Docking, bnp_node.RmpModel.TruckArcs, bnp_node.RmpModel.DroneArcs, bnp_node.RmpModel.NoTruckPerPath, bnp_node.RmpModel.TruckDistance, bnp_node.RmpModel.DroneDistance, bnp_node.RmpModel.constant_dict)
#     newRmpOne.model = bnp_node.RmpModel.model.copy()
    newRmpZero=None
    newRmpOne=None
#     newRmpZero.model.setParam('OutputFlag',0)
#     newRmpOne.model.setParam('OutputFlag',0)
    
#     new_zero_rmp = brachZero(newRmpZero,pre_chosen_constrs_zero,chosen_constrs)
#     new_one_rmp = brachOne(newRmpOne,pre_chosen_constrs_one,chosen_constrs)
    
    # append J branched in the child node
    zero_node = BnPNode(str(i)+'_zero',newRmpZero, bnp_node.PricingModel, pre_chosen_constrs_zero)
    one_node = BnPNode(str(i)+'_one',newRmpOne, bnp_node.PricingModel, pre_chosen_constrs_one)
    return zero_node, one_node

def branchRmp(rmp_model, pre_chosen_constrs):
    new_zero_rmp_leaf = rmp_model.model
    # Not Optimized
    deleted_vars=[]
    for p_j1,p_j0 in pre_chosen_constrs.items(): deleted_vars+=generateJSetFromBranch(new_zero_rmp_leaf,p_j1,p_j0)
    deleted_vars_name = [v.VarName for v in deleted_vars]
    unique_deleted_vars = pd.Series(deleted_vars_name).unique()
    print('TO BE deleted vars:',len(unique_deleted_vars))
    print(unique_deleted_vars)
    new_zero_rmp_leaf.remove(deleted_vars)
    rmp_model.Path.drop(deleted_vars_name,inplace=True)
    rmp_model.model.update()
    return rmp_model

def brachZero(rmp_model, pre_chosen_constrs, chosen_constrs):
    new_zero_rmp_leaf = rmp_model.model
    # Not Optimized
    deleted_vars=[]
#     for p_j in pre_chosen_constrs: deleted_vars+=generateJSetFromBranch(new_zero_rmp_leaf,p_j[0],p_j[1])
    for p_j1,p_j0 in pre_chosen_constrs.items(): deleted_vars+=generateJSetFromBranch(new_zero_rmp_leaf,p_j1,p_j0)
#     print(deleted_vars)
    deleted_vars+=generateJSetFromBranch(new_zero_rmp_leaf,chosen_constrs,'Zero')
#     deleted_vars+=generateJSetFromBranch(new_zero_rmp_leaf,chosen_constrs,0)
    new_zero_rmp_leaf.remove(deleted_vars)
    deleted_vars_name = [v.VarName for v in deleted_vars]
    rmp_model.Path.drop(deleted_vars_name,inplace=True)
    rmp_model.model.update()
    return rmp_model

def brachOne(rmp_model, pre_chosen_constrs, chosen_constrs):
    new_one_rmp_leaf = rmp_model.model
    # Not Optimized
    deleted_vars=[]
    for p_j1,p_j0 in pre_chosen_constrs.items(): deleted_vars+=generateJSetFromBranch(new_one_rmp_leaf,p_j1,p_j0)
#     for p_j in pre_chosen_constrs: deleted_vars+=generateJSetFromBranch(new_one_rmp_leaf,p_j[0],p_j[1])
#     print(deleted_vars)
    deleted_vars+=generateJSetFromBranch(new_one_rmp_leaf,chosen_constrs,'One')
#     deleted_vars+=generateJSetFromBranch(new_one_rmp_leaf,chosen_constrs,1)
    new_one_rmp_leaf.remove(deleted_vars)
    deleted_vars_name = [v.VarName for v in deleted_vars]
    rmp_model.Path.drop(deleted_vars_name,inplace=True)
    rmp_model.model.update()
    return rmp_model


def getFractionalSol(master):
    a = pd.Series(master.relaxedModel.getAttr('X'))
    return a[(a>epsilon*10)&(a<(1-epsilon*10))]
#     return a[a>2]

def generateJSetFromBranch(rmp_model, constrs, b_type):
    rmp_vars = rmp_model.getVars()
    delete_vars = []
    if b_type=='Zero': #0:
        delete_vars = [x for x in rmp_vars if ((rmp_model.getCoeff(constrs[0],x)==1)\
            and(rmp_model.getCoeff(constrs[1],x)==1))]
    elif b_type=='One':#1:
        delete_vars = [x for x in rmp_vars if (((rmp_model.getCoeff(constrs[0],x)==0)\
            and(rmp_model.getCoeff(constrs[1],x)==1))or((rmp_model.getCoeff(constrs[0],x)==1)\
            and(rmp_model.getCoeff(constrs[1],x)==0)))]
    return delete_vars    
        
    
def generateJSet(rmp_model, constrs):
    # don't need solution
    j_sets = {'r1r2':[], '~r1r2':[], 'r1~r2':[]}
    rmp_vars = rmp_model.getVars()
    for x in rmp_vars:
        if ((rmp_model.getCoeff(constrs[0],x)==1)\
            and(rmp_model.getCoeff(constrs[1],x)==1)):j_sets['r1r2'].append(x)
        elif ((rmp_model.getCoeff(constrs[0],x)==0)\
            and(rmp_model.getCoeff(constrs[1],x)==1)):j_sets['~r1r2'].append(x)
        elif ((rmp_model.getCoeff(constrs[0],x)==1)\
            and(rmp_model.getCoeff(constrs[1],x)==0)):j_sets['r1~r2'].append(x)
    return j_sets