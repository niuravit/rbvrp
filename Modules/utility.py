import numpy as np
import random 
import pandas as pd
import sys
import pickle as pk
from datetime import datetime
from itertools import combinations,permutations 
import importlib
sys.path.insert(0,'/Users/admin/Desktop/EE_Year4_Term2/OR/VRP+d/Combine_variables/Modules')
    
# Utility function

def ReloadModules():
    modules = ['visualize_sol','random_instance','initialize_path','model','experiment','bnp','utility']
    for m in modules: importlib.reload(sys.modules[m])

def getFormatNodeName(s,mode=1,affli=None):
    '''mode=1: remove vehicle affliation
    mode=2: add vehicle affliation'''
    if mode==1:
        if ('depot' in s) and (affli is not None):
            return s+'_%s'%affli
        elif ('dock' in s ) or ('depot' in s) or (affli is not None):
            return s
        else: return '_'.join(s.split('_')[:-1])
    elif mode==2:
        return s+'_%s'%affli
    
def saveObjToPath(_obj,f_name,f_path=None):
    if f_path is None: f_path = './'
    with open(f_path+f_name+'.pk','wb') as f1:
        pk.dump(_obj,f1)
        f1.close()
    print('Save Obj to:',f_path+f_name)

def openObjFromPath(f_name,f_path=None):
    if f_path is None: f_path = './'
    with open(f_path+f_name+'.pk','rb') as f1:
        log = pk.load(f1)
        f1.close()
    return log

def getFormattedRecord(_record_df):
    timeStat_df = _record_df['SolvingTime'].apply(pd.Series)
    _record_df['#Nodes']=_record_df['#Cus']+_record_df['#Dock']
    data_df = _record_df[['InstanceName','#Cus','#Dock','#Nodes',\
                           'OBJ','InitColsType','#InitCols',\
                           '#ColsAdded@Root','#NodesExplored','#TotalColsAdded','#TruckUsed','#DroneUsed',\
 'OriginalRmpMipObj','OriginalRmpRelaxObj',\
                            'PercentIntersectionColumn']]
    out_table = pd.concat([data_df,timeStat_df[['Solving MIP through BnP','Root RMP Build-up','Solving Root RMP','Pricing Build-up','Total Operation time']]],axis=1)
    return out_table

def getFormattedRecord2(_record_df):
    timeStat_df = _record_df['SolvingTime'].apply(pd.Series)
    _record_df['#Nodes']=_record_df['#Cus']+_record_df['#Dock']
    _record_df['#CusServTruck']=_record_df['#CusServTruck'].apply(lambda x: x[x>0.01].shape[0])
    _record_df['#CusServDrone']=_record_df['#CusServDrone'].apply(lambda x: x[x>0.01].shape[0])
    data_df = _record_df[['InstanceName','#Cus','#Dock','#Nodes',\
                           'OBJ','InitColsType','#InitCols',\
                           '#ColsAdded@Root','#NodesExplored','#TotalColsAdded','#TruckUsed','#DroneUsed','#CusServTruck','#CusServDrone',\
 'OriginalRmpMipObj','OriginalRmpRelaxObj',\
                            'PercentIntersectionColumn']]
    out_table = pd.concat([data_df,timeStat_df[['Solving MIP through BnP','Root RMP Build-up','Solving Root RMP','Pricing Build-up','Total Operation time']]],axis=1)
    return out_table
 
def removeConstrBranching(self):
        all_constrs = pd.Series(self.PricingModel.model.getConstrs())
        bch_constrs = all_constrs[all_constrs.apply(lambda x: 'branching' in x.ConstrName)]
        self.PricingModel.model.remove(bch_constrs.values.tolist())
        self.PricingModel.model.update()
        
def getMipGurobiObj(_rmp_model):
    '''_rmp_model: VRPdMasterClass '''
    _rmp_model.model.setParam('OutputFlag',0)
    _rmp_model.solveModel(10000,0)
    mip_obj = _rmp_model.model.ObjVal
    print('OBJ Gurobi MIP:', mip_obj)
    
    _rmp_model.relaxedModel.setParam('OutputFlag',0)
    _rmp_model.solveRelaxedModel()
    relax_obj = _rmp_model.relaxedModel.ObjVal
    print('OBJ Gurobi relax:',relax_obj)
    return (mip_obj, relax_obj)

def getPercentIntersec(_solvedNodePool_couple):
    _rmp_model_1=_solvedNodePool_couple[0]
    _rmp_model_2=_solvedNodePool_couple[1]
    vars_pd_1 =getVarsAddedByColGen(_rmp_model_1).drop_duplicates()
    vars_pd_2 =getVarsAddedByColGen(_rmp_model_2).drop_duplicates()
    intersection_vars = pd.merge(vars_pd_1,vars_pd_2,on=vars_pd_1.columns[0],how='inner')
    no_intersection_vars = intersection_vars.shape[0]
    no_vars_pd_1 = vars_pd_1.shape[0]
    no_vars_pd_2 = vars_pd_2.shape[0]
#     print(no_vars_pd_1, no_vars_pd_2,no_intersection_vars )
    percent_intersec = no_intersection_vars*100/(no_vars_pd_1+no_vars_pd_2-no_intersection_vars)
    print('#A/#B/#intersec:{}/{}/{}'.format(no_vars_pd_1,no_vars_pd_2,no_intersection_vars),'Percent intersection:',percent_intersec)
    return percent_intersec

def getVarsAddedByColGen(_rmp_model):
    '''input:_rmp_model
    output: vars_pd (attribute Path, dataframe)'''
    vars_pd = _rmp_model.Path[_rmp_model.Path.index.str.contains('colGen')]
    #round coeff to be integer
    if vars_pd.shape[0]!=0:vars_pd[:][vars_pd.columns[0]] = vars_pd.apply(lambda x: ''.join(np.around(x[0],decimals=0).astype(int).astype(str).tolist()),axis=1)
    #     vars_pd.loc[:,vars_pd.columns[0]] = vars_pd.apply(lambda x: ''.join(np.around(x[0],decimals=0).astype(int).astype(str).tolist()),axis=1)
#     print(vars_pd)
    return vars_pd

def getAdditionalStat(_solvedNodePools):
    (mip_obj_1, relax_obj_1) = getMipGurobiObj(_solvedNodePools['originalRmp'])
    total_cols_1 = getVarsAddedByColGen(_solvedNodePools['originalRmp']).shape[0]
    add_result_1 = [total_cols_1,mip_obj_1,relax_obj_1]
    return add_result_1

def addRecord(_solveNodePool_list,_result_list,_record_df):
    all_combi_intersec = list(combinations([d['originalRmp'] for d in _solveNodePool_list],2))
    for node_idx in range(len(_solveNodePool_list)):
        add_result = getAdditionalStat(_solveNodePool_list[node_idx])
        if len(all_combi_intersec)==3:
            add_result.append(getPercentIntersec(all_combi_intersec[node_idx]))
        elif len(all_combi_intersec)==1:
            add_result.append(getPercentIntersec(all_combi_intersec[0]))
        Col_names=_record_df.columns.values
        _record_df=_record_df.append([dict(zip(Col_names,_result_list[node_idx]+add_result))])
    _record_df.index = range(_record_df.shape[0])
    return _record_df
#Not work!
def saveSolvedNodeLog(instance_name,solvedNodePools,integerSolPools,result,attrib=''):
    save_log = {"solveNodePools":solvedNodePools,
            'integerSolPools':integerSolPools_1,
           'InstanceType:':instance_name,
           'GenMode:':solvedNodePools['gen_mode'],
            'Result:':result}
    file_name = instance_name+'_'+solvedNodePools['gen_mode']+attrib
    with open('./%s.pickle'%file_name ,mode='wb') as f1:
        pk.dump(save_log,f1)