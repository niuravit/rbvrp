import numpy as np
import random 
from scipy.spatial import distance
import pandas as pd
import pickle as pk

# depot = ['depot']
# depot_s = ['depot_s']
# depot_t = ['depot_t']
def import_instance(_dir,_file_name):
    print("Importing instance:",_file_name);
    with open(_dir+'%s.pickle'%_file_name,'rb') as f1:
        r_instance = pk.load(f1)
    distance_matrix = r_instance['distance_matrix']
    node_trace = r_instance['node_trace']
    customer_demand = r_instance['customer_demand_df']
    node_position = r_instance['node_position']
    return distance_matrix,node_trace,customer_demand,node_position

def rand_cust_demand(_customers, _lb = 1, _ub = 6):
    demand = np.random.randint(_lb,_ub,len(_customers)).tolist()
    customer_demand = pd.Series(demand,index = _customers)
    return customer_demand

def rand_uniform_dis_mat(_node_list,_depot_node, _service_region_len = 20,_norm_order=1):
    # random the nodes position
    no_nodes = len(_node_list)
    _distance_matrix=dict()
    _nodes_position = dict()
    rand_seed = np.random.uniform(low=-1.5,high=0)
    for i in range(no_nodes):
        if _node_list[i]!=_depot_node:
            _nodes_position[_node_list[i]] = list(np.random.rand(2)*_service_region_len)
#     print(nodes_position)
    _nodes_position[_depot_node] = [_service_region_len/2,_service_region_len/2]
#     print(nodes_position)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i>=j:
                point_i = np.array(_nodes_position[_node_list[i]])
                point_j = np.array(_nodes_position[_node_list[j]])
                t_dist_rand = np.linalg.norm(point_i-point_j,ord=_norm_order)
                _distance_matrix[(_node_list[i],_node_list[j])] = t_dist_rand
                _distance_matrix[(_node_list[j],_node_list[i])] = t_dist_rand
    _distance_matrix[(_depot_node,_depot_node)] = 0
    return _distance_matrix, _nodes_position

def rand_uniform_corner_depot_dis_mat(_node_list,_depot_node, _service_region_len = 20,_norm_order=1):
    # random the nodes position
    no_nodes = len(_node_list)
    _distance_matrix=dict()
    _nodes_position = dict()
    rand_seed = np.random.uniform(low=-1.5,high=0)
    for i in range(no_nodes):
        if _node_list[i]!=_depot_node:
            _nodes_position[_node_list[i]] = list(np.random.rand(2)*_service_region_len)
#     print(nodes_position)
    _nodes_position[_depot_node] = [0,0]
#     print(nodes_position)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i>=j:
                point_i = np.array(_nodes_position[_node_list[i]])
                point_j = np.array(_nodes_position[_node_list[j]])
                t_dist_rand = np.linalg.norm(point_i-point_j,ord=_norm_order)
                _distance_matrix[(_node_list[i],_node_list[j])] = t_dist_rand
                _distance_matrix[(_node_list[j],_node_list[i])] = t_dist_rand
    _distance_matrix[(_depot_node,_depot_node)] = 0
    return _distance_matrix, _nodes_position

def rand_uniform_avg_depot_dis_mat(_node_list,_depot_node, _service_region_len = 20,_norm_order=1):
    # random the nodes position
    no_nodes = len(_node_list)
    _distance_matrix=dict()
    _nodes_position = dict()
    rand_seed = np.random.uniform(low=-1.5,high=0)
    for i in range(no_nodes):
        if _node_list[i]!=_depot_node:
            _nodes_position[_node_list[i]] = list(np.random.rand(2)*_service_region_len)
#     print(nodes_position)
    _nodes_position[_depot_node] = np.mean(np.array(list(_nodes_position.values())), axis=0)
#     print(nodes_position)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i>=j:
                point_i = np.array(_nodes_position[_node_list[i]])
                point_j = np.array(_nodes_position[_node_list[j]])
                t_dist_rand = np.linalg.norm(point_i-point_j,ord=_norm_order)
                _distance_matrix[(_node_list[i],_node_list[j])] = t_dist_rand
                _distance_matrix[(_node_list[j],_node_list[i])] = t_dist_rand
    _distance_matrix[(_depot_node,_depot_node)] = 0
    return _distance_matrix, _nodes_position

def rand_uniform_radius_center_depot_dis_mat(_node_list,_depot_node, _service_region_len = 20,_norm_order=1):
    # random the nodes position
    no_nodes = len(_node_list)
    _distance_matrix=dict()
    _nodes_position = dict()
    rand_seed = np.random.uniform(low=-1.5,high=0)
    for i in range(no_nodes):
        radius = np.random.rand(1)[0]*0.5*_service_region_len
        degree = np.random.rand(1)[0]*2*np.pi
        if _node_list[i]!=_depot_node:
            _nodes_position[_node_list[i]] = [radius*np.cos(degree),radius*np.sin(degree)]
#     print(nodes_position)
    _nodes_position[_depot_node] = [0,0]
#     print(nodes_position)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i>=j:
                point_i = np.array(_nodes_position[_node_list[i]])
                point_j = np.array(_nodes_position[_node_list[j]])
                t_dist_rand = np.linalg.norm(point_i-point_j,ord=_norm_order)
                _distance_matrix[(_node_list[i],_node_list[j])] = t_dist_rand
                _distance_matrix[(_node_list[j],_node_list[i])] = t_dist_rand
    _distance_matrix[(_depot_node,_depot_node)] = 0
    return _distance_matrix, _nodes_position