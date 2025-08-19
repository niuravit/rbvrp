import plotly.graph_objects as go
import networkx as nx
# import plotly.offline as py
from itertools import combinations,permutations 
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import random 
import pandas as pd
import copy



def create_nodes(no_dock,no_customer,_dock_prefix='dock',_n_prefix='c',_dep_prefix='O'):
    nodes = set()
    docking = [_dock_prefix+'_'+str(i+1) for i in range(no_dock)]
    customers = [_n_prefix+'_'+str(i+1) for i in range(no_customer)]
    depot = [_dep_prefix]
    depot_s = [_dep_prefix+'_s']
    depot_t = [_dep_prefix+'_t']
    all_depot = depot_s+depot_t
    nodes = list(nodes.union(set(docking),set(customers),set(depot)))
    no_node = len(nodes)
    node_combi = list(combinations(nodes,2))
    arc_permute = [list(permutations(list(c))) for c in node_combi ]
    sh = np.shape(arc_permute)
    arcs = np.reshape(arc_permute,(sh[0]*sh[1],sh[2])).tolist()
    arcs = [tuple(l) for l in arcs]
    output_dict = dict(zip(['docking','customers','depot','depot_s','depot_t','all_depot','nodes','arcs'
],[docking,customers,depot,depot_s,depot_t,all_depot,nodes,arcs
  ]))
#     print(output_dict)
    return output_dict

def create_color_list(_nodes,_prefix_color_map=None):
    if _prefix_color_map is None: cr_list = dict.fromkeys(_nodes, 0)
    else:
        cr_list = []
        for nn in _nodes:
            pref_idx = nn.rfind('_')
            if nn.rfind('_')==-1: pref_idx = len(nn)
            cr_list.append((nn, _prefix_color_map[nn[:pref_idx]]))
        cr_list = dict(cr_list)
    return cr_list

def create_node_trace(_nodes_position_map,_color_map,_colorscale=None,_node_size=None,_symbol_dict=None):
    # colorscale options
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    node_name_ss = pd.Series(_nodes_position_map.keys())
    if _colorscale is None:
        _colorscale='Blackbody'
    if _node_size is None:
        _node_size=30
    if _symbol_dict is None: _symbol_dict = dict.fromkeys(node_name_ss.to_list(),'circle')
    else:
        sym_list = []
        for nn in node_name_ss.to_list():
            pref_idx = nn.rfind('_')
            if nn.rfind('_')==-1: pref_idx = len(nn)
            sym_list.append((nn,_symbol_dict[nn[:pref_idx]]))
        _symbol_dict = dict(sym_list)
        
    node_position_coor = np.matrix(list(_nodes_position_map.values())).T
    x_order = node_position_coor.getA()[0]
    y_order = node_position_coor.getA()[1]
    node_symbol_list = node_name_ss.apply(lambda x: _symbol_dict[x]).to_list()
    color_map_list = node_name_ss.apply(lambda x: _color_map[x]).to_list()
    # Define Nodes' properties
    node_trace = go.Scatter(
        x=x_order, y=y_order,
        mode='markers+text',
        text = node_name_ss.to_list(),
        textposition="top center",
        name='Node',
        marker=dict(
            showscale=False,
            colorscale=_colorscale,
            reversescale=True,
            color=color_map_list,
            symbol=node_symbol_list,
            size=_node_size,
            line_width=1))
#     print(node_trace)
    return node_trace


def create_arc_trace(_node_trace,_arc_list,_trace_name='',_line_width=None,_line_color=None,_dash_config=None):
    if _line_width is None: _line_width = 2.5
    if _line_color is None: _line_color = '#888'
    node_num = len(_node_trace['text'])
    node_dict = dict([(_node_trace['text'][n],(_node_trace['x'][n],_node_trace['y'][n])) for n in range(node_num)])
    edge_x = []
    edge_y = []
    for edge in _arc_list:
        #get coordinate of each edge;start point & end point!
        #Use None as separator between arcs.
        x0, y0 = node_dict[edge[0]]
        x1, y1 = node_dict[edge[1]]
        edge_x.append(x0);edge_x.append(x1);edge_x.append(None)
        edge_y.append(y0);edge_y.append(y1);edge_y.append(None)
    edge_trace = go.Scatter( x=edge_x, y=edge_y,
                            text = "",
                            textposition="middle center",
            line=dict(width=_line_width, color=_line_color,dash=_dash_config),
#             hoverinfo='text', 
                            mode='lines', name=_trace_name)
    arrow_ann = [dict(showarrow=True,arrowcolor=_line_color,
                     arrowhead=5,arrowsize=_line_width,
                 x=(2*edge_x[i]+8*edge_x[i+1])/(10),
                 y=(2*edge_y[i]+8*edge_y[i+1])/(10),
                     xref='x', yref='y',
                 ax=(3*edge_x[i]+7*edge_x[i+1])/(10),
                 ay=(3*edge_y[i]+7*edge_y[i+1])/(10),
#                  ax=edge_x[i],
#                  ay=edge_y[i],
                    axref='x', ayref='y')
                 for i in range(0, len(edge_x),3)]
    
    return dict([('edge_trace',edge_trace),('arrow_annotation',arrow_ann)])


def generate_arcs_from_node_list(_node_list):
    path_arcs = [(_node_list[n],_node_list[n+1]) for n in range(len(_node_list)-1)]
    return path_arcs

def plot_network(_path_arcs_list,_node_trace,_display_cus_dem=True,_cus_dem=None,_symbol_dict=None,_title=None,_save_to_file=None,_display_plot=True,_display_info_table=False,_show_all_info = True):
    '''Input: 
        node_trace:= object storing information of the node output from create_node_trace()
        path_arcs_list:= List of objects storing information of each path_arcs
            e.g. path_arcs_list = [path_arcs1,path_arcs2, ...], s.t. 
                path_arcs1 = {arcs_list: [('O','c_1'),('c_1','c_2'),('c_2','O')]
                                  config: {'line_width':.., 'line_color':..., 'dash':..., 'name':...}}
        
        _display_cus_dem: show demand after node name
        _cus_dem: dictionary defining demand for each node, default is zero
        _symbol_dict= dictionary defining symbol(circle, diamond, square) for each prefix, default is circle
        _title: name of the graph
        _save_to_file: path for saving file
        Output: a graph
    '''
    node_ss = pd.Series(_node_trace['text'])
    _node_trace4plot = copy.deepcopy(_node_trace)
    if _title is None: _title = 'Network graph made with Python'
    if _cus_dem is None: _cus_dem = dict.fromkeys(_node_trace['text'],0)
    if _symbol_dict is None: _symbol_dict = dict.fromkeys(_node_trace['text'],'circle')
    if _display_cus_dem: _node_trace4plot['text'] = node_ss.apply(lambda x: x+'[{}]'.format(_cus_dem[x])).to_list()
        
        ###### Plot nodes #########
    plot_nodes_trace = _node_trace4plot
    ###### Plot arcs #########
    path_arcs_trace = [create_arc_trace(_node_trace,p['arcs_list'],
                                       _trace_name=p['config']['name'],_line_width=p['config']['line_width'],
                                        _line_color=p['config']['line_color'],_dash_config=p['config']['dash']) for p in _path_arcs_list]
    arr_annotation_stack = []
    layout=go.Layout(title=_title, titlefont_size=16,
                     showlegend=True,hovermode='closest',
                     margin=dict(b=20,l=5,r=5,t=40),
                     xaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
                     yaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
                     plot_bgcolor='rgba(0,0,0,0)',)
    
    
        
    #######if display table, need to structure subplot
    if _display_info_table:
        table_h = calc_table_height(_path_arcs_list)
        plot_h= 300
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]],
            row_heights=[table_h/(table_h+plot_h),plot_h*2/(table_h+plot_h)]
        )
        for l in range(len(_path_arcs_list)):
            dem_wait_str = "<br>".join([k+': '+str(round(v,2)) for (k,v) in _path_arcs_list[l]['route_info']['demand_waiting'].items()])
            if _show_all_info:
                _path_arcs_list[l]['route_info']['demand_waiting_str'] = dem_wait_str
            else:
                max_key = max(_path_arcs_list[l]['route_info']['demand_waiting'], key=_path_arcs_list[l]['route_info']['demand_waiting'].get)
                _path_arcs_list[l]['route_info']['demand_waiting'] = max_key+': '+str(round(_path_arcs_list[l]['route_info']['demand_waiting'][max_key],2))
        col_keys = _path_arcs_list[0]['info_topics']
        col_width = _path_arcs_list[0]['column_width']
        col_format = _path_arcs_list[0]['column_format']
        header_row = ['c','Route']+[" ".join([s.capitalize() for s in t.split("_")]) for t in col_keys]
#                 ['twAvg<br>Factor','Lr','Demand<br>Waiting','AvgWaiting<br>/Pkg',
#                 'Pkgs','Utilization<br>%']
#         ['tw_avg_factor','lr','demand_waiting_str','avg_waiting_per_pkg','pkgs','utilization']
        
        table_trace2 = go.Table(
                        columnwidth=[1,5]+col_width,
                        columnorder=[i for i in range(len(header_row))],
                        header = dict(height = 30,
                                      values = [['<b>%s</b>'%h] for h in header_row],
                                      line = dict(color='rgb(50, 50, 50)'),
                                      align = ['center'] * len(header_row),
                                      font = dict(color=['rgb(45, 45, 45)'] * len(header_row), size=12),
                                      fill = dict(color='#DEF7FF')),
                        cells = dict(values = [[" "]]+[[_path_arcs_list[idx]['config']['name'] for idx in range(len(_path_arcs_list))]]+[[_path_arcs_list[idx]['route_info'][k] for idx in range(len(_path_arcs_list))] for k in col_keys],
                                     line = dict(color='#506784'),
                                     align = ['center'] * len(header_row),
                                     font = dict(color=['rgb(40, 40, 40)'] * len(col_keys), size=12),
                                     format = [None,None] + col_format,
                                     prefix = [None,None] + [None]*len(header_row),
                                     suffix= [None,None] + [None]*len(header_row),
                                     height = 27,
                                     fill_color = [[_path_arcs_list[idx]['config']['line_color'] for idx in range(len(_path_arcs_list))],['#FFFFFF']]))
#         print(table_trace2)
        fig.add_trace(table_trace2,row=1,col=1)
        fig.add_trace(plot_nodes_trace,row=2,col=1)
#         fig.add_trace(path_arcs_trace,row=2,col=1)
        for path in path_arcs_trace:
            fig.add_trace(path['edge_trace'],row=2,col=1)
            arr_annotation_stack+=(path['arrow_annotation'])
        fig.update_layout(
#             width=800,
            height=plot_h+table_h,
            barmode='stack',
            autosize=False,
            title=_title,
            showlegend=False,
            titlefont_size=16,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
            yaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
            plot_bgcolor='rgba(0,0,0,0)',annotations=arr_annotation_stack)
    else:
        ###### Plot nodes #########
        fig = go.Figure(data=[_node_trace4plot],
                        layout=go.Layout(
                            title=_title, titlefont_size=16,
                            showlegend=True,hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
                            yaxis=dict(showgrid=True, zeroline=True, showticklabels=True,gridcolor='#eee'),
                            plot_bgcolor='rgba(0,0,0,0)',)
                        )
        ###### Plot arcs #########
        path_arcs_trace = [create_arc_trace(_node_trace,p['arcs_list'],
                                           _trace_name=p['config']['name'],_line_width=p['config']['line_width'],
                                            _line_color=p['config']['line_color'],_dash_config=p['config']['dash']) for p in _path_arcs_list]
        arr_annotation_stack = []
        for path in path_arcs_trace:
            fig.add_trace(path['edge_trace'])
            arr_annotation_stack+=(path['arrow_annotation'])
        fig.update_layout(annotations=arr_annotation_stack)
        
    
    
    if _save_to_file is not None:
        fig.write_image(_save_to_file, format='png')
        print(_save_to_file+" is saved succuessfully!")
    if _display_plot:
        fig.show()
    


    
def calc_table_height(df, base=208, height_per_row=30, char_limit=30, height_padding=16.5):
    '''
    df: The dataframe with only the columns you want to plot
    base: The base height of the table (header without any rows)
    height_per_row: The height that one row requires
    char_limit: If the length of a value crosses this limit, the row's height needs to be expanded to fit the value
    height_padding: Extra height in a row when a length of value exceeds char_limit
    '''
    total_height = 0 + base
    for x in range(len(df)):
        total_height += height_per_row
    return total_height
#################################
########Not yet edited###########

def convertSolIndex(sol_idx,class_mastermodel):
    new_path = pd.DataFrame(data = class_mastermodel.Path.loc[sol_idx][0],index =class_mastermodel.mergeDepotCusArcsVar(class_mastermodel.CoeffSeries.values))
    passed = new_path.loc[new_path[0]>0.00001]
    print(passed)
    ans = passed[passed.index.isin(class_mastermodel.mergeDepotCusArcsVar(class_mastermodel.Arcs))]
    return ans

def visualizeSolutions(class_mastermodel,i,sol_list=None,relaxation = False,vis_plot=True,node_trace=None, ignore_fix_cost=False):
    ## VISUALIZE THE SOLUTIONS
    if sol_list is None:
        #input is MIP
        if not relaxation:
            class_mastermodel.model.setParam('SolutionNumber',i)
            sol_list = pd.Series(data = class_mastermodel.model.getAttr('Xn'),index=[v.VarName for v in class_mastermodel.model.getVars()])
            sol_idx = sol_list[sol_list>0.1].index
        else: # input is lp relaxation
            class_mastermodel.relaxedModel.optimize()
            sol_list = pd.Series(data = class_mastermodel.relaxedModel.getAttr('X'),index=[v.VarName for v in class_mastermodel.relaxedModel.getVars()])
            sol_idx = sol_list[sol_list>0.1].index
            
    else:
        sol_idx = sol_list #[solidx1, solidx2, solidx3,...]
    ans_list = []
    total_truck_no = 0
    total_drone_no = 0
    truck_serve_cus = pd.Series()
    drone_serve_cus = pd.Series()
    
    for s_idx in sol_idx: 
        ans = convertSolIndex(s_idx,class_mastermodel)
        ans_list.append(ans.index)
        total_truck_no+= class_mastermodel.getTruckNumberFromRoute(s_idx,True)
        total_drone_no+= class_mastermodel.getDroneNumberFromRoute(s_idx)
        t_cus = class_mastermodel.getNumberCustomerServedByTruck(s_idx)
        d_cus = class_mastermodel.getNumberCustomerServedByDrone(s_idx)
        print(t_cus,d_cus)
        truck_serve_cus = truck_serve_cus.append(t_cus)
        drone_serve_cus = drone_serve_cus.append(d_cus)
       
    if not ignore_fix_cost:  trucks_cost = total_truck_no*class_mastermodel.truck_fix_cost
    else: trucks_cost=0
        
    if relaxation: 
        obj_cost =  class_mastermodel.relaxedModel.ObjVal+trucks_cost
        print("Master Model OBJ:", obj_cost)
    else: 
        obj_cost =  class_mastermodel.model.ObjVal+trucks_cost
        print("Master Model OBJ:", obj_cost)
    if vis_plot: plot_solution(ans_list,node_trace)
    return obj_cost, total_truck_no, total_drone_no,truck_serve_cus,drone_serve_cus,ans_list

def plot_solution(sol_idx,node_trace,save_to_file=None,_title=None,_cus_dem=None):
    list_path = []
    for sol in sol_idx:
#         print(sol)
        list_path+=return_path4vis(sol,node_trace)
#     print(list_path)
    plot_network(node_trace,list_path,save_to_file,_title=_title,_cus_dem=_cus_dem)
    


def abbreviateCusName(_cus_name):
    if 'customer' in _cus_name:
        cus_no = _cus_name.split('_')[-1]
        return 'C_'+cus_no
    else:
        return _cus_name
def createNodeLabelWithDemand(node_label_text,cus_dem_df):
    '''node_label_text is node_trace['text']'''
    if cus_dem_df is not None:
#         cus_dem = cus_dem_df[cus_dem_df.index.str.contains('D')]
        cus_dem = cus_dem_df
        node_label = pd.Series(node_label_text)
#         node_label = node_label.apply(lambda x:abbreviateCusName(x)+ str(cus_dem[cus_dem.index.str.contains(x+'_')].values))
        node_label = node_label.apply(lambda x: abbreviateCusName(x)+"["+str(cus_dem[cus_dem.index.str.contains(x)].values[0])+"]" )
    else:
        node_label = pd.Series(node_label_text)
        node_label = node_label.apply(lambda x:abbreviateCusName(x))
    return node_label.values.tolist()

def getSymbolType(x):
    '''x is string name of nodes'''
    if 'customer' in x: return 'circle'
    elif 'dock' in x: return 'triangle-up'
    elif 'depot' in x: return 'diamond'
    
def createNodeSymbolList(node_label_text):
    n_text = pd.Series(node_label_text)
    symbol_text = n_text.apply(lambda x: getSymbolType(x)).to_list()
    return symbol_text


def random_distance_matrix(nodes):
    # random the nodes position
    no_nodes = len(nodes)
    truck_distance=dict()
    drone_distance=dict()
    nodes_position = dict()
    rand_seed = np.random.uniform(low=-1.5,high=0)
    for i in range(no_nodes):
        nodes_position[nodes[i]] = list(np.random.rand(2)*10)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i>j:
                point_i = np.array(nodes_position[nodes[i]])
                point_j = np.array(nodes_position[nodes[j]])
                t_dist_rand = np.linalg.norm(point_i-point_j)
                d_dist_rand =t_dist_rand+rand_seed
                if d_dist_rand<0.00001: d_dist_rand = t_dist_rand
                truck_distance[(nodes[i],nodes[j])] = t_dist_rand
                truck_distance[(nodes[j],nodes[i])] = t_dist_rand

                drone_distance[(nodes[i],nodes[j])] = d_dist_rand
                drone_distance[(nodes[j],nodes[i])] = d_dist_rand
    return truck_distance, drone_distance, nodes_position
# truck_distance, drone_distance, nodes_position=random_distance_matrix(nodes)