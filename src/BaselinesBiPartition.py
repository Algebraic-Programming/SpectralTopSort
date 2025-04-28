'''
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Raphael S. Steiner
'''

import functools
import math
import metis
import networkx as nx
import numpy as np
import pydot
import os
import scipy
import sys

from SpectralTopologicalOrdering import lin_constraint, nonlin_constraint

def is_valid_bi_partition(graph: nx.MultiDiGraph, parts: list[list, list]) -> bool:
    if not len(parts) == 2:
        return False
    
    part_set_0 = set(parts[0])
    part_set_1 = set(parts[1])
    
    if not graph.number_of_nodes() == len(part_set_0) + len(part_set_1):
        return False
    
    for vert in graph.nodes:
        if (vert in part_set_0) and (vert in part_set_1):
            return False
        if (vert not in part_set_0) and (vert not in part_set_1):
            return False
        
    return True

def nx_graph_from_matrix(mat: list[list]) -> nx.MultiDiGraph:
    assert(len(mat) == len(mat[0]))
    mat_size = len(mat)
    
    graph = nx.MultiDiGraph()
    
    for i in range(mat_size):
        graph.add_node(i)
        
    for i in range(mat_size):
        for j in range(mat_size):
            if (abs(mat[i][j]) > 0.0):
                graph.add_edge(i,j)
    
    return graph

def metis_bi_partition(graph: nx.MultiDiGraph) -> list[list, list]:
    (edgecuts, split) = metis.part_graph(graph, 2)
    
    parts = [list(), list()]
    nodes = list(graph.nodes)
    for ind, part in enumerate(split):
        parts[part].append(nodes[ind])

    return parts



def homogenous_quadratic_form(x: np.ndarray, graph: nx.MultiDiGraph, lq: float = 2.0):
    assert(x.size == graph.number_of_nodes())
    
    outvalue = 0
    vertex_list = list(graph.nodes)
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            outvalue += (abs(x[ind_dict[src]] - x[i]))**(lq)

    return outvalue

def homogenous_quadratic_form_jac(x: np.ndarray, graph: nx.MultiDiGraph, lq: float = 2.0):
    assert(x.size == graph.number_of_nodes())
    
    vertex_list = list(graph.nodes)
    
    outvalue = np.zeros_like(x)
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            outvalue[i] += lq * math.copysign((abs(x[i] - x[ind_dict[src]]))**(lq - 1.0), x[i] - x[ind_dict[src]]) 
    
    for i in range(x.size):
        vert = vertex_list[i]
        for out_edge in graph.out_edges(vert):
            tgt = out_edge[1]
            outvalue[i] += lq * math.copysign((abs(x[i] - x[ind_dict[tgt]]))**(lq - 1.0), x[i] - x[ind_dict[tgt]])
                
        
    return outvalue


def homogenous_quadratic_form_hess(x: np.ndarray, graph: nx.MultiDiGraph, lq: float = 2.0):
    assert(x.size == graph.number_of_nodes())
    
    vertex_list = list(graph.nodes)
    
    outvalue = np.zeros((x.size, x.size))
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            outvalue[i][i] += lq * (lq - 1.0) * (abs(x[ind_dict[src]] - x[i]))**(lq - 2.0)
            outvalue[i][ind_dict[src]] -= lq * (lq - 1.0) * (abs(x[ind_dict[src]] - x[i]))**(lq - 2.0)
            outvalue[ind_dict[src]][i] -= lq * (lq - 1.0) * (abs(x[ind_dict[src]] - x[i]))**(lq - 2.0)
    
    for i in range(x.size):
        vert = vertex_list[i]
        for out_edge in graph.out_edges(vert):
            tgt = out_edge[1]
            outvalue[i][i] += lq * (lq - 1.0) * (abs(x[i] - x[ind_dict[tgt]]))**(lq - 2.0)
    
    return outvalue



def spectral_split_classic(graph: nx.MultiDiGraph, lq: float = 2.0, lp: float = 2.0) -> list[list[None],list[None]]:
    assert(lq > 1.0)
    assert(lp > 1.0)    
    
    x0 = np.random.normal(loc=0.0, scale=(10.0)**(-8.0), size=graph.number_of_nodes())
    result = np.zeros_like(x0)
    
    p_iter = 2.0
    q_iter = 2.0
    
    while True:
        x0 = x0 / np.power(np.sum(np.power(np.absolute(x0), p_iter)), 1.0 / p_iter) * np.power(np.size(x0), 1.0 / p_iter)
        
        opt_func = functools.partial(homogenous_quadratic_form, graph=graph, lq=q_iter)
        opt_jac = functools.partial(homogenous_quadratic_form_jac, graph=graph, lq=q_iter)
        opt_hess = functools.partial(homogenous_quadratic_form_hess, graph=graph, lq=q_iter)
        
        result = scipy.optimize.minimize(opt_func, x0, method='trust-constr', jac=opt_jac, hess=opt_hess, constraints=[lin_constraint(graph.number_of_nodes()), nonlin_constraint(p_iter)], options={'verbose': 1})
        
        if (p_iter > lp or q_iter > lq):
            p_iter = max(lp, 1 + (min(p_iter, 1.95) - 1.0)**(1.1))
            q_iter = max(lq, 1 + (min(q_iter, 1.95) - 1.0)**(1.1))
            x0 = result.x
        else:
            break
        
    earlier = []
    later = []
    
    vertex_list = list(graph.nodes)
    
    for ind, val in enumerate(result.x):
        if val > 0:
            earlier.append(vertex_list[ind])
        else:
            later.append(vertex_list[ind])
    
    return [earlier, later]


def main():
    if (len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <graph.dot>")
        print("Usage: " + sys.argv[0] + " <graph.mtx>")
        print("\nIf the flag \"--low\" is passed then only the strictly lower triangular part is taken.")
        return 1
        
    graph_file = sys.argv[1]
    graph_name = graph_file[graph_file.rfind("/") + 1: graph_file.rfind(".")]
    
    graph = None
    if (graph_file[-3:] == "dot"):
        graph = nx.nx_pydot.read_dot(graph_file)
        
    elif (graph_file[-3:] == "mtx"):
        matrix = scipy.io.mmread(graph_file, spmatrix=True)
        if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
            matrix = matrix.transpose()
            matrix = scipy.sparse.triu(matrix, k=1)
        graph = nx_graph_from_matrix(matrix.toarray())
        
    else:
        print("Unknown file format!")
        return 1
    
    parts = metis_bi_partition(graph)
    print(parts)
    assert(is_valid_bi_partition(graph, parts))
    
    parts = spectral_split_classic(graph)
    print(parts)
    assert(is_valid_bi_partition(graph, parts))

    return 0

if __name__ == "__main__":
    main()