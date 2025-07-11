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

@author Pal Andras Papp, Raphael S. Steiner
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
import subprocess
import time

from SpectralTopologicalOrdering import lin_constraint, nonlin_constraint, spectral_acyclic_bi_partition, top_order_small_cut_fix
from convert_for_dagP import convert_dot_file

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

def metis_bi_partition(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list, list]:
    
    simpleGraph = nx.Graph( graph.to_undirected() )
    
    (edgecuts, split) = metis.part_graph(simpleGraph, nparts=2, ubvec=[imbalance], objtype='cut')
    
    parts = [list(), list()]
    nodes = list(graph.nodes)
    for ind, part in enumerate(split):
        parts[part].append(nodes[ind])

    return parts

def metis_with_acyclic_fix_with_stable_proportion(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list, list, float]:
    earlier, later = metis_bi_partition(graph, imbalance)
    e_set = set(earlier)
    l_set = set(later)
    
    edge_diff = 0
    for edge in graph.edges:
        if (edge[0] in e_set) and (edge[1] in l_set):
            edge_diff += 1
        if (edge[0] in l_set) and (edge[1] in e_set):
            edge_diff -= 1
    
    if (edge_diff < 0):
        earlier, later = later, earlier
    
    nx.set_node_attributes(graph, "", "part")
    
    return top_order_small_cut_fix(graph, earlier, later)

def metis_with_acyclic_fix(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list, list]:
    return metis_with_acyclic_fix_with_stable_proportion(graph, imbalance)[:2]



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
        
        result = scipy.optimize.minimize(opt_func, x0, method='trust-constr', jac=opt_jac, hess=opt_hess, constraints=[lin_constraint(graph.number_of_nodes()), nonlin_constraint(p_iter)], options={'verbose': 0})
        
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

def spectral_split_classic_acyclic_fix_with_stable_proportion(graph: nx.MultiDiGraph, lq: float = 2.0, lp: float = 2.0) -> list[list, list, float]:
    earlier, later = spectral_split_classic(graph, lq, lp)
    e_set = set(earlier)
    l_set = set(later)
    
    edge_diff = 0
    for edge in graph.edges:
        if (edge[0] in e_set) and (edge[1] in l_set):
            edge_diff += 1
        if (edge[0] in l_set) and (edge[1] in e_set):
            edge_diff -= 1
    
    if (edge_diff < 0):
        earlier, later = later, earlier
    
    nx.set_node_attributes(graph, "", "part")
    
    return top_order_small_cut_fix(graph, earlier, later)

def spectral_split_classic_acyclic_fix(graph: nx.MultiDiGraph, lq: float = 2.0, lp: float = 2.0) -> list[list, list]:
    return spectral_split_classic_acyclic_fix_with_stable_proportion(graph, lq, lp)[:2]

def directed_fiduccia_mattheyses(graph: nx.MultiDiGraph, earlier: list[None], later: list[None], max_in_part: int, must_be_acyclic: bool = True) -> list[list[None], list[None]]:
    vertices = []
    vertices.extend(earlier)
    vertices.extend(later)

    top_ord = []
    induced_graph = nx.induced_subgraph(graph, vertices)

    ind_dict = dict()
    for ind, vert in enumerate(vertices):
        ind_dict[vert] = ind

    max_degree = 0
    for v in vertices:
        if induced_graph.in_degree(v) + induced_graph.out_degree(v) > max_degree:
            max_degree = induced_graph.in_degree(v) + induced_graph.out_degree(v)
    
    partition = (earlier, later)
    nr_passes = 5

    for pass_nr in range(nr_passes):
        # print(f"Pass {pass_nr} begins.")

        # init
        vertex_to_part = [0 for v in vertices]
        locked = [False for v in vertices]
        obstacles_to_move = [0 for v in vertices]
        gain = [0 for v in vertices]
        cost = 0
        for v in partition[0]:
            vertex_to_part[ind_dict[v]] = 0
        for v in partition[1]:
            vertex_to_part[ind_dict[v]] = 1
        for v in partition[0]:
            for edge in induced_graph.out_edges(v):
                if vertex_to_part[ind_dict[edge[1]]] == 0:
                    obstacles_to_move[ind_dict[v]] += 1
                    gain[ind_dict[v]] -= 1
                else:
                    cost +=1
                    gain[ind_dict[v]] += 1
            for edge in induced_graph.in_edges(v):
                if vertex_to_part[ind_dict[edge[0]]] == 0:
                    gain[ind_dict[v]] -= 1
                elif must_be_acyclic:
                    # partition is incorrect
                    if pass_nr == 0:
                        print("ERROR: The partitioning used to initialize FM is not acyclic.")
                    else:
                        print("ERROR during FM: the created partition is not acyclic.")
                    return partition
                else:
                    gain[ind_dict[v]] += 1

        for v in partition[1]:
            for edge in induced_graph.in_edges(v):
                if vertex_to_part[ind_dict[edge[0]]] == 1:
                    obstacles_to_move[ind_dict[v]] += 1
                    gain[ind_dict[v]] -= 1
                else:
                    gain[ind_dict[v]] += 1
            for edge in induced_graph.out_edges(v):
                if vertex_to_part[ind_dict[edge[1]]] == 1:
                    gain[ind_dict[v]] -= 1
                elif not must_be_acyclic:
                    gain[ind_dict[v]] += 1
                    cost +=1

        gain_bucket_array = [[[] for i in range(2*max_degree+1)] for part in range(2)] 
        max_gain = [-(max_degree+1) for part in range(2)]
        for v in vertices:
            if obstacles_to_move[ind_dict[v]] == 0 or not must_be_acyclic:
                gain_bucket_array[vertex_to_part[ind_dict[v]]][gain[ind_dict[v]] + max_degree].append(v)
                if gain[ind_dict[v]] > max_gain[vertex_to_part[ind_dict[v]]]:
                    max_gain[vertex_to_part[ind_dict[v]]] = gain[ind_dict[v]]

        best_index = 0
        best_cost = cost
        moved_nodes = []
        left_size = len(partition[0])

        # moves
        while len(moved_nodes) < len(vertices):

            gain_idx = max(max_gain[0], max_gain[1]) + max_degree
            to_move = -1
            while gain_idx >= 0:
                choose_left = len(gain_bucket_array[0][gain_idx])>0 and len(vertices) - left_size < max_in_part
                choose_right = len(gain_bucket_array[1][gain_idx])>0 and left_size < max_in_part
                if choose_left and choose_right:
                    if left_size >= len(vertices) - left_size:
                        choose_left = False
                    else:
                        choose_right = False

                if choose_left:
                    to_move = gain_bucket_array[0][gain_idx].pop(len(gain_bucket_array[0][gain_idx])-1)
                    break
                elif choose_right:
                    to_move = gain_bucket_array[1][gain_idx].pop(len(gain_bucket_array[1][gain_idx])-1)
                    break
                gain_idx -= 1

            if to_move == -1:
                break
            
            moved_nodes.append(to_move)
            cost -= gain[ind_dict[to_move]]
            if cost < best_cost:
                best_index = len(moved_nodes)
                best_cost = cost
            locked[ind_dict[to_move]] = True
            vertex_to_part[ind_dict[to_move]] = 1 - vertex_to_part[ind_dict[to_move]]

            if choose_right:
                left_size += 1
                for edge in induced_graph.in_edges(to_move):
                    source  = edge[0]
                    if not locked[ind_dict[source]] and (obstacles_to_move[ind_dict[source]] == 0 or not must_be_acyclic):
                        gain_bucket_array[vertex_to_part[ind_dict[source]]][gain[ind_dict[source]] + max_degree].remove(source)
                    obstacles_to_move[ind_dict[source]] += 1
                    if vertex_to_part[ind_dict[source]] == 0:
                        gain[ind_dict[source]] -= 2
                    else:
                        gain[ind_dict[source]] += 2
                    if not locked[ind_dict[source]] and not must_be_acyclic:
                        gain_bucket_array[vertex_to_part[ind_dict[source]]][gain[ind_dict[source]] + max_degree].append(source)
                        if gain[ind_dict[source]] > max_gain[vertex_to_part[ind_dict[source]]]:
                            max_gain[vertex_to_part[ind_dict[source]]] = gain[ind_dict[source]] 
                for edge in induced_graph.out_edges(to_move):
                    target = edge[1]
                    if not locked[ind_dict[target]] and not must_be_acyclic:
                        gain_bucket_array[vertex_to_part[ind_dict[target]]][gain[ind_dict[target]] + max_degree].remove(target)
                    obstacles_to_move[ind_dict[target]] -= 1
                    if vertex_to_part[ind_dict[target]] == 1:
                        gain[ind_dict[target]] += 2
                    else:
                        gain[ind_dict[target]] -= 2
                    if not locked[ind_dict[target]] and (obstacles_to_move[ind_dict[target]] == 0 or not must_be_acyclic):
                        gain_bucket_array[vertex_to_part[ind_dict[target]]][gain[ind_dict[target]] + max_degree].append(target)
                        if gain[ind_dict[target]] > max_gain[vertex_to_part[ind_dict[target]]]:
                            max_gain[vertex_to_part[ind_dict[target]]] = gain[ind_dict[target]]
            else:
                left_size -= 1
                for edge in induced_graph.out_edges(to_move):
                    target = edge[1]
                    if not locked[ind_dict[target]] and (obstacles_to_move[ind_dict[target]] == 0 or not must_be_acyclic):
                        gain_bucket_array[vertex_to_part[ind_dict[target]]][gain[ind_dict[target]] + max_degree].remove(target)
                    obstacles_to_move[ind_dict[target]] += 1
                    if vertex_to_part[ind_dict[target]] == 1:
                        gain[ind_dict[target]] -= 2
                    else:
                        gain[ind_dict[target]] += 2
                    if not locked[ind_dict[target]] and not must_be_acyclic:
                        gain_bucket_array[vertex_to_part[ind_dict[target]]][gain[ind_dict[target]] + max_degree].append(target)
                        if gain[ind_dict[target]] > max_gain[vertex_to_part[ind_dict[target]]]:
                            max_gain[vertex_to_part[ind_dict[target]]] = gain[ind_dict[target]] 
                for edge in induced_graph.in_edges(to_move):
                    source = edge[0]
                    if not locked[ind_dict[source]] and not must_be_acyclic:
                        gain_bucket_array[vertex_to_part[ind_dict[source]]][gain[ind_dict[source]] + max_degree].remove(source)
                    obstacles_to_move[ind_dict[source]] -= 1
                    if vertex_to_part[ind_dict[source]] == 0:
                        gain[ind_dict[source]] += 2
                    else:
                        gain[ind_dict[source]] -= 2
                    if not locked[ind_dict[source]] and (obstacles_to_move[ind_dict[source]] == 0 or not must_be_acyclic):
                        gain_bucket_array[vertex_to_part[ind_dict[source]]][gain[ind_dict[source]] + max_degree].append(source)
                        if gain[ind_dict[source]] > max_gain[vertex_to_part[ind_dict[source]]]:
                            max_gain[vertex_to_part[ind_dict[source]]] = gain[ind_dict[source]]


        # select best
        if best_index == 0:
            break

        best_partition = ([], [])
        moved = [False for v in vertices]

        for i in range(best_index):
            moved[ind_dict[moved_nodes[i]]] = True
        
        for v in partition[0]:
            if moved[ind_dict[v]]:
                best_partition[1].append(v)
            else:
                best_partition[0].append(v)
        for v in partition[1]:
            if moved[ind_dict[v]]:
                best_partition[0].append(v)
            else:
                best_partition[1].append(v)
            
        partition = best_partition

    #print(f"Size of formed partition: {len(partition[0])}, {len(partition[1])} (cf. limit: {max_in_part})")

    return partition

def Undirected_FM_from_scratch(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list[None],list[None]]:
    assert(imbalance > 1.0)
    
    vertex_list = list(graph.nodes)
    
    if len(vertex_list) <= 1:
        return [vertex_list, []]
    
    num_e = (len(vertex_list) + 1) // 2
    earlier = vertex_list[:num_e]
    later = vertex_list[num_e:]

    return directed_fiduccia_mattheyses(graph, earlier, later, int(num_e * imbalance), False)
    
    

def FM_split_from_scratch(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list[None],list[None]]:
    assert(imbalance > 1.0)
    assert(nx.is_directed_acyclic_graph(graph))

    vertex_list = list(nx.topological_sort(graph))
    
    if len(vertex_list) <= 1:
        return [vertex_list, []]
    
    num_e = (len(vertex_list) + 1) // 2
    earlier = vertex_list[:num_e]
    later = vertex_list[num_e:]

    return directed_fiduccia_mattheyses(graph, earlier, later, int(num_e * imbalance), True)

def FM_split_improving_spectral(graph: nx.MultiDiGraph, imbalance: float = 1.3) -> list[list[None],list[None]]:
    assert(imbalance > 1.0)
    assert(nx.is_directed_acyclic_graph(graph))

    vertex_list = list(graph.nodes) 
    
    if len(vertex_list) <= 1:
        return [vertex_list, []]

    (earlier, later) = spectral_acyclic_bi_partition(graph, 2.0, 0.5)
    weight_limit = max(int(((len(vertex_list) + 1) // 2) * imbalance), len(earlier), len(later))
    return directed_fiduccia_mattheyses(graph, earlier, later, weight_limit, True)


# Wrapper to run rMLGP
def rmlgp_partition(graph: nx.MultiDiGraph, binary_path: str) -> list[list]:
    if graph.number_of_nodes() < 3:
        small_top_ord = list(nx.topological_sort(graph))
        return [small_top_ord[:1], small_top_ord[1:]]
    if graph.number_of_edges() == 0:
        n_nodes_halv = graph.number_of_nodes() // 2
        small_top_ord = list(nx.topological_sort(graph))
        return [small_top_ord[:n_nodes_halv], small_top_ord[n_nodes_halv:]]
    
    sleep_time = 0.2 # seconds
    
    temp_file_name = "temp_graph.dot"
    temp_graph_path = os.path.abspath("./" + temp_file_name)
    nx.drawing.nx_agraph.write_dot(graph, temp_graph_path)
    time.sleep(sleep_time)
    
    # graph_path = os.path.abspath(graph_file)
    graph_folder = os.path.dirname(temp_graph_path)
    
    # Convert .dot file for rMLGP
    converted_file = os.path.join(graph_folder, "_rmlgp.dot")
    convert_dot_file(temp_graph_path, converted_file)
    time.sleep(sleep_time)

    # Setting for rMLGP
    cmd = [
        binary_path,
        converted_file,
        "2",
        "--print", "0",
        "--ratio", "1.3",
        "--write_parts=1"
    ]
    subprocess.run(cmd, check=True)
    time.sleep(sleep_time)

    # Find the output .txt file generated by rMLGP for this converted file
    find_txt = os.path.basename(converted_file)
    matching_txt_files = [
        f for f in os.listdir(graph_folder)
        if f.startswith(find_txt) and f.endswith(".txt")
    ]

    # Assign it to the original variable name
    part_file = os.path.join(graph_folder, matching_txt_files[0])

    # Read bi-partitions
    with open(part_file, "r") as f:
        assignments = [int(line.strip()) for line in f if line.strip() != ""]

    # Sort node list to match assignment order
    # graph = nx.nx_pydot.read_dot(graph_file)
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    part_0 = [nodes[i] for i, val in enumerate(assignments) if val == 0]
    part_1 = [nodes[i] for i, val in enumerate(assignments) if val == 1]

    # Save/remove the files _rmlgp has generated
    save_res = False
    if save_res != True:
        for filename in os.listdir(graph_folder):
            if filename.startswith("_rmlgp.dot") or filename.startswith(temp_file_name):
                os.remove(os.path.join(graph_folder, filename))
                
    time.sleep(sleep_time)

    return [part_0, part_1]


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