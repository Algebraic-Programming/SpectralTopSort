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

@author Pal Andras Papp, Dimosthenis Pasadakis, Raphael S. Steiner
'''

import functools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pydot
import os
import scipy
import seaborn as sns
import sys

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix, check_valid_top_order, spec_top_order_whole, spec_top_order_whole_with_spec_values
from BaselinesTopOrder import bfs_topOrd, dfs_topOrd, earliest_parent_topOrd, access_pattern_max_topOrd, access_pattern_avg_topOrd, access_pattern_sum_topOrd, sum_edge_length_parent_topOrd, max_sibling_score_in_window, cuthill_Mckee, recursive_acyclic_bisection
from BaselinesBiPartition import rmlgp_partition

def compute_edge_lengths(graph: nx.MultiDiGraph, topOrder: list) -> list:
    topOrdInd = dict()
    for ind, vert in enumerate(topOrder):
        topOrdInd[vert] = ind
    
    edge_lengths = []
    for src, tgt in graph.edges():
        assert(topOrdInd[tgt] > topOrdInd[src])
        edge_lengths.append( topOrdInd[tgt] - topOrdInd[src] )
    
    return edge_lengths

def compute_stack_distances(graph: nx.MultiDiGraph, topOrder: list) -> list:
    topOrdInd = dict()
    for ind, vert in enumerate(topOrder):
        topOrdInd[vert] = ind
    
    access_sequence = []
    for vert in topOrder:
        parents = []
        for par, _ in graph.in_edges(vert):
            parents.append((topOrdInd[par], par))

        parents.sort()                              # Sorting parents as they are in the topological order
        for _, par in parents:
            access_sequence.append(par)
        
        access_sequence.append(vert)
    
    stack_distances = []
    stack = []
    in_stack = set()
    for vert in access_sequence:
        if vert in in_stack:
            dist = stack[::-1].index(vert)          # Distance from the back
            stack_distances.append(dist)
            stack.remove(vert)                      # Removes unique vert in stack to avoid counting duplicates
            stack.append(vert)
        else:
            stack.append(vert)
            in_stack.add(vert)
    
    
    return stack_distances

def compute_cut_widths(graph: nx.MultiDiGraph, topOrder: list) -> list:
    cut_widths = []
    
    cuts = 0
    for i in range(max(0,len(topOrder)-1)):
        vert = topOrder[i]
        cuts += graph.out_degree(vert)
        cuts -= graph.in_degree(vert)
        
        cut_widths.append(cuts)
    
    return cut_widths





def combine_top_order_weakly_connected_components(algorithm, graph: nx.MultiDiGraph):
    weak_comp = nx.weakly_connected_components(graph)
    
    top_order = []
    
    for comp in weak_comp:
        subgraph = nx.induced_subgraph(graph, comp)
        subgraph = subgraph.copy()
        top_order.extend( algorithm(subgraph) )
        
    return top_order






def main():
    if (len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <graph directory>")
        print("Usage: " + sys.argv[0] + " <graph.dot>")
        print("Usage: " + sys.argv[0] + " <graph.mtx>" + " (optional --low)")
        print("\nOnly the strictly upper triangular part of an mtx matrix is taken. If the flag \"--low\" is passed then only the strictly lower triangular part is taken.")
        return 1
    
    graph_dict = dict()
    graph_files = []
    
    name_for_stats_file = ''
    
    isDir = None
    if os.path.isfile(sys.argv[1]):
        isDir = False
        name_for_stats_file = os.path.basename(sys.argv[1])
        name_for_stats_file = name_for_stats_file.split('.')[-2]
    if os.path.isdir(sys.argv[1]):
        isDir = True
        name_for_stats_file = os.path.basename(os.path.normpath(sys.argv[1]))

    
    if not isDir:
        graph_files.append(sys.argv[1])
    else:
        dir = sys.argv[1]
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            graph_files.append(file_path)
        
    for graph_file in graph_files:
        graph_name = graph_file[graph_file.rfind("/") + 1: graph_file.rfind(".")]
        print(f"Reading {graph_name}.")
        
        graph = None
        if (graph_file[-3:] == "dot"):
            graph = nx.nx_pydot.read_dot(graph_file)
            if (not nx.is_directed_acyclic_graph(graph)):
                print("Graph is not acyclic" + " (" + graph_file + ")")
                continue
            
        elif (graph_file[-3:] == "mtx"):
            matrix = scipy.io.mmread(graph_file)
            if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
                matrix = matrix.transpose()
            if (len(sys.argv) == 3) and (sys.argv[2] == "--acyc"):
                upper_matrix = scipy.sparse.triu(matrix, k=1)
                top_graph = nx_graph_from_upper_triangular_matrix(upper_matrix.toarray())
                
                lower_matrix = scipy.sparse.tril(matrix, k=-1)
                lower_matrix = lower_matrix.transpose()
                low_graph = nx_graph_from_upper_triangular_matrix(lower_matrix.toarray())
                
                if top_graph.number_of_edges() >= low_graph.number_of_edges():
                    print("Taking top part of matrix")
                    matrix = upper_matrix
                else:
                    print("Taking bottom part of matrix")
                    matrix = lower_matrix
                    
            matrix = scipy.sparse.triu(matrix, k=1)
            graph = nx_graph_from_upper_triangular_matrix(matrix.toarray())
            
        else:
            print("Unknown file format!" + " (" + graph_file + ")")
            continue
        
        graph_dict[graph_name] = graph
    
    algorithms_to_run = {
        # "Original": lambda G: sorted([v for v in G.nodes], key=int),
        "Spectral_2.0": functools.partial(spec_top_order_whole, lp=2.0, const_dir=0.5),
        # "Spectral_1.8": functools.partial(spec_top_order_whole, lp=1.8),
        # "Spectral_1.5": functools.partial(spec_top_order_whole, lp=1.5),
        # "Spectral_1.2": functools.partial(spec_top_order_whole, lp=1.2),
        # "Spectral_spec_2.0": functools.partial(spec_top_order_whole_with_spec_values, lp=2.0),
        # "Spectral_spec_1.8": functools.partial(spec_top_order_whole_with_spec_values, lp=1.8),
        # "Spectral_spec_1.5": functools.partial(spec_top_order_whole_with_spec_values, lp=1.5),
        # "Spectral_spec_1.2": functools.partial(spec_top_order_whole_with_spec_values, lp=1.2),
        "Spectral_classic_2.0": functools.partial(spec_top_order_whole, lp=2.0, const_dir=0.0),
        "BFS": bfs_topOrd,
        # "DFS": dfs_topOrd,
        # "Earliest_Parent": earliest_parent_topOrd,
        # "Edge_Length_Sum": sum_edge_length_parent_topOrd,
        # "Access_Pattern_Max": access_pattern_max_topOrd,
        # "Access_Pattern_Sum": access_pattern_sum_topOrd,
        # "Access_Pattern_Avg": access_pattern_avg_topOrd,
        "Gorder": functools.partial(max_sibling_score_in_window, window_size=5),
        "Cuthill–Mckee": cuthill_Mckee,
        # "rMLGP_recursive": functools.partial(recursive_acyclic_bisection, acyc_bisec_method=functools.partial(rmlgp_partition, binary_path="./rMLGP")), 
    }
    
    edge_length_metric_name = "Edge length"
    stack_reuse_metric_name = "Reuse distance"
    cut_width_metric_name = "Edge cut"
    
    df_list_dict = []
    
    for graph_name, graph in graph_dict.items():
        print("Graph: " + graph_name + " Vertices: "+ str(graph.number_of_nodes()) + " Edges: " + str(graph.number_of_edges()))
    
        for alg_name, alg_func in algorithms_to_run.items():
            try:
                graph_copy = graph.copy()
                top_order = combine_top_order_weakly_connected_components(alg_func, graph_copy)
                if (not check_valid_top_order(graph, top_order)):
                    print("Invalid Topological order generated by " + alg_name + " during graph " + graph_name + "!")
                    continue
                
                edge_lengths = compute_edge_lengths(graph, top_order)
                for dist in edge_lengths:
                    df_list_dict.append({
                        "Algorithm": alg_name,
                        "Metric": edge_length_metric_name,
                        "Distance": dist,
                        "Graph": graph_name
                    })
                
                stack_distances = compute_stack_distances(graph, top_order)
                for dist in stack_distances:
                    df_list_dict.append({
                        "Algorithm": alg_name,
                        "Metric": stack_reuse_metric_name,
                        "Distance": dist,
                        "Graph": graph_name
                    })
                    
                cut_widths = compute_cut_widths(graph, top_order)
                for dist in cut_widths:
                    df_list_dict.append({
                        "Algorithm": alg_name,
                        "Metric": cut_width_metric_name,
                        "Distance": dist,
                        "Graph": graph_name
                    })
                    
                plot_graphs_after_reordering = False
                
                if plot_graphs_after_reordering:
                    plt.figure(graph_name + " --- " + alg_name)
                    
                    for vert in graph_copy.nodes:
                        if "part" in graph_copy.nodes[vert].keys():
                            del graph_copy.nodes[vert]["part"]
                    for edge in graph_copy.edges:
                        if "weight" in graph_copy.edges[edge].keys():
                            del graph_copy.edges[edge]["weight"]
                    
                    
                    plt.spy(nx.to_numpy_array(graph_copy, top_order), marker=".", markersize=4)
                    # plt.savefig(graph_name + "_" + alg_name + ".eps")
                    
            except:
                print("Error during graph " + graph_name + " and algorithm " + alg_name + ".")
        
    df = pd.DataFrame(df_list_dict, columns=["Algorithm", "Metric", "Distance", "Graph"])
    
    
    for graph_n, group in df.groupby("Graph"):
        if not isDir:
            plt.figure("Edge Length Distribution Graph: "+ graph_n)
            sns.violinplot(x="Algorithm", y="Distance", inner="quart", data=group[ group["Metric"] == edge_length_metric_name ], cut=0)
            plt.xticks(rotation=90)
        
            plt.figure("Stack Reuse Distance Distribution Graph: "+ graph_n)
            sns.violinplot(x="Algorithm", y="Distance", inner="quart", data=group[ group["Metric"] == stack_reuse_metric_name ], cut=0)
            plt.xticks(rotation=90)
        
            plt.figure("Cut Width Distribution Graph: "+ graph_n)
            sns.violinplot(x="Algorithm", y="Distance", inner="quart", data=group[ group["Metric"] == cut_width_metric_name ], cut=0)
            plt.xticks(rotation=90)
    
        print("\nGraph: " + graph_n)
        print(group.groupby(["Metric", "Algorithm"]).describe().to_string())
    
    if not isDir:
        plt.show()
        
    write_data = False
    if (write_data):
        df.to_csv("TopOrder_" + name_for_stats_file + ".csv")

    return 0

if __name__ == "__main__":
    main()
