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

@author Dimosthenis Pasadakis, Raphael S. Steiner
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

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix, check_valid_top_order, spec_top_order_whole
from BaselinesTopOrder import bfs_topOrd, dfs_topOrd, earliest_parent_topOrd, access_pattern_max_topOrd, access_pattern_avg_topOrd, access_pattern_sum_topOrd, sum_edge_length_parent_topOrd, max_sibling_score_in_window

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















def main():
    if (len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <graph.dot>")
        print("Usage: " + sys.argv[0] + " <graph.mtx>" + " (optional --low)")
        print("\nOnly the strictly upper triangular part of an mtx matrix is taken. If the flag \"--low\" is passed then only the strictly lower triangular part is taken.")
        return 1
        
    graph_file = sys.argv[1]
    graph_name = graph_file[graph_file.rfind("/") + 1: graph_file.rfind(".")]
    
    graph = None
    if (graph_file[-3:] == "dot"):
        graph = nx.nx_pydot.read_dot(graph_file)
        if (not nx.is_directed_acyclic_graph(graph)):
            print("Graph is not acyclic")
            return 1
        
    elif (graph_file[-3:] == "mtx"):
        matrix = scipy.io.mmread(graph_file, spmatrix=True)
        if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
            matrix = matrix.transpose()
        matrix = scipy.sparse.triu(matrix, k=1)
        graph = nx_graph_from_upper_triangular_matrix(matrix.toarray())
        
    else:
        print("Unknown file format!")
        return 1
    
    
    
    algorithms_to_run = {
        "Original": lambda G: sorted([v for v in G.nodes], key=int),
        "Spectal_2.0": functools.partial(spec_top_order_whole, lp=2.0),
        # "Spectal_1.5": functools.partial(spec_top_order_whole, lp=1.5),
        # "Spectal_1.2": functools.partial(spec_top_order_whole, lp=1.2),
        "BFS": bfs_topOrd,
        "DFS": dfs_topOrd,
        "Earliest_Parent": earliest_parent_topOrd,
        "Edge_Length_Sum": sum_edge_length_parent_topOrd,
        "Access_Pattern_Max": access_pattern_max_topOrd,
        "Access_Pattern_Sum": access_pattern_sum_topOrd,
        "Access_Pattern_Avg": access_pattern_avg_topOrd,
        "Max_Windowed_Sibling": max_sibling_score_in_window
    }
    
    edge_length_metric_name = "Edge Length"
    stack_reuse_metric_name = "Stack Reuse Distance"
    
    df_list_dict = []
    
    for alg_name, alg_func in algorithms_to_run.items():
        top_order = alg_func(graph)
        if (not check_valid_top_order(graph, top_order)):
            print("Invalid Topological order generated by " + alg_name + "!")
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
        
    df = pd.DataFrame(df_list_dict, columns=["Algorithm", "Metric", "Distance", "Graph"])
    
    plt.figure("Edge Length Distribution Graph: "+ graph_name)
    sns.violinplot(x="Algorithm", y="Distance", inner="quart", data=df[ df["Metric"] == edge_length_metric_name ], cut=0)
    plt.xticks(rotation=90)
    
    for graph_n, group in df.groupby("Graph"):
        plt.figure("Stack Reuse Distance Distribution Graph: "+ graph_n)
        sns.violinplot(x="Algorithm", y="Distance", inner="quart", data=group[ group["Metric"] == stack_reuse_metric_name ], cut=0)
        plt.xticks(rotation=90)
    
        print("\nGraph: " + graph_n)
        print(group.groupby(["Metric", "Algorithm"]).describe().to_string())
    
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()
