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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pydot
import os
import scipy
import seaborn as sns
import sys

from SpectralTopologicalOrdering import spectral_split, spectral_acyclic_bi_partition
from BaselinesBiPartition import is_valid_bi_partition, spectral_split_classic, metis_bi_partition, nx_graph_from_matrix, FM_split_from_scratch, FM_split_improving_spectral

def cut_edges_ratio(graph: nx.MultiDiGraph, parts: list[list, list]) -> float:
    if graph.number_of_edges() == 0:
        return 0
    
    part_set_0 = set(parts[0])
    part_set_1 = set(parts[1])
    
    cut_edges = 0
    
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        if u in part_set_0 and v in part_set_1:
            cut_edges += 1
        if u in part_set_1 and v in part_set_0:
            cut_edges += 1
    
    return cut_edges / graph.number_of_edges()

def partition_imbalance(graph: nx.MultiDiGraph, parts: list[list, list]) -> float:
    bigger = max(len(parts[0]), len(parts[1]))
    smaller = min(len(parts[0]), len(parts[1]))
    
    return (bigger - smaller) / graph.number_of_nodes()

def misaligned_cut_edges_ratio(graph: nx.MultiDiGraph, parts: list[list, list]) -> float:
    part_set_0 = set(parts[0])
    part_set_1 = set(parts[1])
    
    cut_edges_forwards = 0
    cut_edges_backwards = 0
    
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        if u in part_set_0 and v in part_set_1:
            cut_edges_forwards += 1
        if u in part_set_1 and v in part_set_0:
            cut_edges_backwards += 1
            
    if cut_edges_backwards + cut_edges_forwards == 0:
        return 0
    else:
        return min(cut_edges_forwards, cut_edges_backwards) / (cut_edges_forwards + cut_edges_backwards)
    




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
        matrix = scipy.io.mmread(graph_file)
        if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
            matrix = matrix.transpose()
            matrix = scipy.sparse.triu(matrix, k=1)
        graph = nx_graph_from_matrix(matrix.toarray())
        
    else:
        print("Unknown file format!")
        return 1
    
    
    # Key: Algorithm Name, Value: (Function, Requires Acyclic)
    algorithms_to_run = {
        "Spectal_directional_2.0": (functools.partial(spectral_split, lp=2.0), False),
        # "Spectal_directional_1.5": (functools.partial(spectral_split, lp=1.5), False),
        # "Spectal_directional_1.1": (functools.partial(spectral_split, lp=1.1), False),
        "Spectal_acyclic_2.0": (functools.partial(spectral_acyclic_bi_partition, lp=2.0), True),
        # "Spectal_acyclic_1.5": (functools.partial(spectral_acyclic_bi_partition, lp=1.5), True),
        # "Spectal_acyclic_1.1": (functools.partial(spectral_acyclic_bi_partition, lp=1.1), True),
        "Spectal_classic_2.0": (functools.partial(spectral_split_classic, lp=2.0), False),
        # "Spectal_classic_1.5": (functools.partial(spectral_split_classic, lp=1.5), False),
        # "Spectal_classic_1.1": (functools.partial(spectral_split_classic, lp=1.1), False),
        "FM_from_scratch": (functools.partial(FM_split_from_scratch, imbalance=1.3), True),
        "FM_after_spectral": (functools.partial(FM_split_improving_spectral, imbalance=1.3), True),
        "METIS": (functools.partial(metis_bi_partition, imbalance=1.3), False),
    }
    
    graph_acyclic = nx.is_directed_acyclic_graph(graph)
    
    df_list_dict = []
    
    for alg_name, val in algorithms_to_run.items():
        try:
            alg_func, require_acyc = val
            if (not graph_acyclic) and require_acyc:
                continue
            graph_copy = graph.copy()
            parts = alg_func(graph_copy)
            if (not is_valid_bi_partition(graph, parts)):
                print("Invalid bi-partition generated by " + alg_name + " during graph " + graph_name + "!")
                continue
            
            df_list_dict.append({
                "Algorithm": alg_name,
                "Cut Ratio": cut_edges_ratio(graph, parts),
                "Weight Imbalance": partition_imbalance(graph, parts),
                "Misalignment Ratio": misaligned_cut_edges_ratio(graph, parts),
                "Graph": graph_name
            })
        except:
            print("Error during graph " + graph_name + " and algorithm " + alg_name + ".")
        
    df = pd.DataFrame(df_list_dict, columns=["Algorithm", "Cut Ratio", "Weight Imbalance", "Misalignment Ratio", "Graph"])
    
    for graph_n, group in df.groupby("Graph"):
        print("\nGraph: " + graph_n)
        print(group.to_string())
    
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()
