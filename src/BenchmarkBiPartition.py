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
import scipy.sparse
import seaborn as sns
import sys

from SpectralTopologicalOrdering import spectral_split, spectral_acyclic_bi_partition, spectral_acyclic_bi_partition_with_spec_values
from BaselinesBiPartition import is_valid_bi_partition, spectral_split_classic, metis_bi_partition, nx_graph_from_matrix, FM_split_from_scratch, FM_split_improving_spectral, Undirected_FM_from_scratch, metis_with_acyclic_fix, spectral_split_classic_acyclic_fix, rmlgp_partition
# from BaselineKahip import kahip_bi_partition

def cut_edges_ratio(graph: nx.MultiDiGraph, parts: list) -> float:
    if graph.number_of_edges() == 0:
        return 0
    
    part_set_0 = set(parts[0])
    part_set_1 = set(parts[1])
    
    cut_edges = 0
    
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        
        if u == v:
            continue
        
        if u in part_set_0 and v in part_set_1:
            cut_edges += 1
        if u in part_set_1 and v in part_set_0:
            cut_edges += 1
    
    return cut_edges / graph.number_of_edges()

def ncut_edges_ratio(graph: nx.MultiDiGraph, parts: list) -> float:
    if graph.number_of_edges() == 0:
        return 0
    
    part_set_0 = set(parts[0])
    part_set_1 = set(parts[1])
    
    cut_edges = 0
    edges_p0 = 0
    edges_p1 = 0
    
    for edge in graph.edges:
        u = edge[0]
        v = edge[1]
        
        if u == v:
            continue
        
        if u in part_set_0 and v in part_set_1:
            cut_edges += 1
            edges_p0 += 1
            edges_p1 += 1
        if u in part_set_1 and v in part_set_0:
            cut_edges += 1
            edges_p0 += 1
            edges_p1 += 1
        if u in part_set_0 and v in part_set_0:
            edges_p0 += 1
        if u in part_set_1 and v in part_set_1:
            edges_p1 += 1
    
    if cut_edges == 0:
        return 0
    else:
        return ((cut_edges / edges_p0) + (cut_edges / edges_p1))
        
    

    

def partition_imbalance(graph: nx.MultiDiGraph, parts: list) -> float:
    bigger = max(len(parts[0]), len(parts[1]))
    smaller = min(len(parts[0]), len(parts[1]))
    
    return (bigger - smaller) / graph.number_of_nodes()

def misaligned_cut_edges_ratio(graph: nx.MultiDiGraph, parts: list) -> float:
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
        print("Usage: " + sys.argv[0] + " <graph directory>")
        print("Usage: " + sys.argv[0] + " <graph.dot>")
        print("Usage: " + sys.argv[0] + " <graph.mtx>" + " (optional --low)")
        print("\nIf the flag \"--low\" is passed then only the strictly lower triangular part is taken.")
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
            
        elif (graph_file[-3:] == "mtx"):
            matrix = scipy.io.mmread(graph_file)
            if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
                matrix = matrix.transpose()
                matrix = scipy.sparse.triu(matrix, k=1)
            if (len(sys.argv) == 3) and (sys.argv[2] == "--acyc"):
                upper_matrix = scipy.sparse.triu(matrix, k=1)
                top_graph = nx_graph_from_matrix(upper_matrix.toarray())
                
                lower_matrix = scipy.sparse.tril(matrix, k=-1)
                lower_matrix = lower_matrix.transpose()
                low_graph = nx_graph_from_matrix(lower_matrix.toarray())
                
                if top_graph.number_of_edges() >= low_graph.number_of_edges():
                    print("Taking top part of matrix")
                    matrix = upper_matrix
                else:
                    print("Taking bottom part of matrix")
                    matrix = lower_matrix
                
                
            graph = nx_graph_from_matrix(matrix.toarray())
            
        else:
            print("Unknown file format!" + " (" + graph_file + ")")
            continue
        
        if not nx.is_weakly_connected(graph):
            print(f"Graph {graph_name} is not weakly connected.\nReplacing it with largest weakly connected component!\n")
            temp_graph = None
            
            weak_comp = nx.weakly_connected_components(graph)
            for comp in weak_comp:
                if temp_graph == None or len(comp) > temp_graph.number_of_nodes():
                    temp_graph = nx.induced_subgraph(graph, comp).copy()
            
            graph = temp_graph
        
        graph_dict[graph_name] = graph
    
    # Key: Algorithm Name, Value: (Function, Requires Acyclic)
    algorithms_to_run = {
        # "Spectal_directed_2.0": (functools.partial(spectral_split, lp=2.0, lq=2.0, const_dir=0.5), False),
        # "Spectal_directed_1.5": (functools.partial(spectral_split, lp=1.5, lq=1.5), False),
        # "Spectal_directed_1.1": (functools.partial(spectral_split, lp=1.1, lq=1.1), False),
        "Spectal_directed_acyclic_2.0": (functools.partial(spectral_acyclic_bi_partition, lp=2.0, const_dir=0.5), True),
        # "Spectal_directed_acyclic_1.5": (functools.partial(spectral_acyclic_bi_partition, lp=1.5), True),
        # "Spectal_directed_acyclic_1.1": (functools.partial(spectral_acyclic_bi_partition, lp=1.1), True),
        # "Spectal_directed_acyclic_spec_2.0": (functools.partial(spectral_acyclic_bi_partition_with_spec_values, lp=2.0), True),
        # "Spectal_directed_acyclic_spec_1.5": (functools.partial(spectral_acyclic_bi_partition_with_spec_values, lp=1.5), True),
        # "Spectal_directed_acyclic_spec_1.1": (functools.partial(spectral_acyclic_bi_partition_with_spec_values, lp=1.1), True),
        # "Spectal_classic_2.0": (functools.partial(spectral_split_classic, lp=2.0, lq=2.0), False),
        # "Spectal_classic_1.5": (functools.partial(spectral_split_classic, lp=1.5, lq=1.5), False),
        # "Spectal_classic_1.1": (functools.partial(spectral_split_classic, lp=1.1, lq=1.1), False),
        "Spectal_classic_acyclic_2.0": (functools.partial(spectral_split_classic_acyclic_fix, lp=2.0, lq=2.0), True),
        # "Spectal_classic_acyclic_1.5": (functools.partial(spectral_split_classic_acyclic_fix, lp=1.5, lq=1.5), True),
        # "Spectal_classic_acyclic_1.1": (functools.partial(spectral_split_classic_acyclic_fix, lp=1.1, lq=1.1), True),
        # "FM_Undirected": (functools.partial(Undirected_FM_from_scratch, imbalance=1.3), False), 
        # "FM_from_scratch": (functools.partial(FM_split_from_scratch, imbalance=1.3), True),
        # "FM_after_spectral": (functools.partial(FM_split_improving_spectral, imbalance=1.3), True),
        # "METIS": (functools.partial(metis_bi_partition, imbalance=1.3), False),
        # "METIS_with_acyclic_fix": (functools.partial(metis_with_acyclic_fix, imbalance=1.3), True),
        # "rMLGP": (functools.partial(rmlgp_partition, binary_path="./rMLGP"), True),
        # "KaHIP": (functools.partial(kahip_bi_partition, imbalance=0.3), False),
    }
    
    df_list_dict = []
    
    all_graphs_acyclic = True
    all_alg_require_acyc = True
    
    for graph_name, graph in graph_dict.items():    
        print("Graph: " + graph_name + " Vertices: "+ str(graph.number_of_nodes()) + " Edges: " + str(graph.number_of_edges()))
    
        graph_acyclic = nx.is_directed_acyclic_graph(graph)
        all_graphs_acyclic = all_graphs_acyclic and graph_acyclic
    
        for alg_name, val in algorithms_to_run.items():
            try:
                alg_func, require_acyc = val
                all_alg_require_acyc = all_alg_require_acyc and require_acyc
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
                    "Normalised Cut Ratio": ncut_edges_ratio(graph, parts),
                    "Weight Imbalance": partition_imbalance(graph, parts),
                    "Misalignment Ratio": misaligned_cut_edges_ratio(graph, parts),
                    "Graph": graph_name
                })
            except Exception as e:
                print("Error during graph " + graph_name + " and algorithm " + alg_name + ".")
                print(e)
        
    df = pd.DataFrame(df_list_dict, columns=["Algorithm", "Cut Ratio", "Normalised Cut Ratio", "Weight Imbalance", "Misalignment Ratio", "Graph"])
    
    for graph_n, group in df.groupby("Graph"):
        print("\nGraph: " + graph_n)
        print(group.to_string())
        
        
    write_data = False
    if (write_data):
        data_output_file_name = ''
        if all_graphs_acyclic and all_alg_require_acyc:
            data_output_file_name += 'Acyc'
        data_output_file_name += 'BiPart_'
        data_output_file_name += name_for_stats_file
        data_output_file_name += '.csv'
        df.to_csv(data_output_file_name)
    
    
    if not isDir:
        plt.show()

    return 0

if __name__ == "__main__":
    main()
