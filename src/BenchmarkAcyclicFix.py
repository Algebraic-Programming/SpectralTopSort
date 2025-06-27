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

from BenchmarkBiPartition import is_valid_bi_partition, nx_graph_from_matrix, cut_edges_ratio, partition_imbalance, spectral_split_classic, Undirected_FM_from_scratch, metis_bi_partition
from SpectralTopologicalOrdering import spectral_split, top_order_small_cut_fix


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
            print("Unknown file format!" + " (" + graph_file + ")")
            continue
        
        graph_dict[graph_name] = graph
    
    # Key: Base Algorithm Name, Function
    base_algorithms_to_run = {
        "Spectal_directed_2.0": functools.partial(spectral_split, lp=2.0, lq=2.0, const_dir=0.5),
        # "Spectal_directed_1.5": functools.partial(spectral_split, lp=1.5, lq=1.5),
        # "Spectal_directed_1.1": functools.partial(spectral_split, lp=1.1, lq=1.1),
        "Spectal_classic_2.0": functools.partial(spectral_split_classic, lp=2.0, lq=2.0),
        # "Spectal_classic_1.5": functools.partial(spectral_split_classic, lp=1.5, lq=1.5),
        # "Spectal_classic_1.1": functools.partial(spectral_split_classic, lp=1.1, lq=1.1),
        "FM_Undirected": functools.partial(Undirected_FM_from_scratch, imbalance=1.3), 
        "METIS": functools.partial(metis_bi_partition, imbalance=1.3),
    }
    
    # Key: AcyclicFix Algorithm Name, Function
    fix_algorithms_to_run = {
        "Small_Cut_Fix": top_order_small_cut_fix,
    }
    
    df_list_dict = []
    
    for graph_name, graph in graph_dict.items():    
        print("Graph: " + graph_name + " Vertices: "+ str(graph.number_of_nodes()) + " Edges: " + str(graph.number_of_edges()))
    
        graph_acyclic = nx.is_directed_acyclic_graph(graph)
    
        for base_alg_name, base_alg_func in base_algorithms_to_run.items():
            try:
                if not graph_acyclic:
                    print("Graph not acyclic -- Skipped")
                    continue
                graph_copy = graph.copy()
                earlier, later = base_alg_func(graph_copy)
                
                e_set = set(earlier)
                l_set = set(later)
                edge_diff = 0
                for edge in graph_copy.edges:
                    if (edge[0] in e_set) and (edge[1] in l_set):
                        edge_diff += 1
                    if (edge[0] in l_set) and (edge[1] in e_set):
                        edge_diff -= 1
                
                if (edge_diff < 0):
                    earlier, later = later, earlier                
                
                parts = [earlier, later]
                if (not is_valid_bi_partition(graph, parts)):
                    print("Invalid bi-partition generated by " + base_alg_name + " during graph " + graph_name + "!")
                    continue
                
                original_cut_edge_ratio = cut_edges_ratio(graph, parts)
                original_weight_imbalance = partition_imbalance(graph, parts)
                
                for fix_alg_name, fix_alg_func in fix_algorithms_to_run.items():
                    try:
                        graph_copy = graph.copy()
                        nx.set_node_attributes(graph_copy, "", "part")
                        early_fix, late_fix, correctly_assigned = fix_alg_func(graph_copy, earlier, later)
                        parts_fixed = [early_fix, late_fix]
                        
                        df_list_dict.append({
                            "Base Algorithm": base_alg_name,
                            "Fix Algorithm": fix_alg_name,
                            "Cut Ratio Before Fix": original_cut_edge_ratio,
                            "Cut Ratio After Fix": cut_edges_ratio(graph, parts_fixed),
                            "Weight Imbalance Before Fix": original_weight_imbalance,
                            "Weight Imbalance After Fix": partition_imbalance(graph, parts_fixed),
                            "Correctly Assigned": correctly_assigned,
                            "Graph": graph_name
                        })
                        
                    except Exception as e:
                        print("Error during graph " + graph_name + " and fix algorithm " + fix_alg_name + ".")
                        print(e)
                
            except Exception as e:
                print("Error during graph " + graph_name + " and base algorithm " + base_alg_name + ".")
                print(e)
        
    df = pd.DataFrame(df_list_dict, columns=[
                                                "Base Algorithm",
                                                "Fix Algorithm",
                                                "Cut Ratio Before Fix",
                                                "Cut Ratio After Fix",
                                                "Weight Imbalance Before Fix",
                                                "Weight Imbalance After Fix",
                                                "Correctly Assigned",
                                                "Graph"
                                            ])
    
    for graph_n, group in df.groupby("Graph"):
        print("\nGraph: " + graph_n)
        print(group.to_string())
        
        
    write_data = False
    if (write_data):
        data_output_file_name = 'AcyclicFix_' + name_for_stats_file + '.csv'
        df.to_csv(data_output_file_name)

    return 0

if __name__ == "__main__":
    main()
