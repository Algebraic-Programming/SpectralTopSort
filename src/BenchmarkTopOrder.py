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

import networkx as nx
import pydot
import os
import scipy
import sys

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix, check_valid_top_order
from BaselinesTopOrder import bfs_topOrd

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
    
    top_order = bfs_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(compute_edge_lengths(graph, top_order))
    print(compute_stack_distances(graph, top_order))

    return 0

if __name__ == "__main__":
    main()