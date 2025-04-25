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

import metis
import networkx as nx
import pydot
import os
import scipy
import sys

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix

def is_valid_bi_partition(graph: nx.MultiDiGraph, parts: list[set, set]):
    if not len(parts) == 2:
        return False
    
    if not graph.number_of_nodes == len(parts[0] + parts[1]):
        return False
    
    for vert in graph.nodes:
        if (vert in parts[0]) and (vert in parts[1]):
            return False
        if (vert not in parts[0]) and (vert not in parts[1]):
            return False
        
    return True

def metis_bi_partition(graph: nx.MultiDiGraph):
    (edgecuts, split) = metis.part_graph(graph, 2)
    
    parts = [set(), set()]
    nodes = list(graph.nodes)
    for ind, part in enumerate(split):
        parts[part].add(nodes[ind])

    return parts


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
    
    parts = metis_bi_partition(graph)
    assert(is_valid_bi_partition(graph, parts))
    print(parts)

    return 0

if __name__ == "__main__":
    main()