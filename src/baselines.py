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

import collections
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import pydot
import os
import scipy
import sys

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix, check_valid_top_order

def bfs_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    deque = collections.deque()
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            deque.append(v)
            
    while deque:
        v = deque.popleft()
        topOrd.append(v)
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                deque.append(tgt)

    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd

def dfs_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    deque = collections.deque()
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            deque.append(v)
            
    while deque:
        v = deque.pop()
        topOrd.append(v)
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                deque.append(tgt)
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd

def earliest_parent_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    priority = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        priority[v] = graph.number_of_nodes()
        
    topOrd = []
    queue = []
    heapq.heapify(queue)
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            heapq.heappush(queue, (priority[v], v))
            
    while queue:
        _, v = heapq.heappop(queue)
        topOrd.append(v)
        for edge in graph.out_edges(v):
            tgt = edge[1]
            priority[tgt] = min(priority[tgt], len(topOrd) - 1)
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                heapq.heappush(queue, (priority[tgt], tgt))
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd

def access_pattern_max_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    ready = set()
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            ready.add(v)
            
    accesses = []
    last_access = dict()
            
    while ready:
        max_score = None
        v = None
        for u in ready:
            score = 0
            for edge in graph.in_edges(u):
                src = edge[0]
                score = max(score, len(accesses) - last_access[src])
                
            if max_score == None or score > max_score:
                max_score = score
                v = u
        
        topOrd.append(v)
        for edge in graph.in_edges(v):
            src = edge[0]
            accesses.append(src)
            last_access[src] = len(accesses) - 1
        
        accesses.append(v)
        last_access[v] = len(accesses) - 1
        ready.remove(v)
        
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                ready.add(tgt)
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd





def access_pattern_sum_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    ready = set()
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            ready.add(v)
            
    accesses = []
    last_access = dict()
            
    while ready:
        max_score = None
        v = None
        for u in ready:
            score = 0
            for edge in graph.in_edges(u):
                src = edge[0]
                score += len(accesses) - last_access[src]
                
            if max_score == None or score > max_score:
                max_score = score
                v = u
        
        topOrd.append(v)
        for edge in graph.in_edges(v):
            src = edge[0]
            accesses.append(src)
            last_access[src] = len(accesses) - 1
        
        accesses.append(v)
        last_access[v] = len(accesses) - 1
        ready.remove(v)
        
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                ready.add(tgt)
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd



def access_pattern_avg_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    ready = set()
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            ready.add(v)
            
    accesses = []
    last_access = dict()
            
    while ready:
        max_score = None
        v = None
        for u in ready:
            score = 0
            for edge in graph.in_edges(u):
                src = edge[0]
                score += len(accesses) - last_access[src]
            if graph.in_degree(u) > 0:
                score /= graph.in_degree(u)
                
            if max_score == None or score > max_score:
                max_score = score
                v = u
        
        topOrd.append(v)
        for edge in graph.in_edges(v):
            src = edge[0]
            accesses.append(src)
            last_access[src] = len(accesses) - 1
        
        accesses.append(v)
        last_access[v] = len(accesses) - 1
        ready.remove(v)
        
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                ready.add(tgt)
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd





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
        plt.figure("Original Graph: " + graph_name)
        plt.spy(nx.to_numpy_array(graph, list(nx.topological_sort(graph))))
        
    elif (graph_file[-3:] == "mtx"):
        matrix = scipy.io.mmread(graph_file, spmatrix=True)
        if (len(sys.argv) == 3) and (sys.argv[2] == "--low"):
            matrix = matrix.transpose()
        matrix = scipy.sparse.triu(matrix, k=1)
        graph = nx_graph_from_upper_triangular_matrix(matrix.toarray())
        
        plt.figure("Original Graph: " + graph_name)
        plt.spy(matrix.toarray())
    else:
        print("Unknown file format!")
        return 1




    top_order = bfs_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("BFS Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = dfs_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("DFS Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    
    top_order = earliest_parent_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("Earliest Parent Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_max_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("Access Pattern Max Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_sum_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("Access Pattern Sum Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_avg_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    print(top_order)
    
    plt.figure("Access Pattern Avg Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()