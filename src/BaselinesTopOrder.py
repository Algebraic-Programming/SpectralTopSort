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

import collections
import copy
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import pydot
import os
import scipy
import sys

from SpectralTopologicalOrdering import nx_graph_from_upper_triangular_matrix, check_valid_top_order, part_requiring_recursion

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
        
        ready_children = []
        for edge in graph.out_edges(v):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                ready_children.append(tgt)
                
        ready_children.sort(key=graph.out_degree)
        for chld in ready_children:
            deque.append(chld)

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


def sum_edge_length_parent_topOrd(graph: nx.MultiDiGraph):
    dependency_counter = dict()
    priority = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        priority[v] = 0
        
    topOrd = []
    queue = []
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            queue.append(v)
            
    while len(queue) > 0:
        max_score = None
        best_vert = None
        for v in queue:
            if max_score == None or priority[v] > max_score:
                max_score = priority[v]
                best_vert = v
        
        topOrd.append(best_vert)
        queue.remove(best_vert)
        for v in queue:
            priority[v] += graph.in_degree(v)
        
        for edge in graph.out_edges(best_vert):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                queue.append(tgt)
                for par, _ in graph.in_edges(tgt):
                    priority[tgt] += topOrd[::-1].index(par)
        
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





def access_pattern_min_topOrd(graph: nx.MultiDiGraph):
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
        min_score = None
        v = None
        for u in ready:
            score = len(accesses) + 1
            for edge in graph.in_edges(u):
                src = edge[0]
                score = min(score, len(accesses) - last_access[src])
                
            if min_score == None or score < min_score:
                min_score = score
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


def max_sibling_score_in_window(graph: nx.MultiDiGraph, window_size: int = 5):
    """
    Locality reordering based on the paper \"Speedup Graph Processing by Graph Ordering\" by Hao Wei, Jeffrey Xu Yu, Can Lu, and Xuemin Lin.
    """
    
    assert(window_size >= 1)
    
    def score(u: None, v: None):
        score_acc = 0
        
        u_par = set([par for par, _ in graph.in_edges(u)])
        v_par = set([par for par, _ in graph.in_edges(v)])
        
        if u in v_par:
            score_acc += 1
        if v in u_par:
            score_acc += 1
            
        score_acc += len( u_par & v_par )
        
        return score_acc
    
    def score_sum(u: None, vert_list: list[None]):
        return sum([score(u,v) for v in vert_list])
        
    
    dependency_counter = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        
    topOrd = []
    queue = []
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            queue.append(v)
            
    while len(queue) > 0:
        max_score = None
        best_vert = None
        for v in queue:
            v_score = score_sum(v, topOrd[-window_size:])
            if max_score == None or v_score > max_score:
                max_score = v_score
                best_vert = v
        
        topOrd.append(best_vert)
        queue.remove(best_vert)
        
        for edge in graph.out_edges(best_vert):
            tgt = edge[1]
            dependency_counter[tgt] += 1
            if dependency_counter[tgt] == graph.in_degree(tgt):
                queue.append(tgt)
        
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd


def cuthill_Mckee(graph: nx.MultiDiGraph):
    """
    Locality reordering based on the Cuthill--Mckee heuristic.
    """
    
    dependency_counter = dict()
    priority = dict()
    for v in graph.nodes:
        dependency_counter[v] = 0
        priority[v] = graph.number_of_nodes()
        
    topOrd = []
    sources = []
    queue = []
    next_queue = []
    heapq.heapify(sources)
    heapq.heapify(queue)
    heapq.heapify(next_queue)
    
    for v in graph.nodes:
        if dependency_counter[v] == graph.in_degree(v):
            heapq.heappush(sources, (graph.out_degree(v), v))
            
    while sources:
        _, v = heapq.heappop(sources)
        heapq.heappush(queue, (priority[v], v))
        
        while queue or next_queue:
            if not queue:
                queue = copy.deepcopy(next_queue)
                heapq.heapify(queue)
                next_queue.clear()
                heapq.heapify(next_queue)
                
            _, u = heapq.heappop(queue)
            topOrd.append(u)
        
            for edge in graph.out_edges(u):
                tgt = edge[1]
                priority[tgt] = min(priority[tgt], len(topOrd) - 1)
                dependency_counter[tgt] += 1
                if dependency_counter[tgt] == graph.in_degree(tgt):
                    heapq.heappush(next_queue, (priority[tgt], tgt))
    
    assert(len(topOrd) == graph.number_of_nodes())
    return topOrd

def recursive_acyclic_bisection(graph: nx.MultiDiGraph, acyc_bisec_method):
    if (not nx.is_directed_acyclic_graph(graph)):
        print("Graph is not acyclic")
        return []
    
    nx.set_node_attributes(graph, "", "part")
    
    # iterations
    processing_part = part_requiring_recursion(graph)
    while (processing_part != None):
        vertices_of_part = [vert for vert in graph.nodes if graph.nodes[vert]["part"] == processing_part ]
        assert(len(vertices_of_part) > 1)
        
        subgraph = nx.induced_subgraph(graph, vertices_of_part)
        subgraph = subgraph.copy()
        for vert in subgraph.nodes:
            del subgraph.nodes[vert]["part"]
        
        earlier, later = acyc_bisec_method(subgraph)
        e_set = set(earlier)
        l_set = set(later)
        
        # Swapping should be needed only for first iteration
        edge_diff = 0
        for edge in subgraph.edges:
            if (edge[0] in e_set) and (edge[1] in l_set):
                edge_diff += 1
            if (edge[0] in l_set) and (edge[1] in e_set):
                edge_diff -= 1
        
        if (edge_diff < 0):
            earlier, later = later, earlier
            
        assert(len(earlier)>0 and len(later)>0)
    
        for vert in earlier:
            graph.nodes[vert]["part"] = graph.nodes[vert]["part"] + "0"
            
        for vert in later:
            graph.nodes[vert]["part"] = graph.nodes[vert]["part"] + "1"
        
        processing_part = part_requiring_recursion(graph)

    # Generate Topological order from parts
    vert_and_parts = [[graph.nodes[vert]["part"], vert] for vert in graph.nodes]
    vert_and_parts.sort()
    
    return [ item[1] for item in vert_and_parts ]




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
    
    plt.figure("BFS Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = dfs_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("DFS Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    
    top_order = earliest_parent_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Earliest Parent Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    top_order = sum_edge_length_parent_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Sum Edge Length Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_max_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Access Pattern Max Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_min_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Access Pattern Min Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_sum_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Access Pattern Sum Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    top_order = access_pattern_avg_topOrd(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Access Pattern Avg Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    top_order = max_sibling_score_in_window(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Max Windowed Sibling Score Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    top_order = cuthill_Mckee(graph)
    
    if (not check_valid_top_order(graph, top_order)):
        print("Invalid Topological order!")
        return 1
    
    plt.figure("Cuthill--Mckee Reordered Graph: " + graph_name)
    plt.spy(nx.to_numpy_array(graph, top_order))
    
    
    
    
    plt.show()

    return 0

if __name__ == "__main__":
    main()