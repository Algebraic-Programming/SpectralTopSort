import kahip
import networkx as nx

def kahip_bi_partition(graph: nx.MultiDiGraph, imbalance: float):
    
    vertices = list(graph.nodes)
    
    vert_to_index = dict()
    for i, vert in enumerate(vertices):
        vert_to_index[vert] = i

    #build adjacency array representation of the graph
    
    xadj = [0]
    adjncy = []
    for vert in vertices:
        neighbours = set()
        for nbr in graph.successors(vert):
            neighbours.add( nbr )
        for nbr in graph.predecessors(vert):
            neighbours.add( nbr )
            
        neighbours.discard( vert )
        
        for nbr in neighbours:
            adjncy.append( vert_to_index[nbr] )
            
        xadj.append( len(adjncy) )
    
    vwgt           = [1 for v in vertices]
    adjcwgt        = [1 for e in adjncy]
    supress_output = 0
    nblocks        = 2 
    seed           = 0

    # set mode 
    #const int FAST           = 0;
    #const int ECO            = 1;
    #const int STRONG         = 2;
    #const int FASTSOCIAL     = 3;
    #const int ECOSOCIAL      = 4;
    #const int STRONGSOCIAL   = 5;
    mode = 2 

    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, 
                                adjncy,  nblocks, imbalance, 
                                supress_output, seed, mode)

    return [[vert for vert in vertices if blocks[ vert_to_index[vert] ] == 0  ], [vert for vert in vertices if blocks[ vert_to_index[vert] ] == 1  ]]