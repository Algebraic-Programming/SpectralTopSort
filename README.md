# SpectralTopSort

## About
Generates a topological order of a directed acyclic graph which is good for locality/data reuse, meaning the non-zero entries of the adjacency matrix are closer to the diagonal.

It works by recursively bi-sectioning using a direction incetivised spectral bi-partitioning algorithm.

## How to run
With a dot file:
```
python ./SpectralTopologicalOrdering.py ../test_data/test_graph.dot
```
With an upper-triangular matrix in MatrixMarket File Format:
```
python ./SpectralTopologicalOrdering.py <Adjacency Matrix>.mtx
```
With a lower-triangular matrix in MatrixMarket File Format:
```
python ./SpectralTopologicalOrdering.py ../test_data/test_graph.mtx --low
```

## Reference
TBA
