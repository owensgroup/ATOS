# ATOS
## Prerequisite
- CUDA (>10.0)
- NVSHMEM (can be download from https://developer.nvidia.com/nvshmem by joinning NVIDIA developer)
- GCC
- METIS
- OpenMPI (>4.0)


## Compile BFS
Under bfs\_nvlink and bfs\_ib directory, compile the code with make

To test the given datasets, run the figure5\_persist.sh and figure5\_prio.sh to generate performance data for BFS on NVLinks.
To generate performance results for BFS on InfiniBand, run run\_bfs.batch first and extract the results with script figure10\_bfs.sh

## Compile PageRank
Under pr\_nvlink and pr\_ib directory, compile the code with make

To test the given datasets, run the figure7\_persist.sh and figure7\_discrete.sh to generate performance data for PageRank on NVLinks.
To generate performance results for BFS on InfiniBand, run run\_pr.batch first and extract the results with script figure11\_pr.sh


## Pre-generated Performance Data
Pre-generated performance data are under perf\_data directory. To extract the performance results, run figure5\_persist.sh, figure5\_prio.sh and figure10\_bfs.sh under bfs\_nvlink and bfs\_ib directory and run figure7\_discrete.sh, figure7\_persist.sh and figure11\_pr.sh under pr\_nvlink and pr\_ib directory.
