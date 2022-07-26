# ATOS
## Content
* `dataset`: Folder containing graph dataset downloader and metadata for each graph.
* `single-GPU`: Folder containing single-GPU Atos asynchronous graph analytics implementations.
* `bfs_nvlink`: Folder containing multi-GPU Atos asynchronous BFS implementations on NVLink connected multi-GPU system.
* `pr_nvlink`: Folder containing multi-GPU Atos asynchronous PageRank implementations on NVLink connected multi-GPU system.
* `bfs_ib`: Folder containing multi-GPU Atos asynchronous BFS implementations on InfiniBand(IB) connected multi-GPU system.
* `pr_ib`: Folder containing multi-GPU Atos asynchronous PageRank implementations on InfiniBand(IB) connected multi-GPU system.
* `comm`: Folder containing implementations of distributed standard/priority queues and communciation aggregator.
* `perf_data`: Folder containing the performance results of BFS and PageRank on NVLink and IB systems.


## Prerequisite
### Single-GPU Atos Graph Analytices
- CUDA (>10.0)
- GCC
#### Required Environment Variables
Set the followling environment variables accordingly based on your depedency path
- CUDA\_HOME
### Multi-GPU Atos BFS and PageRank
- CUDA (>10.0)
- GCC
- NVSHMEM (can be download from https://developer.nvidia.com/nvshmem by joinning NVIDIA developer)
- METIS (https://github.com/KarypisLab/METIS) 
   Compile METIS with 64 bites option
- OpenMPI (>4.0)
#### Required Environment Variables
Set the followling environment variables accordingly based on your depedency path
- CUDA\_HOME
- METIS64\_HOME
- NVSHMEM\_HOME
- MPI\_HOME

## Compile BFS 
Under bfs\_nvlink and bfs\_ib directory, compile the code with make

To test the given datasets, run the figure5\_persist.sh and figure5\_discrete.sh to generate performance data for BFS on NVLinks.
To generate performance results for BFS on InfiniBand, run run\_pr.batch first and extract the results with script figure10\_pr.sh



## Compile PageRank
Under pr\_nvlink and pr\_ib directory, compile the code with make

To test the given datasets, run the figure7\_persist.sh and figure7\_discrete.sh to generate performance data for PageRank on NVLinks.
To generate performance results for PageRank on InfiniBand, run run\_pr.batch first and extract the results with script figure11\_pr.sh


## Pre-generated Performance Data
Pre-generated performance data are under perf\_data directory. To extract the performance results, run figure5\_persist.sh, figure5\_prio.sh and figure10\_bfs.sh under bfs\_nvlink and bfs\_ib directory and run figure7\_discrete.sh, figure7\_persist.sh and figure11\_pr.sh under pr\_nvlink and pr\_ib directory.
