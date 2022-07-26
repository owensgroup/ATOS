# Compile BFS

## Required Environment Variables
Set the followling environment variables accordingly based on your depedency path for `Makefile`
- CUDA\_HOME
- METIS64\_HOME
- NVSHMEM\_HOME
- MPI\_HOME

Then `make` to compile the code

To generate performance results for BFS on InfiniBand, run `run\_bfs.batch` first and extract the results with script `figure10\_bfs.sh`


