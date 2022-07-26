## Compile PageRank

## Required Environment Variables
Set the followling environment variables accordingly based on your depedency path for `Makefile`
- CUDA\_HOME
- METIS64\_HOME
- NVSHMEM\_HOME
- MPI\_HOME

Then `make` to compile the code

To generate performance results for BFS on InfiniBand, run `run_pr.batch` first and extract the results with script `figure11_pr.sh`


