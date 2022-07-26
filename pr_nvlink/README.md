## Compile PageRank
## Required Environment Variables
Set the followling environment variables accordingly based on your depedency path for `Makefile`
- CUDA\_HOME
- METIS64\_HOME
- NVSHMEM\_HOME
- MPI\_HOME

Then `make` to compile the code

To test the given datasets, run the `figure7_persist.sh` and `figure7_discrete.sh` to generate performance data for PageRank on NVLinks.


