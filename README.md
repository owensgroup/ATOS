# ATOS
## Content
* `dataset`: Folder containing graph dataset downloader for each graph.
* `single-GPU`: Folder containing single-GPU Atos asynchronous graph analytics implementations.
* `bfs_nvlink`: Folder containing multi-GPU Atos asynchronous BFS implementations on NVLink connected multi-GPU system.
* `pr_nvlink`: Folder containing multi-GPU Atos asynchronous PageRank implementations on NVLink connected multi-GPU system.
* `bfs_ib`: Folder containing multi-GPU Atos asynchronous BFS implementations on InfiniBand(IB) connected multi-GPU system.
* `pr_ib`: Folder containing multi-GPU Atos asynchronous PageRank implementations on InfiniBand(IB) connected multi-GPU system.
* `comm`: Folder containing implementations of distributed standard/priority queues and communication aggregator.
* `perf_data`: Folder containing the performance results of BFS and PageRank on NVLink and IB systems.


## Prerequisite
### Single-GPU Atos Graph Analytics
- CUDA (V11.4.120 or newer)
- GCC (9.4.0 or newer)
- boost (1.63 or newer)
#### Required Environment Variables
Set the following environment variables accordingly based on your dependency path
- CUDA\_HOME
### Multi-GPU Atos BFS and PageRank
- CUDA (V11.4.120 or newer)
- GCC (9.4.0 or newer)
- boost (1.63 or newer)
- NVSHMEM (can be downloaded from https://developer.nvidia.com/nvshmem by joining NVIDIA developer)
- METIS (https://github.com/KarypisLab/METIS) 
   Compile METIS with 64 bites option
- OpenMPI (4.0.5 or newer) or IBM Spectrum MPI on Summit
#### Required Environment Variables
Set the following environment variables accordingly based on your dependency path
- CUDA\_HOME
- METIS64\_HOME
- NVSHMEM\_HOME
- MPI\_HOME

# Reproduce Multi-GPU Performance Test on BFS and PageRank

1. Download Datasets

   Several tested graph datasets are included under the `datasets` directory. 
   Under each graph dataset folder, one needs to run `make` to download the dataset. The downloaded graph datasets are either `.mtx` format or `.csr` format. Atos uses the `.csr` format. In the case of `.mtx` format, one can use `gen_csr` tool under directory `datasets` to convert the `mtx` format to `csr` format.

2. Compile BFS and PageRank

   * BFS implementations on NVLink systems are under the `bfs_nvlink` directory; BFS implementations on InfiniBand(IB) systems are under the `bfs_ib` directory.
   * Under `bfs_nvlink` and `bfs_ib` directory, run `make` to compile the code.

   * PageRank implementations on NVLink systems are under the `pr_nvlink` directory; PageRank implementations on InfiniBand(IB) systems are under the `pr_ib` directory.
   * Under `pr_nvlink` and `pr_ib` directory, run `make` to compile the code

3. Run Performance Test for BFS on NVLink System
   * Go the `bfs_nvlink` folder.
   * To test BFS on the graphs under the datasets directory, run the `figure5_persist.sh` and `figure5_discrete.sh` to generate and extract the performance data for BFS on NVLinks.
   * The script file `figure5_persist.sh` generates performance results for BFS implementation using a standard queue and persistent kernel scheme.
   * The script file `figure5_discrete.sh` generate performance results for BFS implementation using priority queue and discrete kernel scheme.
   * The `figure5_persist.sh` and `figure5_discrete.sh` firstly run the performance tests and output the results to a temporary file; then they extract and print the performance results. 

   Note: If abnormal results are generated by `figure5_persist.sh` and `figure5_discrete.sh`, please re-run the performance tests as the print output from multi-processes can tangle in the way that our script fails to extract the performance output correctly. 

4. Run Performance Test for BFS on InfiniBand(IB) System
   * Go to the `bfs_ib` folder.
   * To test on the graphs under the datasets directory, run the `run_bfs.sh` or the `run_bfs.batch` if on Summit to generate the performance results for BFS on IB system.
   * Then extract the performance results by `./figure10_bfs.sh outputfile` 

   Note: If abnormal results are generated by `figure10_bfs.sh`, please re-run the performance tests as the print output from multi-processes can tangle in the way that our script fails to extract the performance output correctly. 

5. Run Performance Test for PageRank on NVLink System
   * Go to the `pr_nvlink` folder. <br>
   * To test on the graphs under the datasets directory, run the `figure7_persist.sh` and `figure7_discrete.sh` to generate and extract the performance data for PageRank on NVLinks.
   * The script file `figure7_persist.sh` generates performance results for BFS implementation using a standard queue and persistent kernel scheme.
   * The script file `figure7_discrete.sh` generate performance results for BFS implementation using standard queue and discrete kernel scheme.
   * The `figure7_persist.sh` and `figure7_discrete.sh` firstly run the performance tests and output the results to a temporary file; then they extract and print the performance results. 

   Note: If abnormal results are generated by `figure7_persist.sh` and `figure7_discrete.sh`, please re-run the performance tests as the print output from multi-processes can tangle in the way that our script fails to extract the performance output correctly. 

6. Run Performance Test for PageRank on InfiniBand(IB) System
   * Go to the `pr_ib` folder.
   * To test on the graphs under the datasets directory, run the `run_pr.sh` or the `run_pr.batch` if on Summit to generate the performance results for PageRank on IB system.
   * Then extract the performance results by `./figure11_pr.sh outputfile` 

   Note: If abnormal results are generated by `figure11_pr.sh`, please re-run the performance tests as the print output from multi-processes can tangle in the way that our script fails to extract the performance output correctly. 


## Pre-generated Performance Data
Pre-generated performance data are under `perf_data` directory. To extract the performance results, run `figure5_persist.sh`, `figure5_prio.sh` and `figure10_bfs.sh` under `bfs_nvlink` and `bfs_ib` directory and run `figure7_discrete.sh`, `figure7_persist.sh` and `figure11_pr.sh` under `pr_nvlink` and `pr_ib` directory.
