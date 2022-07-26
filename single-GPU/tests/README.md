#Performance Tests

### Compile
* To compile BFS: `make bfs_all` 
* To compile PageRank: `make pr_all` 
* To compile Graph Coloring: `make gc_all` 
* To compile all executables: `make all` 
* To remove all executables: `make clean` 


### Run Performance Test
* `run_bfs.batch` or `run_bfs.sh` to run performance test for single-GPU BFS
* `run_pr.batch` or `run_pr.sh` to run performance test for single-GPU PageRank
* `run_gc.batch` or `run_gc.sh` to run performance test for single-GPU Graph Coloring


NOTE: batch script uses slurm, please modify the slurm batch script accordingly to use the batch scripts