CC=gcc
CUDACC=${CUDA_HOME}/bin/nvcc

CUDACFLAGS=-O3 -lineinfo --expt-extended-lambda --std=c++14 -Xptxas="-v" -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp
LDFLAGS =-lcuda -lgomp 

BFS_DIR = ../bfs
PR_DIR = ../pr
GC_DIR = ../gc

#all: bfs bfs_cta bfs_cta_discrete pr pr_cta pr_cta_discrete
#pr_all: pr pr_cta pr_cta_discrete
#bfs_all: bfs bfs_cta bfs_cta_discrete

FETCHSIZE1 = 32
FETCHSIZE2 = 64
FETCHSIZE3 = 128
FETCHSIZE4 = 256
NROUNDS_twitter = 10
NROUNDS_osm = 20

all: bfs_${FETCHSIZE1} bfs_cta_${FETCHSIZE1} bfs_cta_discrete_${FETCHSIZE1} bfs_${FETCHSIZE2} bfs_cta_${FETCHSIZE2} bfs_cta_discrete_${FETCHSIZE2} bfs_${FETCHSIZE3} bfs_cta_${FETCHSIZE3} bfs_cta_discrete_${FETCHSIZE3} pr pr_cta_${FETCHSIZE1} pr_cta_${FETCHSIZE2} pr_cta_${FETCHSIZE3} pr_cta_${FETCHSIZE4} pr_cta_${FETCHSIZE4}_${NROUNDS_twitter} pr_cta_${FETCHSIZE4}_${NROUNDS_osm} pr_cta_discrete_${FETCHSIZE1} pr_cta_discrete_${FETCHSIZE2} pr_cta_discrete_${FETCHSIZE3} pr_cta_discrete_${FETCHSIZE4} gc_${FETCHSIZE1} gc_${FETCHSIZE3} gc_${FETCHSIZE4}

bfs_all: bfs_${FETCHSIZE1} bfs_cta_${FETCHSIZE1} bfs_cta_discrete_${FETCHSIZE1} bfs_${FETCHSIZE2} bfs_cta_${FETCHSIZE2} bfs_cta_discrete_${FETCHSIZE2} bfs_${FETCHSIZE3} bfs_cta_${FETCHSIZE3} bfs_cta_discrete_${FETCHSIZE3} 

pr_all: pr pr_cta_${FETCHSIZE1} pr_cta_${FETCHSIZE2} pr_cta_${FETCHSIZE3} pr_cta_${FETCHSIZE4} pr_cta_${FETCHSIZE4}_${NROUNDS_twitter} pr_cta_${FETCHSIZE4}_${NROUNDS_osm} pr_cta_discrete_${FETCHSIZE1} pr_cta_discrete_${FETCHSIZE2} pr_cta_discrete_${FETCHSIZE3} pr_cta_discrete_${FETCHSIZE4}

gc_all: gc_${FETCHSIZE1} gc_${FETCHSIZE3} gc_${FETCHSIZE4}
	
bfs_fs_32: bfs_${FETCHSIZE1} bfs_cta_${FETCHSIZE1} bfs_cta_discrete_${FETCHSIZE1} 

pr_fs_32: pr_${FETCHSIZE1} pr_cta_${FETCHSIZE1} pr_cta_discrete_${FETCHSIZE1}

bfs_fs_64: bfs_${FETCHSIZE2} bfs_cta_${FETCHSIZE2} bfs_cta_discrete_${FETCHSIZE2} 

pr_fs_64: pr_${FETCHSIZE2} pr_cta_${FETCHSIZE2} pr_cta_discrete_${FETCHSIZE2}

bfs_fs_128: bfs_${FETCHSIZE3} bfs_cta_${FETCHSIZE3} bfs_cta_discrete_${FETCHSIZE3} 

pr_fs_128: pr_${FETCHSIZE3} pr_cta_${FETCHSIZE3} pr_cta_discrete_${FETCHSIZE3}


bfs_${FETCHSIZE1}: $(BFS_DIR)/test_bfs.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1}

bfs_cta_${FETCHSIZE1}: $(BFS_DIR)/test_bfs_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1}

bfs_cta_discrete_${FETCHSIZE1}: $(BFS_DIR)/test_bfs_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1}


bfs_${FETCHSIZE2}: $(BFS_DIR)/test_bfs.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE2}

bfs_cta_${FETCHSIZE2}: $(BFS_DIR)/test_bfs_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE2}

bfs_cta_discrete_${FETCHSIZE2}: $(BFS_DIR)/test_bfs_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE2}


bfs_${FETCHSIZE3}: $(BFS_DIR)/test_bfs.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3}

bfs_cta_${FETCHSIZE3}: $(BFS_DIR)/test_bfs_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3}

bfs_cta_discrete_${FETCHSIZE3}: $(BFS_DIR)/test_bfs_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3}



pr: $(PR_DIR)/test_pr.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS)

pr_cta_${FETCHSIZE1}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1} -DNROUNDS=1

pr_cta_${FETCHSIZE2}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE2} -DNROUNDS=1

pr_cta_${FETCHSIZE3}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3} -DNROUNDS=1

pr_cta_${FETCHSIZE4}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE4} -DNROUNDS=1

pr_cta_${FETCHSIZE4}_${NROUNDS_twitter}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE4} -DNROUNDS=${NROUNDS_twitter}

pr_cta_${FETCHSIZE4}_${NROUNDS_osm}: $(PR_DIR)/test_pr_cta.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE4} -DNROUNDS=${NROUNDS_osm}

pr_cta_discrete_${FETCHSIZE1}: $(PR_DIR)/test_pr_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1}

pr_cta_discrete_${FETCHSIZE2}: $(PR_DIR)/test_pr_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE2}

pr_cta_discrete_${FETCHSIZE3}: $(PR_DIR)/test_pr_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3}

pr_cta_discrete_${FETCHSIZE4}: $(PR_DIR)/test_pr_discrete.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE4}

gc_${FETCHSIZE1}: $(GC_DIR)/test_gc.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE1}

gc_${FETCHSIZE4}: $(GC_DIR)/test_gc.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE4}

gc_${FETCHSIZE3}: $(GC_DIR)/test_gc.cu
	$(CUDACC) -o $@ $^ $(CUDACFLAGS) -DFETCHSIZE=${FETCHSIZE3}



clean:
	rm -fv bfs_${FETCHSIZE1} bfs_cta_${FETCHSIZE1} bfs_cta_discrete_${FETCHSIZE1} pr pr_cta_${FETCHSIZE1} bfs_${FETCHSIZE2} bfs_cta_${FETCHSIZE2} bfs_cta_discrete_${FETCHSIZE2} pr_cta_${FETCHSIZE2} bfs_${FETCHSIZE3} bfs_cta_${FETCHSIZE3} bfs_cta_discrete_${FETCHSIZE3} pr_cta_${FETCHSIZE3} pr_cta_${FETCHSIZE4} pr_cta_${FETCHSIZE4}_${NROUNDS_twitter} pr_cta_${FETCHSIZE4}_${NROUNDS_osm} pr_cta_discrete_${FETCHSIZE1} pr_cta_discrete_${FETCHSIZE2} pr_cta_discrete_${FETCHSIZE3} pr_cta_discrete_${FETCHSIZE4} gc_${FETCHSIZE1} gc_${FETCHSIZE3} gc_${FETCHSIZE4}
