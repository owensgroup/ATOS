#include <iostream>
#include <string>
#include <assert.h>
#include <unistd.h>

#define MPI_SUPPORT
#include "nvshmem.h"
#include "nvshmemx.h"
#ifdef MPI_SUPPORT
    #include "mpi.h"
#endif

#include "../util/util.cuh"
#include "../util/error_util.cuh"
#include "../util/nvshmem_util.cuh"
#include "../util/time.cuh"

#include "../comm/csr.cuh"
#include "../comm/partition.cuh"
#include "bfs_prio.cuh"
#include "../comm/agent_maxcount.cuh"

#include "validation.cuh"

//#define INTER_BATCH_SIZE (16)
//#define WAIT_TIMES (4)
//#define FETCH_SIZE (32)
#define BLOCK_SIZE (512)
#define PADDING_SIZE (32)

void print_info(uint32_t *end, uint32_t *start, uint32_t *size, int n)
{
    printf("end:");
    for(int i=0; i<n; i++)
        printf(" %8d", end[i]);
    printf("\n");
    printf("start:");
    for(int i=0; i<n; i++)
        printf(" %8d", start[i]);
    printf("\n");
    printf("size:");
    for(int i=0; i<n; i++)
        printf(" %8d", size[i]);
    printf("\n");
}

uint32_t sum(uint32_t *array, int n) {
    uint32_t total=0;
    for(int i=0; i<n; i++)
        total += array[i];
    return total;
}

void print_launch(int my_pe, int n_pes, int iter, int pivot, int threshold, uint32_t *end_host, uint32_t *start_host, uint32_t *size)
{
    if(n_pes == 1)
        printf("iter %4d, pivot %d, threshold %2d, end: %8d, %8d start: %8d, %8d size: %6d, %6d\n", 
        iter, pivot, threshold, end_host[0], end_host[1], start_host[0], start_host[1], size[0], size[1]);
    else if(n_pes ==2)
        printf("pe %d, iter %4d, pivot %d, threshold %2d, end: %8d, %8d, %8d start: %8d, %8d, %8d size: %6d, %6d, %6d\n",
        my_pe, iter, pivot, threshold, end_host[0], end_host[1], end_host[2],
        start_host[0], start_host[1], start_host[2], size[0], size[1], size[2]);
    if(n_pes == 3)
        printf("pe %d, iter %4d, pivot %d, threshold %2d, end: %8d, %8d, %8d, %8d, start: %8d, %8d, %8d, %8d, size: %6d, %6d, %6d, %6d\n", 
        my_pe, iter, pivot, threshold, end_host[0], end_host[1], end_host[2], end_host[3],
        start_host[0], start_host[1], start_host[2], start_host[3], size[0], size[1], size[2], size[3]);
    if(n_pes == 4)
        printf("pe %d, iter %4d, pivot %d, threshold %2d, end: %8d, %8d, %8d, %8d, %8d start: %8d, %8d, %8d, %8d, %8d size: %6d, %6d, %6d, %6d, %6d\n", 
        my_pe, iter, pivot, threshold, end_host[0], end_host[1], end_host[2], end_host[3], end_host[4],
        start_host[0], start_host[1], start_host[2], start_host[3], start_host[4], size[0], size[1], size[2], size[3], size[4]);
}

int main(int argc, char *argv[])
{
   //---------------------Pass from command ---------------// 
    char *input_file = NULL;
    int min_iter = -1;
    int source = 0;
    int option = 1;
    int num_queue=1;
    int rtq_pe = 1;
    bool verbose = 0;
    float ratio = 0;
    int partition_idx = 0;
    int device = 0;
    char * metis_file=NULL;
    bool ifcheck=false;
    int rounds=1;
    int threshold = 3;
    int threshold_increment = 2;
    if(argc == 1)
    {
        std::cout<< "./test -file <file>  -r <runtime queue per pe=1> -iter <min iteration for queue=2500> -source <source node to start=0> \
         -q <number of queues used=1> -v <verbose=true> -ratio <ratio=0> \
         -threshold <threshold for depth> -delta <threshold increment value> \
         -partition <partition 0=vertex partition, 1=edge partition, 2=random partition, 3=metis> -d <start device=0>\n";
        exit(0);
    }
    if(argc > 1)
        for(int i=1; i<argc; i++) {
            if(std::string(argv[i]) == "-file")
               input_file = argv[i+1];
            else if(std::string(argv[i]) == "-source")
                source = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-iter")
                min_iter = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-o")
                option = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-r")
                rtq_pe = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-q")
                num_queue= std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-v")
                verbose= std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-ratio")
                ratio = std::stof(argv[i+1]);
            else if(std::string(argv[i]) == "-partition")
                partition_idx = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-d")
                device = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-fmetis")
                metis_file = argv[i+1];
            else if(std::string(argv[i]) == "-check")
                ifcheck = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-rounds")
                rounds = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-threshold")
                threshold = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-delta")
                threshold_increment = std::stoi(argv[i+1]);
        }
    if(input_file == NULL)
    {
        std::cout << "input file is needed\n";
        std::cout<< "./test -f <file>  -r <runtime queue per pe=1> -i <min iteration for queue=2500> -s <source node to start=0> \
        -q <number of queues used=1> -v <verbose=true> -a <ratio=3> \
        -m <partition 0=vertex partition, 1=edge partition, 2=random partition, 3=metis> -d <start device=0>\n";
        exit(0);
    }

   //-------------------- initialize nvshmem environment ------------/
	int n_pes, my_pe, group_id, group_size, local_id, local_size;
    nvshm_mpi_init(my_pe, n_pes, group_id, group_size, local_id, local_size, &argc, &argv);

   	cudaDeviceProp prop;
    int dev_count;

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, (my_pe%dev_count)));
    if(verbose) {
        if(my_pe == 0) std::cout << "graph " <<  input_file << " partition scheme "<< partition_idx<< " iteration "<< min_iter << " source "<< source <<
                        " threshold "<< threshold << " threshold increment value " << threshold_increment << " rounds "<< rounds << 
                        " FETCH SIZE "<< FETCHSIZE << " BLOCK SIZE " << BLOCK_SIZE << 
						" WAIT TIMES " << WAITTIMES << " INTER BATCH SIZE "<< INTERBATCHSIZE << " iteration ratio " << ratio <<std::endl;
        std::cout << "PE: "<< my_pe << " deviceCount " << dev_count << " set on device " << my_pe%dev_count<<" device name " << prop.name << std::endl; 
    }
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice));
   //----------------------------------------------------------------/
   
   //-------------------- Read CSR and partition --------------------/
    std::string str_file(input_file);
    Csr<int, int> csr;
    if(str_file.substr(str_file.length()-4) == ".csr") {
        csr.ReadFromBinary(input_file);
    }
    else {
        std::cout << "Generate csr file binary file first\n";
        exit(1);
    }
    if(my_pe == 0) csr.PrintCsr();
    nvshmem_barrier_all();
    Csr<int, int> my_csr;
    int partition_scheme[n_pes+1];
    int * new_labels_old;
    int new_source = source;
    if(partition_idx == 2) {
        CUDA_CHECK(cudaMallocManaged(&new_labels_old, sizeof(int)*csr.nodes));
        new_source = partitioner::random(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, source);
    }
    else if(partition_idx == 0)
        partitioner::vertices(n_pes, my_pe, csr, my_csr, partition_scheme);
    else if(partition_idx == 1)
        partitioner::edges(n_pes, my_pe, csr, my_csr, partition_scheme);
    else if(partition_idx == 3) {
        CUDA_CHECK(cudaMallocManaged(&new_labels_old, sizeof(int)*csr.nodes));
        char file_metis[256];
        std::string file_name = str_file.substr(0, str_file.length()-4);
        sprintf(file_metis, "%s_%d_metis_mega.txt", file_name.c_str(), n_pes);
        
        if(exists_file(file_metis) == false) {
            std::cout << "didn't find file: "<< file_metis << std::endl;
            new_source = partitioner::metis(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, source, (my_pe == 0), file_metis); 
        }
        else {
            std::cout << "read metis file: "<< file_metis << std::endl;
            new_source = partitioner::metis(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, source, file_metis);
        }
    }

    if(verbose) {
        SERIALIZE_PRINT(my_pe, n_pes, my_csr.PrintCsr());
        if(my_pe == 0) {
            std::cout << "Partition table:\n";
            for(int i=0; i<n_pes+1; i++)
                std::cout << partition_scheme[i] << "  ";
            std::cout <<std::endl;
        }
    }
    nvshmem_barrier_all();
   //----------------------------------------------------------------/
   //--------------------- initialize BFS ---------------------------/        
    BFS<int, int, uint32_t, PADDING_SIZE> bfs(my_csr, my_pe, n_pes, group_id, group_size, local_id, 
		local_size, partition_scheme, threshold, threshold_increment, 1.5*my_csr.nodes, (1<<25), min_iter);
	Atos::MAXCOUNT::Agent<BFSEntry<int>, uint32_t, INTERBATCHSIZE, PADDING_SIZE> agent(bfs.worklists);
    if(verbose)
        SERIALIZE_PRINT(my_pe, n_pes, bfs.print());

    printf("PE %d, new_source %d\n", my_pe, new_source);
	cudaStream_t agg_stream;
	CUDA_CHECK(cudaStreamCreateWithFlags(&agg_stream, cudaStreamNonBlocking));
	
    std::vector<float> times;
    std::vector<uint32_t> workloads;
    
    for(int round = 0; round < rounds; round++) {

    	bfs.reset();
    	agent.resetAgent();
    	CUDA_CHECK(cudaDeviceSynchronize());
    	bfs.BFSInit(new_source);
    	//SERIALIZE_PRINT(my_pe, n_pes, bfs.worklists.print());
    	nvshmem_barrier_all();
    
   	   //----------------------- warm up ---------------------------------/
   	    warmup_bfs(bfs);
   	    CUDA_CHECK(cudaDeviceSynchronize());
    	agent.launchAgent<WAITTIMES>(agg_stream, NULL);
   	    nvshmem_barrier_all();
   	   //------------------------- start BFS ------------------------------/
		Atos::MAXCOUNT::res_info<uint32_t> res = bfs.BFSStart<FETCHSIZE, BLOCK_SIZE>(threshold, threshold_increment, 0);
        agent.stopAgent(bfs.worklists.streams[0]);
    	CUDA_CHECK(cudaStreamSynchronize(agg_stream));

    	CUDA_CHECK(cudaDeviceSynchronize());
    	if(verbose) {
    	    SERIALIZE_PRINT(my_pe, n_pes, bfs.worklists.print());
    	}
    	times.push_back(res.elapsed_ms);
    	workloads.push_back(res.workload);
    	}
   
    	SERIALIZE_PRINT(my_pe, n_pes, printf("ave time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size()));
    	SERIALIZE_PRINT(my_pe, n_pes, printf("ave workload: %lld\n", std::accumulate(workloads.begin(), workloads.end(), (unsigned long long)(0))/workloads.size()));
   
   //----------------------------------------------------------------/
   
    nvshmem_barrier_all();
    if(ifcheck)
    for(int i=0; i<n_pes; i++) {
        if(my_pe == i)
        {
            std::cout << "[PE "<< my_pe << "]\n";
            host::BFSValid<int, int>(csr, bfs, source, partition_idx, new_labels_old);
        }
        nvshmem_barrier_all();
    }
   //----------------------------------------------------------------/
    nvshmem_barrier_all();
    SERIALIZE_PRINT(my_pe, n_pes,std::cout << "End program "<< my_pe << std::endl);


    nvshmem_barrier_all();
    csr.release();
    my_csr.release();
    nvshmem_finalize();
    #ifdef MPI_SUPPORT
        MPI_Finalize();
    #endif
    return 0;
}
