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
#include "../util/time.cuh"
#include "../util/nvshmem_util.cuh"

#include "../comm/csr.cuh"
#include "../comm/partition.cuh"
#include "pr_queue.cuh"

#include "validation.cuh"

//#define BATCH_SIZE (8192)
//#define FETCH_SIZE (512)
#define BLOCK_SIZE (512)

int main(int argc, char *argv[])
{
   //---------------------Pass from command ---------------// 
    char *input_file = NULL;
    uint32_t min_iter = 2500;
    int num_queue=1;
    int rtq_pe = 1;
    bool verbose = 0;
    float ratio = 0;
    int partition_idx = 0;
    int device = 0;
    char * metis_file=NULL;
    bool ifcheck=false;
    int rounds=1;
    float lambda = 0.85;
    float epsilon = 0.01;
    if(argc == 1)
    {
        std::cout<< "./test -file <file>  -r <runtime queue per pe=1> -iter <min iteration for queue=2500> -source <source node to start=0> \
         -q <number of queues used=1> -v <verbose=true> -ratio <ratio=0> \
         -partition <partition 0=vertex partition, 1=edge partition, 2=random partition, 3=metis> -d <start device=0>\n";
        exit(0);
    }
    if(argc > 1)
        for(int i=1; i<argc; i++) {
            if(std::string(argv[i]) == "-file")
               input_file = argv[i+1];
            else if(std::string(argv[i]) == "-lambda")
                lambda = std::stof(argv[i+1]);
            else if(std::string(argv[i]) == "-iter")
                min_iter = std::stoi(argv[i+1]);
            else if(std::string(argv[i]) == "-epsilon")
                epsilon = std::stof(argv[i+1]);
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
    assert(device+n_pes <= dev_count);
    CUDA_CHECK(cudaSetDevice(my_pe+device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, (device+my_pe)));
    if(verbose) {
        if(my_pe == 0) std::cout << "graph " <<  input_file << " partition scheme "<< partition_idx<< " iteration "<< min_iter << 
                        " num worklist "<< num_queue <<  " rounds "<< rounds << 
                        " FETCH SIZE "<< FETCHSIZE << " iteration ratio " << ratio <<std::endl;
        std::cout << "PE: "<< my_pe << " deviceCount " << dev_count << " set on device " << my_pe+device <<" device name " << prop.name << std::endl; 
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
    if(n_pes > 1) {
        if(partition_idx == 2) {
            CUDA_CHECK(cudaMallocManaged(&new_labels_old, sizeof(int)*csr.nodes));
            partitioner::random(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, 0);
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
                partitioner::metis(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, 0, (my_pe == 0), file_metis); 
            }
            else {
                std::cout << "read metis file: "<< file_metis << std::endl;
                partitioner::metis(n_pes, my_pe, csr, my_csr, new_labels_old, partition_scheme, 0, file_metis);
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
    }
    else {
        my_csr = csr;
        partition_scheme[0] = 0;
        partition_scheme[1] = csr.nodes;
    }
    nvshmem_barrier_all();
   //----------------------------------------------------------------/
   //--------------------- initialize BFS ---------------------------/        
    PageRank<int, int, float> pr(my_csr, my_pe, n_pes, partition_scheme, lambda, epsilon, min_iter, num_queue);
    if(verbose)
        SERIALIZE_PRINT(my_pe, n_pes, pr.print());
    
    cudaStream_t streams[34];
    for(int i=0; i<34; i++)
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));

    std::vector<float> times;
    std::vector<uint32_t> workloads;
   
    
    for(int round = 0; round < rounds; round++) {

    pr.reset();
    pr.PageRankInit();
    SERIALIZE_PRINT(my_pe, n_pes, pr.worklists.print());
    //SERIALIZE_PRINT(my_pe, n_pes, runtime.print_rtq());
    //SERIALIZE_PRINT(my_pe, n_pes, runtime.print_rq());
    nvshmem_barrier_all();
   //----------------------- warm up ---------------------------------/
    host::warmup_pr(pr);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
   //------------------------- start BFS ------------------------------/
   
    uint32_t end_host = 0;
    uint32_t start_host = 0;
    uint32_t size = 0;
    int end_iter = min_iter;
    if(ratio != 0 && my_pe == 1)
        end_iter = min_iter * ratio;
    int iter = 0;
    int log_iter = 0;
    int stream_id = log_iter&31;
    int * record;
    CUDA_CHECK(cudaMalloc(&record, sizeof(int)));
    CUDA_CHECK(cudaMemset(record, 0, sizeof(int)));

    float *check_res=(float *)malloc(sizeof(float)*pr.nodes);
    GpuTimer timer;
    nvshmem_barrier_all();
    
    timer.Start();
    while(iter < end_iter)
    //while(log_iter < 2)
    {
        CUDA_CHECK(cudaMemcpyAsync(&end_host, (uint32_t *)pr.worklists.end, sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[33]));
        CUDA_CHECK(cudaStreamSynchronize(streams[33]));
        size = end_host - start_host;

        // if high priority local queue has items
        if(size > 0) {
        //if(size[0] >= 256 || (iter>5 && size[0] > 0)) {
            //printf("iter %d, pe %d, start %8d, size: %6d\n", iter, my_pe, start_host, size);
            pr.PageRankStart<FETCHSIZE, BLOCK_SIZE>(start_host, size, 0, streams[stream_id], 0);
            log_iter++;
            stream_id = log_iter&31;
            start_host = end_host;
            iter = 0;
        }
        else {
            //printf("pe %d, iter %d\n", my_pe, iter);
            iter++;
        }

        pr.PushVertices(streams[33], record);
    }
    timer.Stop();
    for(int i=0; i<32; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsed = timer.ElapsedMillis();
    SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f, %8d kernels\n", elapsed, log_iter));
    if(verbose) {
        SERIALIZE_PRINT(my_pe, n_pes, printf("end:    %8d\tstart:  %8d\tsize:  %8d\n", end_host, start_host, size));
        SERIALIZE_PRINT(my_pe, n_pes, pr.worklists.print());
    }
    times.push_back(elapsed);
    workloads.push_back(end_host);
    }
   
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size()));
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave workload: %lld\n", std::accumulate(workloads.begin(), workloads.end(), (long long)(0))/workloads.size()));
   
   //----------------------------------------------------------------/
    

    nvshmem_barrier_all();
    if(ifcheck) {
		host::PrRes<int, float> res;
        if(n_pes > 1) {
            res = host::PrValid(csr, pr, partition_idx, new_labels_old);
            for(int i=0; i<n_pes; i++) {
                if(my_pe == i)
                {
                    std::cout << "[PE "<< my_pe << "]\n";
                    res.print(partition_idx, pr);
                    //host::PrInitValid(csr, pr, partition_idx, new_labels_old);
                }
                nvshmem_barrier_all();
            }
        }
        else
            host::PrValid(csr, pr);
    }
    
   //----------------------------------------------------------------/
    nvshmem_barrier_all();
    SERIALIZE_PRINT(my_pe, n_pes,std::cout << "End program "<< my_pe << std::endl);


    nvshmem_barrier_all();
    csr.release();
    if(n_pes > 1)
    my_csr.release();
    nvshm_mpi_finalize();
    return 0;
}
