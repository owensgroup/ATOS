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
#include "bfs_queue.cuh"

#include "validation.cuh"

//#define FETCH_SIZE (32)
#define BLOCK_SIZE (512)

uint32_t sum(uint32_t *array, int n) {
    uint32_t total=0;
    for(int i=0; i<n; i++)
        total += array[i];
    return total;
}

void print_recv(int my_pe, uint32_t *array, int n) {
    printf("pe %d receive:", my_pe);
    for(int i=0; i<n; i++)
        printf(" %6d", array[i]);
    printf("\n");
}

void print_send(int my_pe, uint32_t *array, int n) {
    printf("pe %d Send:", my_pe);
    for(int i=0; i<n; i++)
        printf(" %6d", array[i]);
    printf("\n");
}

int main(int argc, char *argv[])
{
   //---------------------Pass from command ---------------// 
    char *input_file = NULL;
    uint32_t min_iter = 2500;
    int source = 0;
    int option = 1;
    int num_queue=1;
    int rtq_pe = 1;
    bool verbose = 0;
    float ratio = 1;
    int partition_idx = 0;
    int device = 0;
    char * metis_file=NULL;
    bool ifcheck=false;
    int rounds=1;
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
        if(my_pe == 0) std::cout << "graph " <<  input_file << " partition scheme "<< partition_idx<< " iteration "<< min_iter << " source "<< source <<
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
    int new_source = source;
    if(n_pes >= 2) {
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
    }
    else {
        partition_scheme[0] = 0;
        partition_scheme[1] = csr.nodes;
        my_csr = csr;
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
    if(!(new_source >= partition_scheme[my_pe] && new_source < partition_scheme[my_pe+1]))
        min_iter = min_iter*ratio;
    BFS<int, int> bfs(my_csr, my_pe, n_pes, partition_scheme, min_iter, num_queue);
    if(verbose)
        SERIALIZE_PRINT(my_pe, n_pes, bfs.print());
    for(int i=0; i<num_queue; i++){
        CUDA_CHECK(cudaMalloc(&(bfs.worklists.worklist[i].reserve), sizeof(uint32_t)*(n_pes-1))); 
    }
    printf("PE %d, new_source %d\n", my_pe, new_source);

    std::vector<float> times;
    std::vector<uint32_t> workloads;

    int * record;
    CUDA_CHECK(cudaMalloc(&record, sizeof(int)));
    CUDA_CHECK(cudaMemset(record, 0, sizeof(int)));
    
    for(int round = 0; round < rounds; round++) {

        for(int i=0; i<num_queue; i++)
            CUDA_CHECK(cudaMemset(bfs.worklists.worklist[i].reserve, 0xffffffff, sizeof(uint32_t)*(n_pes-1)));
        bfs.reset();
        bfs.BFSInit(new_source, 160, 1024);
        //SERIALIZE_PRINT(my_pe, n_pes, bfs.worklists.print());
        nvshmem_barrier_all();
    
       //----------------------- warm up ---------------------------------/
        warmup_bfs(bfs);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
       //------------------------- start BFS ------------------------------/
        GpuTimer timer;
        nvshmem_barrier_all();
    
        timer.Start();
        bfs.BFSStart_persistent<FETCHSIZE>(160, BLOCK_SIZE, 0);
        bfs.worklists.sync_all_wl();
        timer.Stop();

        CUDA_CHECK(cudaDeviceSynchronize());
        float elapsed = timer.ElapsedMillis();
        SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f\n", elapsed));
        uint32_t check_record[n_pes-1];
        CUDA_CHECK(cudaMemcpy(check_record, bfs.worklists.worklist[0].reserve, sizeof(int)*(n_pes-1), cudaMemcpyDeviceToHost));
        SERIALIZE_PRINT(my_pe, n_pes, print_recv(my_pe, check_record, n_pes-1));
        uint32_t totalworkload=0;
        CUDA_CHECK(cudaMemcpy(&totalworkload, (uint32_t *)(bfs.worklists.end), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        totalworkload = totalworkload + sum(check_record, n_pes-1);
        if(verbose) {
            SERIALIZE_PRINT(my_pe, n_pes, bfs.worklists.print());
            CUDA_CHECK(cudaMemcpy(check_record, bfs.worklists.send_remote_alloc_end, sizeof(uint32_t)*(n_pes-1), cudaMemcpyDeviceToHost));
            SERIALIZE_PRINT(my_pe, n_pes, print_send(my_pe, check_record, n_pes-1));
        }
        times.push_back(elapsed);
        workloads.push_back(totalworkload);
    }
   
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size()));
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave workload: %d\n", int(std::accumulate(workloads.begin(), workloads.end(), 0)/workloads.size())));
   
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
    if(n_pes >= 2) my_csr.release();
	nvshm_mpi_finalize();
    return 0;
}
