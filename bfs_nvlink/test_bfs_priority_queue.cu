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
#include "bfs_priority_queue.cuh"

#include "validation.cuh"

//#define FETCH_SIZE (32)
#define BLOCK_SIZE (512)

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

void print_info(uint32_t *data, int n)
{
    printf("Send:");
    for(int i=0; i<n; i++)
        printf(" %8d", data[i]);
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
    uint32_t min_iter = 10;
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
    int threshold_increment = 1;
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
    assert(device+n_pes <= dev_count);
    CUDA_CHECK(cudaSetDevice(my_pe+device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, (device+my_pe)));
    if(verbose) {
        if(my_pe == 0) std::cout << "graph " <<  input_file << " partition scheme "<< partition_idx<< " iteration "<< min_iter << " source "<< source <<
                        " threshold "<< threshold << " threshold increment value " << threshold_increment << " rounds "<< rounds << 
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
    else if(n_pes == 1) {
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
    BFS<int, int, int> bfs(my_csr, my_pe, n_pes, partition_scheme, threshold, threshold_increment, min_iter, 0);
    if(verbose)
        SERIALIZE_PRINT(my_pe, n_pes, bfs.print());
    
    cudaStream_t streams[34];
    for(int i=0; i<34; i++)
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));

    std::vector<float> times;
    std::vector<uint32_t> workloads;
    
    for(int round = 0; round < rounds; round++) {

    bfs.reset();
    bfs.BFSInit(new_source, 160);
    nvshmem_barrier_all();
    
   //----------------------- warm up ---------------------------------/
    warmup_bfs(bfs);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
   //------------------------- start BFS ------------------------------/
    uint32_t end_host[n_pes+1] = {0};
    uint32_t start_host[n_pes+1] = {0};
    uint32_t size[n_pes+1] = {0};
    int end_iter = min_iter;
    if(ratio != 0 && !(new_source >= partition_scheme[my_pe] && new_source < partition_scheme[my_pe+1]))
        end_iter = min_iter * ratio;
    int iter = 0;
    int log_iter = 0;
    int threshold_temp = threshold;
    MaxCountQueue::Priority pivot = MaxCountQueue::Priority::HIGH;
    int stream_id = log_iter&31;
    int * record;
    CUDA_CHECK(cudaMalloc(&record, sizeof(int)));
    CUDA_CHECK(cudaMemset(record, 0, sizeof(int)));

    GpuTimer timer;
    nvshmem_barrier_all();
    
    timer.Start();
    while(iter < end_iter)
    //while(log_iter < end_iter)
    {
        if(n_pes >= 2)
        CUDA_CHECK(cudaMemcpyAsync(end_host+2, bfs.worklists.recv_read_local_end, 
            sizeof(uint32_t)*(n_pes-1), cudaMemcpyDeviceToHost, streams[32]));
        CUDA_CHECK(cudaMemcpyAsync(end_host, (uint32_t *)bfs.worklists.end, sizeof(uint32_t)*2, cudaMemcpyDeviceToHost, streams[33]));
        CUDA_CHECK(cudaStreamSynchronize(streams[32]));
        CUDA_CHECK(cudaStreamSynchronize(streams[33]));
        for(int q_id=0; q_id < n_pes+1; q_id++)
            size[q_id] = end_host[q_id] - start_host[q_id];
        bool StartBatch = (sum(start_host, n_pes+1) >= 0.9*my_csr.nodes);
        #ifdef PROFILE
        print_launch(my_pe, n_pes, iter, int(pivot), threshold_temp, end_host, start_host, size);
        #endif
        //printf("pe %d, iter %4d, pivot %d, threshold %2d, end: %8d, %8d, %8d, %8d, start: %8d, %8d, %8d, %8d, size: %6d, %6d, %6d, %6d\n", my_pe, iter, pivot, threshold_temp, end_host[0], end_host[1], end_host[2], end_host[3],
        //start_host[0], start_host[1], start_host[2], start_host[3], size[0], size[1], size[2], size[3]);
        //if(size[0]+size[1]+size[2] > 0 && my_pe == 0)
        //printf("iter %2d, pe %d, threshold %d, start batch %d, pivot %d, size: %6d, %6d, %6d\n", 
        //iter, my_pe, threshold_temp, StartBatch, pivot, size[0], size[1], size[2]);

        // if high priority local queue has items
        if(size[pivot] > 0) {
            bfs.BFSStart<FETCHSIZE, BLOCK_SIZE>
                (start_host[pivot], size[pivot], threshold_temp, pivot, 0, 0, streams[stream_id], record);
            log_iter++;
            stream_id = log_iter&31;
            start_host[pivot] = end_host[pivot];
            iter = 0;
        }
        else if(size[!pivot] > 0) {
            pivot = static_cast<MaxCountQueue::Priority>(!pivot);
            threshold_temp += (threshold_increment+StartBatch*3);
            bfs.BFSStart<FETCHSIZE, BLOCK_SIZE>
                (start_host[pivot], size[pivot], threshold_temp, pivot, 0, 0, streams[stream_id], record);
            log_iter++;
            stream_id = log_iter&31;
            start_host[pivot] = end_host[pivot];
            iter = 0;
        }
        for(int recv_id = 2; recv_id < n_pes+1; recv_id++) {
            if(size[recv_id] > 0) {
                bfs.BFSStart<FETCHSIZE, BLOCK_SIZE>
                    (start_host[recv_id], size[recv_id], threshold_temp, pivot, recv_id-1, 0, streams[stream_id], record);
                log_iter++;
                stream_id = log_iter&31;
                start_host[recv_id] = end_host[recv_id];
                iter = 0;
            }
        }
		if(iter != 0) updateEnd<<<1,2, 0, streams[33]>>>(bfs.worklists.DeviceObject());
        iter++;
    }
    timer.Stop();
    for(int i=0; i<32; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsed = timer.ElapsedMillis();
    SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f, %8d kernels\n", elapsed, log_iter));
    int check_record=0;
    #ifdef PROFILE
    CUDA_CHECK(cudaMemcpy(&check_record, record, sizeof(int), cudaMemcpyDeviceToHost));
    uint32_t push_size[80];
    uint32_t push_times[80];
    CUDA_CHECK(cudaMemcpy(push_size, bfs.worklists.execute, sizeof(uint32_t)*80, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(push_times, bfs.worklists.reserve, sizeof(uint32_t)*80, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for(int i=0; i<80; i++)
        if(push_times[i] > 0)
            printf("%4d ", push_size[i]/push_times[i]);
    printf("\n");
    #endif
    if(verbose) {
        SERIALIZE_PRINT(my_pe, n_pes, print_info(end_host, start_host, size, n_pes+1));
        SERIALIZE_PRINT(my_pe, n_pes, bfs.worklists.print());
        uint32_t check_send[n_pes-1];
        CUDA_CHECK(cudaMemcpy(check_send, bfs.worklists.send_remote_alloc_end, sizeof(uint32_t)*(n_pes-1), cudaMemcpyDeviceToHost));
        SERIALIZE_PRINT(my_pe, n_pes, print_info(check_send, n_pes-1));
        SERIALIZE_PRINT(my_pe, n_pes, printf("total: %8d\n", sum(end_host, n_pes+1)-check_record));
        #ifdef PROFILE
        SERIALIZE_PRINT(my_pe, n_pes, printf("PE %d, record %d\n", my_pe, check_record));
        #endif
    }
    times.push_back(elapsed);
    workloads.push_back(sum(end_host, n_pes+1)-check_record);
    }
   
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size()));
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave workload: %lld\n", std::accumulate(workloads.begin(), workloads.end(), (long long)(0))/workloads.size()));
   
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
