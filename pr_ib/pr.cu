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
#include "../comm/agent_maxcount.cuh"
#include "pr.cuh"

#include "validation.cuh"

//#define INTER_BATCH_SIZE (16)
//#define FETCH_SIZE (512)
#define BLOCK_SIZE (512)
#define PADDING_SIZE (32)
//#define WAIT_TIMES (32)

int main(int argc, char *argv[])
{
   //---------------------Pass from command ---------------// 
    char *input_file = NULL;
    int min_iter = -1;
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
	int wl_size_ratio = 8;
	uint32_t recv_size = 23;
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
            else if(std::string(argv[i]) == "-wl_size_ratio")
				wl_size_ratio = std::stoi(argv[i+1]);
			else if(std::string(argv[i]) == "-recv_size")
				recv_size = std::stoi(argv[i+1]);
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
        if(my_pe == 0) std::cout << "graph " <<  input_file << " partition scheme "<< partition_idx<< " iteration "<< min_iter << 
                        " num worklist "<< num_queue <<  " rounds "<< rounds << " WAIT_TIMES " << WAITTIMES <<
                        " INTER_BATCH_SIZE " << INTERBATCHSIZE << " FETCH SIZE "<< FETCHSIZE << " iteration ratio " << ratio <<std::endl;
        std::cout << "PE: "<< my_pe << " deviceCount " << dev_count << " set on device " << my_pe <<" device name " << prop.name << std::endl; 
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
    nvshmem_barrier_all();
   //----------------------------------------------------------------/
   //--------------------- initialize BFS ---------------------------/        
	PageRank<int, int, float, uint32_t, PADDING_SIZE> pr
        (my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, num_queue, my_csr.nodes*wl_size_ratio, (1<<recv_size), min_iter);
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, min_iter, num_queue, my_csr.nodes*16, 150000000);
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, min_iter, num_queue, my_csr.nodes*8, (1<<28));
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, num_queue, my_csr.nodes*8, (1<<26), min_iter);
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, num_queue, my_csr.nodes*8, (1<<27), min_iter);
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, num_queue, my_csr.nodes*16, (1<<23), min_iter);
        //(my_csr, my_pe, n_pes, group_id, group_size, local_id, local_size, partition_scheme, lambda, epsilon, min_iter, num_queue, my_csr.nodes*16, (1<<23));
	Atos::MAXCOUNT::Agent<PREntry<int,float>, uint32_t, INTERBATCHSIZE, PADDING_SIZE> agent(pr.worklists);
    if(verbose)
        SERIALIZE_PRINT(my_pe, n_pes, pr.print());
    
    cudaStream_t streams[35];
    cudaStream_t streams_high[32];
	int highest_pri, lowest_pri;
	CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowest_pri, &highest_pri));
	printf("highest prio %d, lowest prio %d\n", highest_pri, lowest_pri);
    for(int i=0; i<35; i++)
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
	for(int i=0; i<32; i++)
		CUDA_CHECK(cudaStreamCreateWithPriority(streams_high+i, cudaStreamNonBlocking, highest_pri));
	pr.worklists.setExecute(32, 0xffffffff);
    uint32_t *h_execute = (uint32_t *)malloc(sizeof(uint32_t)*32*n_pes);
    uint32_t *d_execute_copy = (uint32_t *)malloc(sizeof(uint32_t)*32*n_pes);
    int *check_finish;
    CUDA_CHECK(cudaMalloc(&check_finish, sizeof(int)*32*PADDING_SIZE));

    std::vector<float> times;
    std::vector<uint64_t> workloads;
   
    
    for(int round = 0; round < rounds; round++) {
	CUDA_CHECK(cudaMemset(check_finish, 0, sizeof(int)*32*PADDING_SIZE));
    CUDA_CHECK(cudaMemset(pr.worklists.execute, 0xffffffff, sizeof(uint32_t)*32*n_pes));
    for(int i=0; i<n_pes*32; i++)
    	h_execute[i] = 0xffffffff;

    pr.reset(0, 0xffffffff);
	agent.resetAgent();
	agent.launchAgent<128>(streams[33], NULL);
    pr.PageRankInit(streams[32]);
	agent.stopAgent(streams[32]);
	CUDA_CHECK(cudaStreamSynchronize(streams[32]));
    CUDA_CHECK(cudaStreamSynchronize(streams[33]));
	nvshmem_barrier_all();
    //SERIALIZE_PRINT(my_pe, n_pes, pr.worklists.print());
	pr.PageRankInitMerge(streams[32]);
	CUDA_CHECK(cudaStreamSynchronize(streams[32]));
    nvshmem_barrier_all();
   	SERIALIZE_PRINT(my_pe, n_pes, pr.worklists.print());
   //----------------------- warm up ---------------------------------/
    host::warmup_pr(pr);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

	agent.resetAgent(streams[32]);
	agent.launchAgent<WAITTIMES>(streams[32], NULL);
   //------------------------- start BFS ------------------------------/
	uint32_t temp_load[n_pes*PADDING_SIZE] = {0};
    uint32_t end_host[n_pes] = {0};
    uint32_t start_host[n_pes] = {0};
    uint32_t size[n_pes] = {0};
	uint32_t start_update[n_pes*PADDING_SIZE] = {0};
    int iter = 0;
    int log_iter = 0;
	int log_iter_high = 0;
    int stream_id = log_iter&31;
	int stream_id_high = 0;
	int stop_flag = 0;

	CUDA_CHECK(cudaMemcpyAsync(d_execute_copy, pr.worklists.execute,
                sizeof(uint32_t)*32*n_pes, cudaMemcpyDeviceToHost, streams[33]));

	CUDA_CHECK(cudaStreamSynchronize(streams[33]));
	for(int i=0; i<n_pes; i++){
		for(int s=0; s<32; s++)
			assert(d_execute_copy[i*32+s] == 0xffffffff);
	} 

    GpuTimer timer;
    nvshmem_barrier_all();
	    
    timer.Start();
    while((stop_flag != 3 && min_iter == -1) || (min_iter != -1 && iter < min_iter))
    {
		CUDA_CHECK(cudaMemcpyAsync(temp_load, pr.worklists.end,
                sizeof(uint32_t)*n_pes*PADDING_SIZE, cudaMemcpyDeviceToHost, streams[33]));
		CUDA_CHECK(cudaStreamSynchronize(streams[33]));
		CUDA_CHECK(cudaMemcpyAsync(d_execute_copy, pr.worklists.execute,
                sizeof(uint32_t)*32*n_pes, cudaMemcpyDeviceToHost, streams[33]));
        for(int i=0; i<n_pes; i++) end_host[i] = temp_load[i*PADDING_SIZE];
        for(int i=0; i<n_pes; i++) size[i] = end_host[i] - start_host[i];

		for(int p=1; p<n_pes; p++) {
			if(size[p] > 0) {
				if(stop_flag != 0) {
                    stop_flag = 0;
                    CUDA_CHECK(cudaMemcpyAsync(pr.worklists.stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[34]));
                }
            	//printf("pe %d, p %d recv start %8d, size: %6d\n", my_pe, p, start_host[p], size[p]);
				pr.PageRankMerge(start_host[p], size[p], streams_high[stream_id_high], p-1, stream_id_high, check_finish);
				h_execute[p*32+stream_id_high] = start_host[p]+size[p];
				log_iter_high++;
				stream_id_high = log_iter_high&31;
				start_host[p] = end_host[p];
				iter = 0;
			}
		}
        // if high priority local queue has items
        if(size[0] > 0) {
            //printf("pe %d, start %8d, size: %6d\n", my_pe, start_host[0], size[0]);
			if(stop_flag != 0) {
                stop_flag = 0;
                CUDA_CHECK(cudaMemcpyAsync(pr.worklists.stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[34]));
            }
            pr.PageRankStart<FETCHSIZE, BLOCK_SIZE>(start_host[0], size[0], 0, streams[stream_id]);
            log_iter++;
            stream_id = log_iter&31;
            start_host[0] = end_host[0];
            iter = 0;
        }

        //printf("pe %d, iter %d, log_iter %d, log_iter_high %d, stop %d\n", my_pe, iter, log_iter, log_iter_high, stop_flag);
		//if(log_iter_high > 5 && cudaStreamQuery(streams[34]) == cudaSuccess)
		//if((log_iter_high > 5 && cudaStreamQuery(streams[34]) == cudaSuccess) || (cudaStreamQuery(streams[34]) == cudaSuccess && log_iter <= 3 && cudaStreamQuery(streams[0]) == cudaSuccess))
		if(cudaStreamQuery(streams[34]) == cudaSuccess)
		{
        	pr.PushVertices(streams[34]);
			CUDA_CHECK(cudaMemcpyAsync(pr.worklists.end, pr.worklists.end_alloc, sizeof(uint32_t), cudaMemcpyDeviceToDevice, streams[34]));
			iter++;
			//printf("pe %d, launch a push kernel, iter %d, min_iter %d\n", my_pe, iter, min_iter);
		}

		CUDA_CHECK(cudaStreamSynchronize(streams[33]));
		bool ifupdate = false;
        for(int q = 1; q<n_pes; q++) {
            uint32_t min_value = 0xffffffff;
            bool valid_d = true;
            bool valid_h = false;
            bool ifUpdateToMax = true;
			if(log_iter_high < 32) {
				min_value = 0;
				for(int streamid = 0; streamid < 32; streamid++) {
					if(h_execute[q*32+streamid]!=0xffffffff && d_execute_copy[q*32+streamid]!=0xffffffff) 
					{
						min_value = d_execute_copy[q*32+streamid];
					}
					else break;
				}
				if(start_update[q*PADDING_SIZE] < min_value) {
					if(min_value > start_host[q])
						printf("PE %d, ERROR2 q %d, start update %d > start host %d\n", my_pe, q, min_value, start_host[q]);
					start_update[q*PADDING_SIZE] = min_value;
					ifupdate = true;
				}	
			}
			else 
			{
            	for(int streamid = 0; streamid < 32; streamid++) {
            	    if(h_execute[q*32+streamid] != 0xffffffff) {
            	        valid_h = true;
            	        if(d_execute_copy[q*32+streamid] == 0xffffffff) {
            	            valid_d = false;
            	        }
            	        else {
            	            if(start_host[q] == 0)
            	                printf("ERROR, PE %d, start_host[%d] %d, h_execute[%d*32+%d] %d!=(%d), d_execut_copy[%d*32+%d] %d\n", my_pe, q, start_host[q],
            	                    q, streamid, h_execute[q*32+streamid], 0xffffffff, q, streamid, d_execute_copy[q*32+streamid]);
            	            min_value = min(min_value, d_execute_copy[q*32+streamid]);
            	            if(d_execute_copy[q*32+streamid] != h_execute[q*32+streamid])
            	                ifUpdateToMax = false;
            	        }
            	    }
            	}
            	if(valid_d && valid_h && (start_update[q*PADDING_SIZE] < min_value || ifUpdateToMax))
            	{
            	    if(min_value == 0xffffffff) {
						for(int streamid=0; streamid < 32; streamid++)
						{
							if(h_execute[q*32+streamid] != 0xffffffff && d_execute_copy[q*32+streamid] != 0xffffffff)
								printf("PE %d, ERROR update q %d, start[%d] %d (device %d), min_value == 0xfffffff\n", my_pe, q, streamid, h_execute[q*32+streamid], d_execute_copy[q*32+streamid]);
						}
            	        printf("PE %d, ERROR update q %d, start, but min_value == 0xfffffff (%d)\n", my_pe, q, min_value);
					}
            	    if(min_value > start_host[q])
            	        printf("PE %d, ERROR q %d, start update %d > start host %d\n", my_pe, q, min_value, start_host[q]);
            	    if(ifUpdateToMax)
            	        min_value = start_host[q];
					if(start_update[q*PADDING_SIZE] < min_value) {
            	    	start_update[q*PADDING_SIZE] = min_value;
            	    	ifupdate = true;
					}
            	}
			}
        }
        if(ifupdate)
            CUDA_CHECK(cudaMemcpyAsync(pr.worklists.start, start_update, sizeof(uint32_t)*n_pes*PADDING_SIZE, cudaMemcpyHostToDevice, streams[33]));

		if(iter >= 5 && min_iter == -1) {
            //printf("pe %d, iter %d, log_iter %d, log_iter_high %d, stop %d\n", my_pe, iter, log_iter, log_iter_high, stop_flag);
			bool not_complete = false;
            for(int i=0; i<32; i++) {
                if(cudaStreamQuery(streams[i]) == cudaErrorNotReady || cudaStreamQuery(streams_high[i]) == cudaErrorNotReady) {
                    not_complete = true;
                    break;
                }
            }

			if(not_complete == false) {
                for(int q=0; q<pr.worklists.num_local_queues; q++)
                    CUDA_CHECK(cudaMemcpyAsync(pr.worklists.start+q*PADDING_SIZE, start_host+q, sizeof(uint32_t), cudaMemcpyHostToDevice, streams[33]));
                CUDA_CHECK(cudaMemcpyAsync(&stop_flag, pr.worklists.stop, sizeof(int), cudaMemcpyDeviceToHost, streams[33]));
                CUDA_CHECK(cudaStreamSynchronize(streams[33]));
                //printf("pe %d, stop_flag %d, recv_end %d, recv_tart %d\n", my_pe, stop_flag, end_host[1], start_host[1]);
				if(stop_flag == 3) break;
				else if(stop_flag == 0) {
					stop_flag = 1;
					CUDA_CHECK(cudaMemcpyAsync(pr.worklists.stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[33]));
				}
				if(my_pe == 0)
					pr.worklists.checkStop(streams[33], &stop_flag);
            }
		}
    }
    timer.Stop();
	agent.stopAgent(streams[33]);
	for(int i=0; i<32; i++) CUDA_CHECK(cudaStreamSynchronize(streams_high[i]));
    for(int i=0; i<35; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsed = timer.ElapsedMillis();
    SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f, %8d kernels, %8d kernels\n", elapsed, log_iter, log_iter_high));
	nvshmem_barrier_all();
    if(verbose) {
		for(int p=0; p<n_pes; p++)
			CUDA_CHECK(cudaMemcpy(pr.worklists.start+PADDING_SIZE*p, start_host+p, sizeof(uint32_t), cudaMemcpyHostToDevice));
        SERIALIZE_PRINT(my_pe, n_pes, pr.worklists.print());
    }
    times.push_back(elapsed);
	uint64_t totalwl=0;
	for(int p=0; p<n_pes; p++)
		totalwl+=end_host[p];
    workloads.push_back(totalwl);
    }
   
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size()));
    SERIALIZE_PRINT(my_pe, n_pes, printf("ave workload: %lld\n", (long long)(std::accumulate(workloads.begin(), workloads.end(), (long long)0)/workloads.size())));
   
   //----------------------------------------------------------------/
    
    nvshmem_barrier_all();
	if(ifcheck) {
        host::PrRes<int, float> res = host::PrValid(csr, pr, partition_idx, new_labels_old);
        for(int i=0; i<n_pes; i++) {
            if(my_pe == i)
            {
                std::cout << "[PE "<< my_pe << "]\n";
                //host::PrInitValid(csr, pr, partition_idx, new_labels_old);
                res.print(partition_idx, pr);
            }
            nvshmem_barrier_all();
        }
    }
   //----------------------------------------------------------------/
    nvshmem_barrier_all();
    SERIALIZE_PRINT(my_pe, n_pes,std::cout << "End program "<< my_pe << std::endl);


    nvshmem_barrier_all();
    csr.release();
    my_csr.release();
	nvshm_mpi_finalize();
    return 0;
}
