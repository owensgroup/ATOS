#include <iostream>
#include <string>
#include <numeric>

#include "../../comm/csr.cuh"
#include "bfs_cta_discrete.cuh"

#include "validation_bfs.cuh"
#include "../../util/time.cuh"
#include "profile.cuh"

#define FETCH_SIZE (FETCHSIZE)
#define BLOCK_SIZE (512)

using namespace std;

template<typename VertexId, typename SizeT, typename QueueT>
__global__ void warpup_mallocManaged(BFS<VertexId, SizeT, QueueT> bfs, uint32_t *out)
{
    uint32_t sum = 0;
    for(int i=TID; i<bfs.nodes+1; i=i+gridDim.x*blockDim.x)
    {
        SizeT node = bfs.csr_offset[i];
        sum =sum+node;
    }
    for(int i=TID; i<bfs.edges; i=i+gridDim.x*blockDim.x)
    {
        VertexId node = bfs.csr_indices[i]; 
        sum = sum+node;
    }
    
    out[TID] = sum;
}

struct prof_entry {
    uint32_t start;
    uint32_t end;
    int stream_id;
    uint32_t size;
    prof_entry() {}
    prof_entry(uint32_t start_, uint32_t end_, int id, uint32_t size_){
	start =start_;
	end = end_;
	stream_id = id;
	size = size_;
    }
};

int main(int argc, char *argv[])
{
     char *input_file = NULL;
     bool start_from_0 = false;
     bool write_profile = false;
     uint32_t min_iter = 2500;
     int source = 0;
     int option = 1;
     int num_queue=4;
     int device = 0;
     int rounds=1;
     if(argc == 1)
     {
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -r <source node to start=0> -d <device id=0>\n";
         exit(0);
     }
     if(argc > 1)
         for(int i=1; i<argc; i++) {
             if(string(argv[i]) == "-f")
                input_file = argv[i+1];
             else if(string(argv[i]) == "-s")
                 start_from_0 = stoi(argv[i+1]);
             else if(string(argv[i]) == "-w")
                 write_profile = stoi(argv[i+1]);
             else if(string(argv[i]) == "-i")
                 min_iter = stoi(argv[i+1]);
             else if(string(argv[i]) == "-o")
                 option = stoi(argv[i+1]);
             else if(string(argv[i]) == "-r")
                 source = stoi(argv[i+1]);
             else if(string(argv[i]) == "-q")
                 num_queue= stoi(argv[i+1]);
             else if(string(argv[i]) == "-d")
                 device= stoi(argv[i+1]);
            else if(string(argv[i]) == "-rounds")
                rounds = stoi(argv[i+1]);
         }
     if(input_file == NULL)
     {
         cout << "input file is needed\n";
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -r <source node to start=0> -d <device id=0>\n";
         exit(0);
     }

    std::cout << "set on device "<< device << std::endl;
    CUDA_CHECK(cudaSetDevice(device));

    int numBlock = 56*5;
    int numThread = 256;
    std::cout << "file: "<< input_file << " start from 0: " << start_from_0 << " write profile file: "<< write_profile << " " << " min iter "<< min_iter<< " Fetch size "<< FETCH_SIZE << " Block size "<< BLOCK_SIZE << std::endl;
    std::string str_file(input_file);
    Csr<int, int> csr;
    if(str_file.substr(str_file.length()-4) == ".csr")
    {
        csr.ReadFromBinary(input_file);
    }
    else { std::cout  <<  "file type not supported\n"; exit(1);}
    csr.PrintCsr();

    GpuTimer timer;
    prof_entry *profile_output;
    uint32_t profile_size = 3000000;
    cudaEvent_t *event_start;
    cudaEvent_t *event_stop;
    cudaEvent_t event_begin[32];
    int event_counter[32];
    if(write_profile)
    {
      profile_output = (prof_entry *)malloc(sizeof(prof_entry)*profile_size);
      event_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t)*32*1000);
      event_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t)*32*1000);
      for(int i=0; i<32*1000; i++) {
         cudaEventCreate(event_start+i);
         cudaEventCreate(event_stop+i);
	 if(i<32) {
	    event_counter[i]=0;
      	    cudaEventCreate(event_begin+i);
	 }
      }
    }
    
    uint32_t *warmup_out;
    CUDA_CHECK(cudaMalloc(&warmup_out, sizeof(uint32_t)*numBlock*numThread));
    BFS<int, int> bfs(csr, min_iter);
    warpup_mallocManaged<<<numBlock, numThread>>>(bfs, warmup_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "source "<< source << std::endl;

    cudaStream_t streams[33];
    for(int i=0; i<33; i++)
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
    if(write_profile)
        for(int i=0; i<32; i++)
            CUDA_CHECK(cudaEventRecord(event_begin[i], streams[i]));
    int write_file_log_iter=0;
    
    std::vector<float> times;
    std::vector<uint64_t> workloads;
    std::vector<uint64_t> launched_kernels;
    #ifdef COMPUTE_EDGES
        std::vector<uint64_t> workload_edges;
    #endif

    for(int iteration=0; iteration < rounds; iteration++) {
        bfs.reset();
        bfs.BFSInit(source, numBlock, numThread);
        //bfs.worklists.print();

        uint32_t end_host, start_host;
        int end_iter = min_iter;
        int iter = 0;
        int log_iter = 0;
        start_host = 0;
    
        timer.Start();
        while(iter < end_iter)
        {
	        CUDA_CHECK(cudaMemcpyAsync(&end_host, (uint32_t *)bfs.worklists.end, sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[32]));
	        CUDA_CHECK(cudaStreamSynchronize(streams[32]));
	        int size = end_host-start_host;
	        int stream_id = log_iter%32;
	        if(size > 0)
	        //if(size > 256 || (iter > 5 && size > 0))
	        {
                if(write_profile)
                    cudaEventRecord(event_start[stream_id*1000+event_counter[stream_id]], streams[stream_id]);
                bfs.BFSStart<FETCH_SIZE, BLOCK_SIZE>(start_host, size, 0, streams[stream_id]); 
	            if(write_profile) { 
	                profile_output[log_iter] = prof_entry(start_host, end_host, stream_id, size);
	                cudaEventRecord(event_stop[stream_id*1000+event_counter[stream_id]], streams[stream_id]);
	                event_counter[stream_id]++;
	            }
	            log_iter++;
                start_host = end_host;
	            iter = 0;
	        }
	        else {
                if(iter >= end_iter/2)
                checkEnd<<<1, 32, 0, streams[32]>>>(bfs.worklists);
                iter++;
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        timer.Stop();

        float elapsed = timer.ElapsedMillis();
        CUDA_CHECK(cudaMemcpy((uint32_t *)(bfs.worklists.start), &start_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
        bfs.worklists.print();
        std::cout << "Time: " << elapsed << std::endl;
        std::cout << "workload: " << start_host << std::endl;
        std::cout << "kernels launched: "<< log_iter << std::endl;
        times.push_back(elapsed);
        workloads.push_back(start_host);
        launched_kernels.push_back(log_iter);
        #ifdef COMPUTE_EDGES
            int *host_queue, *host_workload;
            host_queue = (int *)malloc(sizeof(int)*start_host);
            host_workload = (int *)malloc(sizeof(int)*start_host);
            cudaMemcpy(host_queue, bfs.worklists.queue, sizeof(int)*start_host, cudaMemcpyDeviceToHost);
            uint32_t totalworkload = 0;
            for(int i=0; i<start_host; i++) {
    	        int node_id = host_queue[i];
    	        host_workload[i] = csr.row_offset[node_id+1]-csr.row_offset[node_id];
    	        totalworkload = totalworkload + host_workload[i];
            }
            workload_edges.push_back(totalworkload);
            std::cout << "workload edges: "<< totalworkload << std::endl;
        #endif
        write_file_log_iter = log_iter;
    }

    std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
    std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;
    std::cout << "Ave. kernels launched: "<< std::accumulate(launched_kernels.begin(), launched_kernels.end(), 0)/launched_kernels.size()<< std::endl;
    #ifdef COMPUTE_EDGES
        std::cout << "Ave. Workload(edges): "<< std::accumulate(workload_edges.begin(), workload_edges.end(), (uint64_t)0)/workload_edges.size()<< std::endl;
    #endif
    host::BFSValid2<int, int>(csr, bfs, source);

//    string file_name_queue = str_file.substr(25, str_file.length()-4-25)+"_queue.txt";
//    string file_name_workload = str_file.substr(25, str_file.length()-4-25)+"_queue_workload.txt";
//    std::cout << "Writing to files: "<< file_name_queue << ", "<< file_name_workload<<std::endl;
//    writeToFile(file_name_queue, host_queue, queue_size, 1);
//    writeToFile(file_name_workload, host_queue, queue_size, 1);

    if(write_profile) {

       float * time_interval[32];
       float * time_accu[32];
       int max_len=0;
       for(int i=0; i<32; i++) {
	   time_interval[i] = (float *)malloc(sizeof(float)*event_counter[i]);
	   time_accu[i] = (float *)malloc(sizeof(float)*event_counter[i]);
	   max_len = max(event_counter[i], max_len);
       }
       
       for(int stream_id=0; stream_id < 32; stream_id++) 
	   for(int i=0; i<event_counter[stream_id]; i++) {
	       CUDA_CHECK(cudaEventElapsedTime(time_interval[stream_id]+i, event_start[stream_id*1000+i], event_stop[stream_id*1000+i]))
	       CUDA_CHECK(cudaEventElapsedTime(time_accu[stream_id]+i, event_begin[stream_id], event_stop[stream_id*1000+i]))
       }

       string file_name_interval = str_file.substr(25, str_file.length()-4-25)+"_interval_time.txt";
       string file_name_accu = str_file.substr(25, str_file.length()-4-25)+"_accu_time.txt";
       string file_name_size = str_file.substr(25, str_file.length()-4-25)+"_kernel_info.txt";
       string file_name_stream_kernel_size = str_file.substr(25, str_file.length()-4-25)+"_stream_size.txt";
       std::cout << "Writing to files: "<< file_name_interval << "\t" <<  file_name_accu << "\t" << file_name_size  << "\t"<< file_name_stream_kernel_size << std::endl;

       int * kernel_size = (int *)malloc(sizeof(int)*max_len);
       int local_size = 0;
       ofstream myfile4 (file_name_stream_kernel_size);
       if(myfile4.is_open()) {
       for(int stream_id =0; stream_id < 32; stream_id++)
       {
           for(int i=0; i<write_file_log_iter; i++)
	       if(profile_output[i].stream_id==stream_id) {
		     kernel_size[local_size] = profile_output[i].size;
		     local_size++;
	       }
	   std::cout << "stream "<< stream_id <<  "\t" << local_size << std::endl;
	   for(int i=0; i<local_size; i++)
		myfile4 << kernel_size[i] << "\t";
	   myfile4 << "\n\n";
	   local_size = 0;
       }
       myfile4.close();
       }

       ofstream myfile (file_name_interval);
       if (myfile.is_open())
       {
           for(int stream_id=0; stream_id<32; stream_id++) {
           for(int count = 0; count < event_counter[stream_id]; count++){
              myfile << time_interval[stream_id][count] << "\t";
           }
	   myfile << "\n\n";
	   }
           myfile.close();
       }

       ofstream myfile2 (file_name_accu);
       if (myfile2.is_open())
       {
           for(int stream_id=0; stream_id<32; stream_id++) {
           for(int count = 0; count < event_counter[stream_id]; count++){
              myfile2 << time_accu[stream_id][count] << "\t";
           }
	   myfile2 << "\n\n";
	   }
           myfile2.close();
       }

       ofstream myfile3 (file_name_size);
       if(myfile3.is_open())
       {
	   for(int count = 0; count < write_file_log_iter; count++)
	      myfile3 << profile_output[count].start << "\t" << profile_output[count].end << "\t"<< profile_output[count].size << "\t" << profile_output[count].stream_id << "\n";
	   myfile3.close();
       }
    }

    csr.release();
    bfs.release();
    CUDA_CHECK(cudaFree(warmup_out));

    return 0;
}
