#include <iostream>
#include <string>
#include <numeric>

#include "../../comm/coo.cuh"
#include "../../comm/csr.cuh"
#include "pr_cta.cuh"

#include "validation.cuh"
#include "../../util/time.cuh"
#include "profile.cuh"

#define FETCH_SIZE (FETCHSIZE)
#define ROUND_SIZE (NROUNDS)

using namespace std;
__global__ void print(float *array, int size)
{
    if(TID ==0)
    {
        for(int i=0; i<size; i++)
            printf("%d: %f, ", i, array[i]);
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
     char *input_file = NULL;
     float lambda=0.85;
     float epsilon=0.01;
     bool start_from_0 = false;
     bool write_profile = false;
     uint32_t min_iter = 300;
     int rounds = 1;
     bool check = false;
     if(argc == 1)
     {
         cout<< "./test -l <lambda=0.85> -f <file> -p <epsilon=0.005> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500>\n";
         exit(0);
     }
     if(argc > 1)
         for(int i=1; i<argc; i++) {
             if(string(argv[i]) == "-l")
                 lambda = stof(argv[i+1]);
             else if(string(argv[i]) == "-f")
                input_file = argv[i+1];
             else if(string(argv[i]) == "-p")
                epsilon = stof(argv[i+1]);
             else if(string(argv[i]) == "-s")
                 start_from_0 = stoi(argv[i+1]);
             else if(string(argv[i]) == "-w")
                 write_profile = stoi(argv[i+1]);
             else if(string(argv[i]) == "-i")
                 min_iter = stoi(argv[i+1]);
            else if(string(argv[i]) == "-rounds")
                rounds = stoi(argv[i+1]);
            else if(string(argv[i]) == "-check")
                check = stoi(argv[i+1]);
         }
     if(input_file == NULL)
     {
         cout << "input file is needed\n";
         cout<< "./test -l <lambda=0.85> -f <file> -p <epsilon=0.005> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> \n";
         exit(0);
     }

    int numBlock = 80;
    int numThread = 1024;
//    CUDA_CHECK(cudaFuncSetCacheConfig(MaxCountQueue::_launchCTA_minIter<int, int, 1, PageRankFuncBlock<int,int,float>, PageRank<int, int, float>>, cudaFuncCachePreferShared));
//    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter_2func<int, int, FETCH_SIZE, PageRankFuncCTA<FETCH_SIZE, int,int,float>, PushVertices<ROUND_SIZE, int, int, float>, PageRank<int, int, float>>, 0, 900);
    numThread = 512;
    std::cout << "num of block: " << numBlock << "  num of threads per block: "<< numThread << std::endl;
    std::cout << "file: "<< input_file << "  lambda: " << lambda << " epsilon: "<< epsilon <<  " start from 0: " << start_from_0 << " write profile file: "<< write_profile << " " 
    << numBlock << "x"<< numThread << " min iter "<< min_iter<< " fetch size " << FETCH_SIZE << " round size "<< ROUND_SIZE << std::endl;
    std::string str_file(input_file);
    Csr<int, int> csr;
    if(str_file.substr(str_file.length()-4) == ".mtx")
    {
        Coo<int, int> coo(start_from_0);
        coo.BuildCooFromMtx(input_file);
        coo.Print();
        csr.FromCooToCsr(coo);
        csr.WriteToBinary(input_file);
    }
    else if(str_file.substr(str_file.length()-4) == ".csr")
    {
        csr.ReadFromBinary(input_file);
    }
    csr.PrintCsr();

    GpuTimer timer;
    uint32_t *profile_output;
    uint32_t profile_size = 400000;
    int *profile_stop;
    cudaStream_t profile_stream;
    if(write_profile)
    {
        CUDA_CHECK(cudaMallocManaged(&profile_output, sizeof(uint32_t)*profile_size));
        CUDA_CHECK(cudaMemset(profile_output, 0, sizeof(uint32_t)*profile_size));
	    CUDA_CHECK(cudaMallocManaged(&profile_stop, sizeof(int)));
        CUDA_CHECK(cudaMemset(profile_stop, 0, sizeof(int)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&profile_stream,cudaStreamNonBlocking));
        numBlock = numBlock;
    }

    PageRank<int, int, float> pr(csr, lambda, epsilon, min_iter);
    cudaStream_t stream1, stream2, stream3, stream4, stream5;
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);

    std::vector<float> times;
    std::vector<uint64_t> workloads;

    for(int iteration=0; iteration < rounds; iteration++) {
    pr.reset();
    //pr.worklists.print();
    pr.PrInit(numBlock, numThread);
    //host::PrInitValid<int, int, float>(csr, pr);
    //pr.worklists.print();
    
    //if(write_profile)
    //    Sampling<<<1,32,0,profile_stream>>>((uint32_t *)pr.worklists.start_count,(uint32_t *)pr.worklists.end_count,(uint32_t *)pr.worklists.end, (uint32_t *)pr.worklists.start_alloc, pr.worklists.num_queues, profile_output, profile_stop, profile_size, 100000);

    timer.Start();
    pr.PrStart_CTA<FETCH_SIZE, ROUND_SIZE>(numBlock, numThread, 0,stream1);
    timer.Stop();

    if(write_profile)
    {
        *profile_stop= 1;
        CUDA_CHECK(cudaStreamSynchronize(profile_stream));
    }

    MaxCountQueue::ReductionMin<<<1, 32>>>(pr.worklists.execute, numBlock, (uint32_t *)(pr.worklists.start));
    CUDA_CHECK(cudaDeviceSynchronize());
    pr.worklists.print();   
    uint32_t totalworkload=0;
    CUDA_CHECK(cudaMemcpy(&totalworkload, (uint32_t *)(pr.worklists.end), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    float elapsed = timer.ElapsedMillis();
    std::cout << "Time: " << elapsed << std::endl;
    std::cout << "workload: " << totalworkload<< std::endl;
    times.push_back(elapsed);
    workloads.push_back(totalworkload);
    uint32_t res_error = 0;
    for(int i=0; i<pr.nodes; i++)
        if(pr.res[i] >= pr.epsilon)
            res_error++;
    std::cout << "num of " << res_error << " res larger than " << pr.epsilon << std::endl;
    }

    printf("Ave. Time: %8.2f\n", std::accumulate(times.begin(), times.end(), 0.0)/times.size());
    printf("Ave Workload: %lld\n", (long long)(std::accumulate(workloads.begin(), workloads.end(), (long long)0)/workloads.size()));
    if(check)
        host::PrValid<int, int, float>(csr, pr);
    std::cout << "num of vertices scanned "<< std::dec<<*(pr.checkres)*32 << std::endl;

    uint32_t num_act=0;
    for(int i=0; i<pr.size_ifact; i++)
        if(pr.ifact[i]!=0)
        {
            num_act++;
            if(num_act <= 1)
                std::cout << "ifact[" << i<< "]: "<< std::hex << pr.ifact[i] << std::endl;
        }
    std::cout << "num of vertices active: "<< std::dec << num_act << std::endl;
    if(write_profile)
    {
	    string file_name = str_file.substr(28, str_file.length()-4-28);
        std::string file_name_end = file_name + "_cta_end.txt";
        std::string file_name_end_count = file_name + "_cta_end_count.txt";
        std::string file_name_start_count = file_name + "_cta_start_count.txt";
        std::string file_name_start_alloc= file_name + "_cta_start_alloc.txt";

    	std::cout << "writing profile to output" <<  file_name_end << ", "<< file_name_end_count << ", "<< file_name_start_count << ", "<< file_name_start_alloc << "\n";
        writeToFile(file_name_start_count, profile_output, profile_size, 4);
        writeToFile(file_name_end_count, profile_output+1, profile_size, 4);
        writeToFile(file_name_end, profile_output+2, profile_size, 4);
        writeToFile(file_name_start_alloc, profile_output+3, profile_size, 4);
        CUDA_CHECK(cudaFree(profile_output));
    }

    csr.release();
    pr.release();

    return 0;
}
