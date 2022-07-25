#include <iostream>
#include <string>
#include <numeric>

#include "../../comm/coo.cuh"
#include "../../comm/csr.cuh"
#include "../../util/time.cuh"

#include "bfs_cta.cuh"
#include "validation_bfs.cuh"

#include "profile.cuh"

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



int main(int argc, char *argv[])
{
     char *input_file = NULL;
     bool start_from_0 = false;
     bool write_profile = false;
     uint32_t min_iter = 2500;
     int option = 1;
     int source = 0;
     int rounds = 50;
     bool verbose = 0;
     if(argc == 1)
     {
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -o <choose queue run launch1 or launch4=launch1> -r <source node=0>\n";
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
            else if(string(argv[i]) == "-rounds")
                 rounds = stoi(argv[i+1]);
             else if(string(argv[i]) == "-v")
                 verbose = stoi(argv[i+1]);
         }
     if(input_file == NULL)
     {
         cout << "input file is needed\n";
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -o <choose queue run launch1 or launch4=launch1>\n";
         exit(0);
     }

    int numBlock = 56*5;
    int numThread = 256;
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, 32, BFSCTA<32, int,int>, BFS<int, int>> , 0, 924);
    std::cout << "num of block: " << numBlock << "  num of threads per block: "<< numThread << std::endl;
    std::cout << "file: "<< input_file << " start from 0: " << start_from_0 << " write profile file: "<< write_profile << " " << numBlock << "x"<< numThread << " min iter "<< min_iter<< " option "<< option<<std::endl;
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
    uint32_t profile_size = 3000000;
    if(write_profile)
    {
        CUDA_CHECK(cudaMallocManaged(&profile_output, sizeof(uint32_t)*profile_size));
        CUDA_CHECK(cudaMemset(profile_output, 0, sizeof(uint32_t)*profile_size));
    }

    uint32_t *warmup_out;
    CUDA_CHECK(cudaMalloc(&warmup_out, sizeof(uint32_t)*numBlock*numThread));
    BFS<int, int> bfs(csr, min_iter);
    warpup_mallocManaged<<<numBlock, numThread>>>(bfs, warmup_out);
    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);

    std::vector<float> times;
    std::vector<uint64_t> workloads;

    for(int iteration=0; iteration<rounds; iteration++)
    {
    bfs.reset();
    bfs.BFSInit(source, numBlock, numThread);

    timer.Start();
    bfs.BFSStart_CTA<FETCHSIZE>(numBlock, numThread, 0, stream1);

    timer.Stop();
    MaxCountQueue::ReductionMin<<<1, 32>>>(bfs.worklists.execute, numBlock, (uint32_t *)(bfs.worklists.start));
    CUDA_CHECK(cudaDeviceSynchronize());
    bfs.worklists.print();
    float elapsed = timer.ElapsedMillis();
    uint32_t workload;
    CUDA_CHECK(cudaMemcpy(&workload, (uint32_t *)bfs.worklists.start,  sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Time: " << elapsed << std::endl;
    std::cout << "workload vertices: " << workload << std::endl;
    times.push_back(elapsed);
    workloads.push_back(workload);
    }

    std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
    std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;

    host::BFSValid2<int, int>(csr, bfs, source);

    CUDA_CHECK(cudaFree(warmup_out));
    csr.release();
    bfs.release();

    return 0;
}
