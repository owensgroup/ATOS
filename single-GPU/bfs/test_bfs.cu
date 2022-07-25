#include <iostream>
#include <string>
#include <numeric>

#include "../../comm/coo.cuh"
#include "../../comm/csr.cuh"
#include "../../util/time.cuh"

#include "bfs.cuh"
#include "validation_bfs.cuh"

#include "profile.cuh"

using namespace std;

#define FETCH_SIZE (FETCHSIZE)
#define BLOCK_SIZE (512)

int main(int argc, char *argv[])
{
     char *input_file = NULL;
     bool start_from_0 = false;
     bool write_profile = false;
     uint32_t min_iter = 2500;
     int source = 0;
     int option = 3;
     int num_queue=1;
     int device = 0;
     int rounds = 50;
     bool verbose = 0;
     if(argc == 1)
     {
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -o <choose queue run launch1 or launch4=launch1> -r <source node to start=0> -q <number of queues used=4> -d <device id=0>\n";
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
             else if(string(argv[i]) == "-v")
                 verbose = stoi(argv[i+1]);
         }
     if(input_file == NULL)
     {
         cout << "input file is needed\n";
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -w <write profile?=false> -i <min iteration for queue=2500> -o <choose queue run launch1 or launch4=launch1> -r <source node to start=0> -q <number of queues used=4> -d <device id=0>\n";
         exit(0);
     }

    std::cout << "set on device "<< device << std::endl;
    CUDA_CHECK(cudaSetDevice(device));

    int numBlock = 56*5;
    int numThread = 256;
    if(option == 1)
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchThreadPerItem_minIter<int, uint32_t, BFSThread<int,int>, BFS<int, int >> );
    else if(option == 2)
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchWarpPer32Items_minIter<int, uint32_t, BFSWarp<int,int>, BFS<int, int>>, 0, 1000);
    else if(option == 3)
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, FETCH_SIZE, BFSCTA<FETCH_SIZE,int,int>, BFS<int, int>>, 0, 600);
//    numThread=64; numBlock=1280;
    std::cout << "num of block: " << numBlock << "  num of threads per block: "<< numThread << std::endl;
    std::cout << "file: "<< input_file << " start from 0: " << start_from_0 << " write profile file: "<< write_profile << " " << numBlock << "x"<< numThread << " min iter "<< min_iter<< 
    " option "<< option<< " number of queues "<< num_queue << " FETCH SIZE " << FETCH_SIZE << " BLOCK SIZE " << BLOCK_SIZE << std::endl;
    std::string str_file(input_file);
    Csr<int, int> csr;
    if(str_file.substr(str_file.length()-4) == ".mtx")
    {
        std::cout << "generate csr file first\n";
        exit(0);
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

    BFS<int, int> bfs(csr, min_iter, num_queue);
    std::cout << "source "<< source << std::endl;

    std::vector<float> times;
    std::vector<uint64_t> workloads;

    for(int iteration=0; iteration < rounds; iteration++) {
        bfs.reset();
        if(option == 1 || option == 2)
        bfs.BFSInit(source, numBlock, numThread);
        else if (option == 3) 
        bfs.BFSInit_CTA(source, numBlock, numThread);
        else if(option == 4) {
        bfs.BFSInit(source);
        bfs.BFSDiscrete_prepare();
        }
        //bfs.worklists.print();

        timer.Start();
        if(option == 1)
        bfs.BFSStart_threadPerItem(numBlock, numThread);
        else if(option == 2)
        bfs.BFSStart_warpPer32Items4(numBlock, numThread);
        else if(option == 3)
        bfs.BFSStart_CTA<FETCH_SIZE>(numBlock, numThread, 0);
        else if(option == 4)
        bfs.BFSStart_discrete<FETCH_SIZE, BLOCK_SIZE>();

        timer.Stop();
        if(verbose)
            bfs.worklists.print();
        float elapsed = timer.ElapsedMillis();
        uint32_t workload = 0;
        for(int q_id=0; q_id<num_queue; q_id++) {
            uint32_t temp;
            CUDA_CHECK(cudaMemcpy(&temp, (uint32_t *)(bfs.worklists.start+q_id*PADDING_SIZE), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            workload += temp;
        }
        std::cout << "Time: " << elapsed << std::endl;
        std::cout << "workload: " << workload << std::endl;
        times.push_back(elapsed);
        workloads.push_back(workload);
    }

    std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
    std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;

    host::BFSValid2<int, int>(csr, bfs, source);
//    for(int i=0; i<40; i++)
//        std::cout << bfs.depth[i] << ", ";
//    std::cout << std::endl;

    if(write_profile)
    {
    std::cout << "writing profile to output\n";
    std::string file_name_end = "output_end.txt";
    std::string file_name_start = "output_start.txt";
    std::string file_name_start_count = "output_start_count.txt";
    writeToFile(file_name_end, profile_output, profile_size, 3000);
    writeToFile(file_name_start_count, profile_output+1, profile_size, 3000);
    writeToFile(file_name_start, profile_output+2, profile_size, 3000);
    CUDA_CHECK(cudaFree(profile_output));
    }

    csr.release();
    bfs.release();

    return 0;
}
