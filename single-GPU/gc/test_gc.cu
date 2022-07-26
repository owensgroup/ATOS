#include <iostream>
#include <string>
#include <numeric>

//#define ITER_WORKLOAD

#include "../../comm/coo.cuh"
#include "../../comm/csr.cuh"
#include "../../util/time.cuh"

#include "baseline.cuh"
#include "gc.cuh"
#include "validation.cuh"

#define FETCH_SIZE (FETCHSIZE) 

using namespace std;

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
     int rounds = 10;
     bool verbose = 0;
     int run_bsp_async = 0;  // 0 run both, 1 run bsp only, 2 run async only
     bool permute = false;
     bool ifmesh = false;
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
            else if(string(argv[i]) == "-run_bsp_async")
                run_bsp_async = stoi(argv[i+1]);
            else if(string(argv[i]) == "-permute")
                permute = stoi(argv[i+1]);
            else if(string(argv[i]) == "-mesh")
                ifmesh = stoi(argv[i+1]);
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
    //if(option == 1)
    //cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchThreadPerItem_minIter<int, uint32_t, BFSThread<int,int>, BFS<int, int >> );
    if(option == 2)
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchWarpPer32Items_minIter<int, uint32_t, GCWarp_op<int,int>, GC_Async<int, int>>, 0, 1000);
    else if(option == 3) {
        if(ifmesh)
            cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, FETCH_SIZE, GCCTA_mesh<FETCH_SIZE, int,int>, GC_Async<int, int>>, 0, 1000);
        else
            cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, FETCH_SIZE, GCCTA<FETCH_SIZE,int,int>, GC_Async<int, int>>, 0, 1000);
    //cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, FETCH_SIZE, GCCTA_simple2<FETCH_SIZE, int,int>, GC_Async<int, int>>, 0, 600);
    //numThread=512; numBlock=160;
    }
    else if(option == 4)
    cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)MaxCountQueue::_launchCTA_minIter<int, uint32_t, FETCH_SIZE, GCCTA_simple<FETCH_SIZE, 1, int,int>, GC_Async<int, int>>, 0, 100);
    std::cout << "num of block: " << numBlock << "  num of threads per block: "<< numThread << std::endl;
    std::cout << "file: "<< input_file << " start from 0: " << start_from_0 << " write profile file: "<< write_profile << " " << numBlock << "x"<< numThread << " min iter "<< min_iter<< std::endl; 
    //" option "<< option<< " FETCH SIZE " << FETCH_SIZE << " BLOCK SIZE " << BLOCK_SIZE << std::endl;
    std::cout << "permute " << permute << std::endl;
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
    std::vector<float> times;
    std::vector<uint64_t> workloads;

    if(run_bsp_async == 0 || run_bsp_async == 1) {
    GC_BSP<int, int> gc_bsp(csr);
    CUDA_CHECK(cudaDeviceSynchronize());

    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //warm up
    //gc_bsp.GCInit(permute, true);
    //gc_bsp.GCStart_op(stream);
    gc_bsp.GCStart_warp_thread_cta();

    for(int iteration=0; iteration < rounds; iteration++) {

        gc_bsp.reset();
        //gc_bsp.GCInit(permute, false);
        timer.Start();
        //uint32_t workload = gc_bsp.GCStart_op(stream);
        uint32_t workload = gc_bsp.GCStart_warp_thread_cta();
        timer.Stop();
        float elapsed = timer.ElapsedMillis();
        std::cout << "Time: " << elapsed << std::endl;
        times.push_back(elapsed);
        workloads.push_back(workload);
    }

    if(times.size() > 0) {
        std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
        std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;
    }
    
    GCValid<int, int>(gc_bsp);
    std::cout << "-----------------------------------------\n\n" << std::endl;
    gc_bsp.release();
    }

    if(run_bsp_async == 0 || run_bsp_async == 2) {
    GC_Async<int, int> gc_async(csr, min_iter, num_queue);

    times.clear();
    workloads.clear();
    if(option == 2) {
        // warm up
        gc_async.GCInit(numBlock, numThread, permute, true);
        gc_async.worklists.print();
        gc_async.GCStart(numBlock, numThread);


        for(int iteration=0; iteration < rounds; iteration++) {
            gc_async.reset();
            gc_async.GCInit(numBlock, numThread, permute, false);
            //gc_async.worklists.print();
            timer.Start();
            gc_async.GCStart(numBlock, numThread);
            timer.Stop();
            gc_async.worklists.print();
            float elapsed = timer.ElapsedMillis();
            std::cout << "Time: " << elapsed << std::endl;
            times.push_back(elapsed);
            uint32_t workload = gc_async.getWorkload();
            workloads.push_back(workload);
        }
        //gc_async.outputVistTimes("chesapeak_warp_freq.txt");
    }
    else if(option == 3) {
        //gc_async.outputNeighborLen("road_ca_neighborlen.txt");
        // warm up
        gc_async.GCInit_CTA(numBlock, numThread, permute, true);
        gc_async.worklists.print();
        gc_async.GCStart_CTA<FETCH_SIZE>(numBlock, numThread, ifmesh);

        for(int iteration=0; iteration < rounds; iteration++) {
            gc_async.reset();
            gc_async.GCInit_CTA(numBlock, numThread, permute, false);
            timer.Start();
            gc_async.GCStart_CTA<FETCH_SIZE>(numBlock, numThread, ifmesh);
            timer.Stop();
            gc_async.worklists.print();
            float elapsed = timer.ElapsedMillis();
            std::cout << "Time: " << elapsed << std::endl;
            times.push_back(elapsed);
            uint32_t workload = gc_async.getWorkload();
            workloads.push_back(workload);
        }
        
        //gc_async.outputVistTimes("road_ca_simple_freq.txt");
    }
    else if(option == 4) {
        gc_async.GCInit_CTA(numBlock, numThread, permute, true);
        gc_async.worklists.print();
        gc_async.GCStart_CTA_simple<FETCH_SIZE>(numBlock, numThread);

        for(int iteration=0; iteration < rounds; iteration++) {
            gc_async.reset();
            gc_async.GCInit_CTA(numBlock, numThread, permute, false);
            timer.Start();
            gc_async.GCStart_CTA_simple<FETCH_SIZE>(numBlock, numThread);
            timer.Stop();
            gc_async.worklists.print();
            float elapsed = timer.ElapsedMillis();
            std::cout << "Time: " << elapsed << std::endl;
            times.push_back(elapsed);
            uint32_t workload = gc_async.getWorkload();
            workloads.push_back(workload);
        }
    }

    if(times.size() > 0) {
        std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
        std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;
    }

    GCValid<int, int>(gc_async);
    gc_async.release();
    }
    
    if(run_bsp_async == 0 || run_bsp_async == 3) {
        GC_Async<int, int> gc_async(csr, min_iter, num_queue);
        
        times.clear();
        workloads.clear(); 

        //warm up
        gc_async.GCInit_discrete(permute, true);
        gc_async.worklists.print();
        gc_async.GCStart_discrete<1, 256>();

        for(int iteration=0; iteration < rounds; iteration++) {
            gc_async.reset();
            gc_async.GCInit_discrete(permute, false);
            //gc_async.worklists.print();
            timer.Start();
            gc_async.GCStart_discrete<1, 256>();
            timer.Stop();
            gc_async.worklists.print();
            float elapsed = timer.ElapsedMillis();
            std::cout << "Time: " << elapsed << std::endl;
            times.push_back(elapsed);
            uint32_t workload = gc_async.getWorkload();
            workloads.push_back(workload);
        }

        if(times.size() > 0) {
            std::cout << "Ave. Time: "<< std::accumulate(times.begin(), times.end(), float(0.0))/times.size() << std::endl;
            std::cout << "Ave. Workload(vertices): "<< std::accumulate(workloads.begin(), workloads.end(), (uint64_t)0)/workloads.size() << std::endl;
        }
        GCValid<int, int>(gc_async);
        gc_async.release();
    }


    csr.release();

    return 0;
}
