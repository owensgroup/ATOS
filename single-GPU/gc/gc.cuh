#ifndef GraphColoring
#define GraphColoring
#include <string>
#include <iostream>
#include <fstream>

#include "../../util/error_util.cuh"
#include "../../util/util.cuh"
#include "../../comm/csr.cuh"
#include "../queue/queue.cuh"

#include <cub/cub.cuh>

template<typename VertexId, typename SizeT, typename QueueT=uint32_t>
struct GC_Async
{
    VertexId nodes;
    SizeT edges;

    SizeT *csr_offset = NULL;
    VertexId *csr_indices = NULL;

    VertexId maxDegree = 0xffffffff;
    VertexId fixSize = 256;  //if fixSize=512, forbiddenColors size will larger than max(int)

    VertexId *colors = NULL;
    bool *forbiddenColors = NULL;
    bool *ifpushed = NULL;
    
    int *pushNext = NULL;

    VertexId *init_queue = NULL;

    MaxCountQueue::Queues<VertexId, QueueT> worklists;

    GC_Async(Csr<VertexId, SizeT> &csr, uint32_t min_iter = 800, int num_queues=1)
    {
        nodes = csr.nodes;
        edges = csr.edges;
        
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;

        maxDegree = 0;
        for(int i=0; i<nodes; i++)
        {
            maxDegree = max(maxDegree, csr.row_offset[i+1]-csr.row_offset[i]);
        }
        printf("max degree %d\n", maxDegree);

        CUDA_CHECK(cudaMalloc(&colors, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&forbiddenColors, sizeof(bool)*fixSize*nodes));
        CUDA_CHECK(cudaMalloc(&ifpushed, sizeof(bool)*nodes));
        CUDA_CHECK(cudaMalloc(&pushNext, sizeof(int)*nodes));

        worklists.init(QueueT(nodes*128), num_queues, min_iter);
        std::cout << "wl_size: "<<worklists.get_capacity()<<std::endl;
        std::cout << "min_iter: "<< min_iter << std::endl;
        
        CUDA_CHECK(cudaMemset(colors, 0xffffffff, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(forbiddenColors, false, sizeof(bool)*fixSize*nodes));
        CUDA_CHECK(cudaMemset(ifpushed, 0, sizeof(bool)*nodes));
        CUDA_CHECK(cudaMemset(pushNext, 0, sizeof(int)*nodes));
    }

    void release()
    {
        worklists.release();
        if(colors!=NULL)
            CUDA_CHECK(cudaFree(colors));
        if(forbiddenColors!=NULL)
            CUDA_CHECK(cudaFree(forbiddenColors));
        if(ifpushed!=NULL)
            CUDA_CHECK(cudaFree(ifpushed));
        if(pushNext!=NULL)
            CUDA_CHECK(cudaFree(pushNext));
        if(init_queue!=NULL)
            free(init_queue);
    }

    void reset()
    {
        worklists.reset();
        CUDA_CHECK(cudaMemset(colors, 0xffffffff, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(forbiddenColors, 0, sizeof(bool)*fixSize*nodes));
        CUDA_CHECK(cudaMemset(ifpushed, 0, sizeof(bool)*nodes));
        CUDA_CHECK(cudaMemset(pushNext, 0, sizeof(int)*nodes));
        CUDA_CHECK(cudaMemset(worklists.queue, 0, sizeof(int)*worklists.capacity));
    }

    void RandomInit(bool ifCreate);
    void GCInit(int numBlock, int numThread, bool random, bool ifcreate);
    void GCStart(int numBlock, int numThread);
    void GCInit_CTA(int numBlock, int numThread, bool random, bool ifcreate);
    template<int FETCH_SIZE>
    void GCStart_CTA(int numBlock, int numThread, bool ifmesh);
    template<int FETCH_SIZE>
    void GCStart_CTA_simple(int numBlock, int numThread);
    void GCInit_discrete(bool random, bool ifcreate);
    template<int FETCH_SIZE, int BLOCK_SIZE>
    void GCStart_discrete();

    VertexId getWorkload();
    void outputVistTimes(std::string file_name);
    void outputNeighborLen(std::string file_name);
};

template<typename VertexId, typename SizeT>
__global__ void init(GC_Async<VertexId, SizeT> gc)
{
    for(VertexId id = TID; id < gc.nodes; id+=gridDim.x*blockDim.x)
    //for(VertexId id = TID; id < 20000; id+=gridDim.x*blockDim.x)
    {
        gc.worklists.push_warp(id+1);
        gc.ifpushed[id] = true;
    }
}

template<typename VertexId, typename SizeT>
__global__ void randomInit(GC_Async<VertexId, SizeT> gc)
{
    for(VertexId id = TID; id < gc.nodes; id+=gridDim.x*blockDim.x)
    {
        gc.ifpushed[id] = true;
    }

    *(gc.worklists.end) = gc.nodes;
    *(gc.worklists.end_alloc) = gc.nodes;
    *(gc.worklists.end_max) = gc.nodes;
    *(gc.worklists.end_count) = gc.nodes;
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::GCInit(int numBlock, int numThread, bool random, bool ifcreate)
{
    if(ifcreate) {
        worklists.launchWarpPer32Items_minIter_preLaunch(numBlock, numThread);
        printf("warpSize_wl_pop %d\n", worklists.warpSize_wl_pop);
    }
    if(random) {
        RandomInit(ifcreate);
    }
    else {
        init<<<320, 512>>>(*this);   
        CUDA_CHECK(cudaDeviceSynchronize());
        MaxCountQueue::checkEnd<<<1, 32>>>(worklists);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::GCInit_CTA(int numBlock, int numThread, bool random, bool ifcreate)
{
    if(ifcreate) {
        worklists.launchCTA_minIter_preLaunch(numBlock, numThread);
        printf("warpSize_wl_pop %d\n", worklists.warpSize_wl_pop);
    }
    if(random)
        RandomInit(ifcreate);
    else {
        init<<<320, 512>>>(*this);   
        CUDA_CHECK(cudaDeviceSynchronize());
        MaxCountQueue::checkEnd<<<1, 32>>>(worklists);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::GCInit_discrete(bool random, bool ifcreate)
{
    if(ifcreate)
        worklists.launchDiscrete_prepare();

    if(random) {
        RandomInit(ifcreate);
    }
    else {
        init<<<320, 512>>>(*this);   
        CUDA_CHECK(cudaDeviceSynchronize());
        MaxCountQueue::checkEnd<<<1, 32>>>(worklists);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::RandomInit(bool ifCreate)
{
    if(ifCreate) {
        init_queue = (VertexId *)malloc(sizeof(VertexId)*nodes);
        VertexId *array = (VertexId *)malloc(sizeof(VertexId)*nodes);
        for(int i=0; i<nodes; i++) {
            array[i] = i;
            init_queue[i] = 0xffffffff;
        }
    
        srand(12321);
        VertexId nodeAdded = 0;
        while(nodeAdded < nodes)
        {
            VertexId nodeId = rand()%nodes;
            if(array[nodeId]!=0xffffffff) {
                assert(array[nodeId] == nodeId);
                init_queue[nodeAdded] = nodeId+1;
                array[nodeId] = 0xffffffff;
                nodeAdded++;
            }
        }
    
        for(int i=0; i<nodes; i++)
            assert(init_queue[i]!=0xffffffff);
    }
    
    CUDA_CHECK(cudaMemcpy((void *)(worklists.queue), init_queue, sizeof(VertexId)*nodes, cudaMemcpyHostToDevice));
    randomInit<<<320, 512>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());

}

template<typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCWarp{
public:
    __forceinline__ __device__ void operator()(VertexId node, GC_Async<VertexId,SizeT, QueueT> gc)
    {
        if(node > 0)  // assign color
        {
            
            node = node-1;
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            bool found = false;

            for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=gc.fixSize)
            {
                for(int item=LANE_; item < gc.fixSize; item = item+32)
                {
                    uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
                    gc.forbiddenColors[forbid_idx] = false;
                }
                __syncwarp();

                for(int item = LANE_; item<neighborlen; item=item+32)
                {
                    VertexId neighbor = gc.csr_indices[node_offset+item];
                    VertexId color = gc.colors[neighbor];
                    uint64_t forbid_idx = uint64_t(node)*gc.fixSize+color-colorRange;
                    if(color!=0xffffffff && color >= colorRange && colorRange < colorRange+gc.fixSize)
                        gc.forbiddenColors[forbid_idx] = true;
                }
                __syncwarp();
                bool ifForbid = false;
                for(int item = LANE_; item-LANE_ < gc.fixSize; item = item+32)
                {
                    if(item < gc.fixSize) {
                        uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
                        ifForbid = gc.forbiddenColors[forbid_idx];
                    }
                    unsigned mask = __ballot_sync(0xffffffff, !ifForbid);
                    if(mask!=0) {
                        found = true;
                        if(LANE_ == 0)
                            gc.colors[node] = colorRange+item+__ffs(mask)-1;
                        break;
                    }
                }
                if(found)
                    break;
                __syncwarp();
            }
            assert(found==true);
            if(LANE_ == 0) {
                gc.ifpushed[node] = false;
                gc.worklists.push_warp((node+1)*(-1));
            }
            __syncwarp();
        }
        else if(node < 0)  // conflict detection
        {
            node = node*(-1)-1;
            VertexId color = gc.colors[node];
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            bool node_notadd = true;
            for(VertexId item=LANE_; item-LANE_ < neighborlen; item=item+32)
            {
                bool ifpushnew = false;
                VertexId neighbor;
                if(item < neighborlen) {
                    neighbor = gc.csr_indices[node_offset+item];
                    VertexId neighbor_color = gc.colors[neighbor];
                    ifpushnew = ((neighbor_color == color) && 
                                 (((node > neighbor) && node_notadd) || ((neighbor > node) && (gc.ifpushed[neighbor] == false)))
                                ) ;
                    neighbor = max(neighbor, node);
                }
                if(node_notadd) {
                    unsigned mask = __ballot_sync(0xffffffff, (ifpushnew && (neighbor == node)));
                    if(__popc(mask) > 0)
                    {
                        node_notadd = false;
                        if(ifpushnew && (neighbor == node) && LANE_ != __ffs(mask)-1)
                            ifpushnew = false;
                    }
                }
                __syncwarp(0xffffffff);
                if(ifpushnew) {
                    //atomicAdd(gc.pushNext+neighbor, 1);
                    gc.worklists.push_warp(neighbor+1);
                    gc.ifpushed[neighbor] = true;
                }
                __syncwarp(0xffffffff);
            }
        }
        else {
            printf("node == %d Fail\n", node);
            assert(0);
        }
    }
};

template<typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCWarp_op{
public:
    __forceinline__ __device__ void operator()(VertexId node, GC_Async<VertexId,SizeT, QueueT> gc)
    {
        if(node == 0)
            printf("tid %d, task %d\n", TID, node);
        if(node > 0)  // assign color
        {
            //extern __shared__ uint32_t forbiddenColors[];        
            __shared__ uint32_t forbiddenColors[1792];
            node = node-1;
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            unsigned found_mask = 0;

            for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=2048)
            {
                forbiddenColors[LANE_+(threadIdx.x>>5)*64] = 0;
                forbiddenColors[LANE_+32+(threadIdx.x>>5)*64] = 0;
                __syncwarp();

                for(int item = LANE_; item<neighborlen; item=item+32)
                {
                    VertexId neighbor = gc.csr_indices[node_offset+item];
                    VertexId color = gc.colors[neighbor];
                    VertexId color_forbid = color-colorRange;
                    if(color!=0xffffffff && color_forbid >= 0 && color_forbid < 2048)
                        atomicOr(forbiddenColors+(threadIdx.x>>5)*64+color_forbid/32, (1<<(color_forbid%32)));
                }
                __syncwarp();
                for(int item = LANE_; item < 64; item = item+32)
                {
                    unsigned mask = forbiddenColors[(threadIdx.x>>5)*64+item];
                    mask = (~mask);
                    found_mask = __ballot_sync(0xffffffff, (mask!=0));
                    if(found_mask!=0) {
                        if(LANE_ == __ffs(found_mask)-1) {
                            gc.colors[node] = colorRange+item*32+__ffs(mask)-1;
                            atomicAdd(gc.pushNext+node, 1);
                        }
                        break;
                    }
                }
                if(found_mask != 0)
                    break;
            }
            assert(found_mask != 0);
            __syncwarp();
            if(LANE_ == 0) {
                gc.ifpushed[node] = false;
                gc.worklists.push_warp((node+1)*(-1));
                assert((node+1)*(-1)!=0);
            }
            __syncwarp();
        }
        else if(node < 0)  // conflict detection
        {
            node = node*(-1)-1;
            VertexId color = gc.colors[node];
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            bool node_notadd = true;
            for(VertexId item=LANE_; item-LANE_ < neighborlen; item=item+32)
            {
                bool ifpushnew = false;
                VertexId neighbor;
                if(item < neighborlen) {
                    neighbor = gc.csr_indices[node_offset+item];
                    VertexId neighbor_color = gc.colors[neighbor];
                    ifpushnew = ((neighbor_color == color) && 
                                 (((node > neighbor) && node_notadd) || ((neighbor > node) && (gc.ifpushed[neighbor] == false)))
                                ) ;
                    neighbor = max(neighbor, node);
                }
                if(node_notadd) {
                    unsigned mask = __ballot_sync(0xffffffff, (ifpushnew && (neighbor == node)));
                    if(__popc(mask) > 0)
                    {
                        node_notadd = false;
                        if(ifpushnew && (neighbor == node) && LANE_ != __ffs(mask)-1)
                            ifpushnew = false;
                    }
                }
                __syncwarp(0xffffffff);
                if(ifpushnew) {
                    //atomicAdd(gc.pushNext+neighbor, 1);
                    gc.worklists.push_warp(neighbor+1);
                    gc.ifpushed[neighbor] = true;
                    assert(neighbor+1!=0);
                }
                __syncwarp(0xffffffff);
            }
        }
        // though there supposed no 0, but sometimes get 0 for unknow reason, comment still get right resuls
        //else {
        //    printf("node == %d Fail\n", node);
        ////    assert(0);
        //} 
    }
};

// MAPSIZE use 16kb for colormaps, depend on FETCH SIZE, each node get 32*1024/FETCH_SIZE/8 uint32_t as map
// FETCH_SIZE =32 MAPSIZE=128 uint32_t = 4096
// FETCH_SIZE =64 MAPSIEE= 64 uint32_t = 2048
// FETCH_SIZE =128 MAPSIZE= 32 uint32_t = 1024
// FETCH_SIZE =256 MAPSIZE= 16 uint32_t = 512
// FETCH_SIZE =512 MAPSIZE= 8 uint32_t =  256
template<int FETCH_SIZE, typename VertexId, typename SizeT, typename QueueT=uint32_t, int MAPSIZE=4096/FETCH_SIZE>
class GCCTA {
public:
    __forceinline__ __device__ void operator()(VertexId *nodes, int size, GC_Async<VertexId, SizeT, QueueT> gc)
    {
        __shared__ SizeT node_offset[FETCH_SIZE];
        __shared__ SizeT neighborLen[FETCH_SIZE+1];
        __shared__ SizeT scan[FETCH_SIZE+1];
        __shared__ uint32_t forbiddenColors[4096];
        __shared__ int ifpush_process[FETCH_SIZE];

        if(threadIdx.x <  size)
        {
            ifpush_process[threadIdx.x] = 0;
            VertexId node = abs(nodes[threadIdx.x])-1;
            if(node == -1) printf("ERROR node = 0\n"); 
            node_offset[threadIdx.x] = gc.csr_offset[node];
            neighborLen[threadIdx.x] = gc.csr_offset[node+1]-node_offset[threadIdx.x];
        }
        __syncthreads();

        typedef cub::BlockScan<int, 512, cub::BLOCK_SCAN_RAKING,1,1,700> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;
        int block_aggregate;
        int thread_data = 0;
        if(threadIdx.x < size && neighborLen[threadIdx.x] <= MAPSIZE*32)
            thread_data = neighborLen[threadIdx.x];
        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
        __syncthreads();
        if(threadIdx.x < size)
            scan[threadIdx.x] = thread_data;
        if(threadIdx.x == 0)
            scan[size] = block_aggregate;
        for(int idx = threadIdx.x; idx<FETCH_SIZE*MAPSIZE; idx+=blockDim.x)
            forbiddenColors[idx] = 0;
        __syncthreads();

        int total_slots = ((block_aggregate+31)>>5);
        int slots_warp = ceil(float(total_slots)/(blockDim.x>>5));
        int start_index=FETCH_SIZE-1;
        for(int item = LANE_; item<FETCH_SIZE; item = item+32)
        {
            unsigned mask_start  = __ballot_sync(0xffffffff, scan[item]> (threadIdx.x>>5)*32*slots_warp);
            if(mask_start!=0)
            {
                start_index = item-LANE_+ (__ffs(mask_start)-2);
                break;
            }
        }

        int abs_neighbors = (threadIdx.x>>5)*32*slots_warp;
        int cur_node_idx=start_index;
        int cur_node_offset;
        for(int iter=0; iter<slots_warp; iter++)
        {
            bool ifpush = false;
            VertexId neighbor = -1;
            if(abs_neighbors+LANE_ < block_aggregate)
            {
                while(abs_neighbors+LANE_ >= scan[cur_node_idx+1])
                    cur_node_idx++;
                cur_node_offset = abs_neighbors+LANE_-scan[cur_node_idx]; // what if next node's neighborlen is 0
                neighbor = gc.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
                //VertexId color = gc.colors[neighbor];
                VertexId color = *((volatile VertexId *)(gc.colors+neighbor));
                if(nodes[cur_node_idx] > 0) {
                    if(color != 0xffffffff && color < MAPSIZE*32 && color >= 0)
                        atomicOr(forbiddenColors+cur_node_idx*MAPSIZE+color/32, (1<<(color%32)));
                }
                else if(nodes[cur_node_idx] < 0) {
                    VertexId temp_node = (-1)*(nodes[cur_node_idx])-1;
                    ifpush = ( (*((volatile VertexId *)(gc.colors+temp_node)) == color) && 
                               ( (neighbor > temp_node && gc.ifpushed[neighbor] == false) ||
                                 (temp_node > neighbor && atomicAdd(ifpush_process+cur_node_idx, 1) == 0) )
                             );
                    neighbor = max(neighbor, temp_node);
                }
            }
            assert((ifpush && nodes[cur_node_idx] < 0) || !ifpush );
            gc.worklists.push_cta(ifpush, neighbor+1);
            if(ifpush) gc.ifpushed[neighbor] = true;
            //if(ifpush) atomicAdd(gc.pushNext+neighbor, 1);
            __syncthreads();
            abs_neighbors = abs_neighbors+32;
        }
        __syncthreads();        
        for(int nodeId = (threadIdx.x>>5); nodeId < size; nodeId += (blockDim.x >>5))
        {
            if(neighborLen[nodeId] <= MAPSIZE*32 && nodes[nodeId] > 0) {
                unsigned found_mask = 0;
                for(int item = LANE_; item-LANE_ < MAPSIZE; item = item+32)
                {
                    unsigned mask = 0;
                    if(item < MAPSIZE) {
                        mask = forbiddenColors[nodeId*MAPSIZE+item];
                        mask = (~mask);
                    }
                    found_mask = __ballot_sync(0xffffffff, (mask!=0));
                    if(found_mask!=0) {
                        if(LANE_ == __ffs(found_mask)-1) {
                            gc.colors[nodes[nodeId]-1] = item*32+__ffs(mask)-1;
                            gc.ifpushed[nodes[nodeId]-1] = false;
                        }
                        break;
                    }
                    __syncwarp();
                }
                assert(found_mask!=0);
            }
        }

        if(threadIdx.x == 0) scan[0] = 0x7fffffff;
        __syncthreads();

        // find out if more nodes with more than MAPSIZE neighbors
        if(threadIdx.x < size) {
            if(neighborLen[threadIdx.x] > MAPSIZE*32) {
                atomicMin(scan, threadIdx.x);
            }
        }
        __syncthreads();

        while(scan[0] != 0x7fffffff) {
        //if(scan[0] != 0x7fffffff) {
            VertexId nodeId = scan[0];
            SizeT len = neighborLen[nodeId];
            __syncthreads();
            neighborLen[nodeId] = 0;
            scan[0] = 0x7fffffff;
            assert(nodeId >= 0);
            assert(nodeId < size);
            //if(threadIdx.x == 0) printf("nodeID %d, node %d neighborLen %d offset\n", nodeId, nodes[nodeId]-1, len);
            //if(threadIdx.x == 0) printf("nodeID %d, node %d neighborLen %d offset %d\n", nodeId, nodes[nodeId]-1, len, node_offset[nodeId]);

            for(VertexId colorRange = 0; colorRange < gc.maxDegree+1; colorRange += 131072) {
                if(nodes[nodeId] >= 0) {
                for(int idx = threadIdx.x; idx<FETCH_SIZE*MAPSIZE; idx+=blockDim.x)
                    forbiddenColors[idx] = 0;
                } else {
                    if(threadIdx.x == 0) scan[1] = gc.colors[(-1)*nodes[nodeId]-1];
                }
                __syncthreads();

                for(VertexId idx = threadIdx.x; idx-threadIdx.x < len; idx+=blockDim.x) {
                    bool ifpush = false;
                    VertexId neighbor = -1;
                    if(idx < len) {
                        neighbor = gc.csr_indices[node_offset[nodeId]+idx];
                        VertexId color = gc.colors[neighbor];
                        if(color >= 131072) printf("Surprising color exceed range %d\n", color);
                        if(nodes[nodeId] >= 0) {
                            VertexId color_forbid = color-colorRange;
                            if(color!=0xffffffff && color_forbid >= 0 && color_forbid < 131072)
                                atomicOr(forbiddenColors+color_forbid/32, (1<<(color_forbid%32)));
                        }
                        else {
                            VertexId temp_node = (-1)*(nodes[nodeId])-1;
                            ifpush = ( (scan[1] == color) && 
                                       ( (neighbor > temp_node && gc.ifpushed[neighbor] == false) ||
                                             (temp_node > neighbor && atomicAdd(ifpush_process+nodeId, 1) == 0) )
                                     );
                            neighbor = max(neighbor, temp_node);
                        }
                    }
                    if(nodes[nodeId] < 0)
                    {
                        gc.worklists.push_cta(ifpush, neighbor+1);
                        if(ifpush) gc.ifpushed[neighbor] = true;
                        __syncthreads();
                    }
                }
                
                if(nodes[nodeId] <= 0)
                    break;

                scan[1] = 0x7fffffff;
                __syncthreads();

                for(int item=threadIdx.x; item-threadIdx.x < MAPSIZE*FETCH_SIZE; item+=blockDim.x) {
                    unsigned mask = 0;
                    if(item < MAPSIZE*FETCH_SIZE) {
                        mask = forbiddenColors[item];
                        mask = ~mask;
                    }
                    unsigned found_mask = __ballot_sync(0xffffffff, (mask!=0));
                    if(found_mask!=0) {
                        if(LANE_ == __ffs(found_mask)-1) {
                            atomicMin(scan+1, __ffs(mask)-1+item*32);
                        }
                    }
                    __syncthreads(); //MAPSIZE*FETCH_SIZE is not multiple of blockDim.x, may deadlock 
                    if(scan[1]!=0x7fffffff)
                        break;
                }

                if(scan[1]!=0x7fffffff)
                {
                    if(threadIdx.x == 0) {
                        //printf("check point 2 nodeID %d, node %d neighborLen %d offset %d\n", nodeId, nodes[nodeId]-1, len, node_offset[nodeId]);
                        //atomicAdd(gc.pushNext+nodes[nodeId]-1, 1);
                        gc.colors[nodes[nodeId]-1]=scan[1]+colorRange;
                        gc.ifpushed[nodes[nodeId]-1] = false;
                    }
                    break;
                }
            }
            //assert(scan[1]!=0x7fffffff);
            if(threadIdx.x < size) {
                if(neighborLen[threadIdx.x]> MAPSIZE*32)
                    atomicMin(scan, threadIdx.x);
            }
            __syncthreads();
        }

        bool ifpush = (threadIdx.x < size && nodes[threadIdx.x] > 0); 
        VertexId push_item = ifpush? nodes[threadIdx.x]:0;
        gc.worklists.push_cta(ifpush, push_item*(-1));
    }
};

template<int FETCH_SIZE, int MAPSIZE, typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCCTA_simple{
public:
    __forceinline__ __device__ void operator()(VertexId *nodes, int size,  GC_Async<VertexId,SizeT, QueueT> gc)
    {
        __shared__ SizeT node_offset[FETCH_SIZE];
        __shared__ SizeT neighborLen[FETCH_SIZE];
        if(threadIdx.x < size) {
            node_offset[threadIdx.x] = gc.csr_offset[abs(nodes[threadIdx.x])-1];
            neighborLen[threadIdx.x] = gc.csr_offset[abs(nodes[threadIdx.x])]-node_offset[threadIdx.x];
        }
        for(int iter = 0; iter<size; iter++) {
            VertexId node = nodes[iter];
            if(node == 0)
                printf("tid %d, task %d\n", TID, node);
            if(node > 0)  // assign color
            {
                __shared__ uint32_t forbiddenColors[MAPSIZE];
                __shared__ VertexId newColor;
                node = node-1;
                if(threadIdx.x == 0) 
                    newColor = 0x7fffffff;

                for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=MAPSIZE*32)
                {
                    for(int item = threadIdx.x; item < MAPSIZE; item+=blockDim.x)
                        forbiddenColors[item] = 0;
                    __syncthreads();

                    for(int item = threadIdx.x; item<neighborLen[iter]; item=item+blockDim.x)
                    {
                        VertexId neighbor = gc.csr_indices[node_offset[iter]+item];
                        VertexId color = gc.colors[neighbor];
                        VertexId color_forbid = color-colorRange;
                        if(color!=0xffffffff && color_forbid >= 0 && color_forbid < MAPSIZE*32)
                            atomicOr(forbiddenColors+color_forbid/32, (1<<(color_forbid%32)));
                    }
                    __syncthreads();
                    for(int item = threadIdx.x; item-threadIdx.x < MAPSIZE; item = item+blockDim.x)
                    {
                        unsigned mask = 0;
                        if(item < MAPSIZE) {
                            mask = forbiddenColors[item];
                            mask = (~mask);
                        }

                        unsigned found_mask = __ballot_sync(0xffffffff, (mask!=0));
                        if(found_mask!=0) {
                            if(LANE_ == __ffs(found_mask)-1) {
                                atomicMin(&newColor, colorRange+__ffs(mask)-1+item*32);
                            }
                        }
                        __syncthreads();
                        if(newColor!=0x7fffffff)
                            break;
                    }
                    if(newColor!=0x7fffffff)
                        break;
                }
                assert(newColor != 0x7fffffff);
                if(newColor!=0x7fffffff) {
                    if(threadIdx.x == 0) {
                        gc.colors[node] = newColor;
                        atomicAdd(gc.pushNext+node, 1);
                        gc.ifpushed[node] = false;
                        gc.worklists.push_warp((node+1)*(-1));
                        assert((node+1)*(-1)!=0);
                    }
                }
            }
            else if(node < 0)  // conflict detection
            {
                __shared__ VertexId color;
                __shared__ bool node_notadd;
                __shared__ int whoPushNode;
                node = node*(-1)-1;
                if(threadIdx.x == 0) {
                    whoPushNode = -1;
                    color = gc.colors[node];
                    node_notadd = true;
                }
                __syncthreads();
                for(VertexId item=threadIdx.x; item-threadIdx.x < neighborLen[iter]; item=item+blockDim.x)
                {
                    bool ifpushnew = false;
                    VertexId neighbor;
                    if(item < neighborLen[iter]) {
                        neighbor = gc.csr_indices[node_offset[iter]+item];
                        VertexId neighbor_color = gc.colors[neighbor];
                        ifpushnew = ((neighbor_color == color) && 
                                     (((node > neighbor) && node_notadd) || ((neighbor > node) && (gc.ifpushed[neighbor] == false)))
                                    ) ;
                        neighbor = max(neighbor, node);
                    }
                    if(node_notadd) {
                        unsigned mask = __ballot_sync(0xffffffff, (ifpushnew && (neighbor == node)));
                        if( mask != 0)
                        {
                            if(ifpushnew && (neighbor ==  node) && LANE_ == __ffs(mask)-1)
                                atomicMin(&whoPushNode, int(threadIdx.x));
                        }
                        __syncthreads();
                        if(whoPushNode != -1 && (ifpushnew && (neighbor == node))) {
                            if(threadIdx.x == whoPushNode) {
                                node_notadd = false;
                            }
                            else ifpushnew = false;
                        }
                    }
                    __syncthreads();
                    gc.worklists.push_cta(ifpushnew, neighbor+1);
                    if(ifpushnew) {
                        //atomicAdd(gc.pushNext+neighbor, 1);
                        gc.ifpushed[neighbor] = true;
                        assert(neighbor+1!=0);
                    }
                    __syncthreads();
                }
            }
            // though there supposed no 0, but sometimes get 0 for unknow reason, comment still get right resuls
            //else {
            //    printf("node == %d Fail\n", node);
            ////    assert(0);
            //} 
        }
    }
};

template<int FETCH_SIZE, typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCCTA_mesh{
public:
    __forceinline__ __device__ void operator()(VertexId *nodes, int size, GC_Async<VertexId, SizeT, QueueT> gc)
    {
        __shared__ SizeT node_offset[FETCH_SIZE];   
        __shared__ SizeT neighborLen[FETCH_SIZE+1];
        __shared__ uint32_t forbiddenColors[FETCH_SIZE];  // each node will get 1 uint32_t as map which is 32 length
        __shared__ int ifpush_process[FETCH_SIZE];
        __shared__ int accesstimes[FETCH_SIZE];
    
        if(threadIdx.x < size)
        {
            VertexId node = abs(nodes[threadIdx.x]);
            if(node == 0) printf("Pop a node 0\n");
            node_offset[threadIdx.x] = gc.csr_offset[node-1];
            neighborLen[threadIdx.x] = gc.csr_offset[node]-node_offset[threadIdx.x];
            //printf("TID %d, tid %d node %d, node_offset %d, len %d\n", TID, threadIdx.x, node, node_offset[threadIdx.x], neighborLen[threadIdx.x]);
            ifpush_process[threadIdx.x] = 0;
            forbiddenColors[threadIdx.x] = 0;
            accesstimes[threadIdx.x] =0;
        }
        __syncthreads();

        typedef cub::BlockScan<int, 512, cub::BLOCK_SCAN_RAKING,1,1,700> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;
        int block_aggregate;
        int thread_data = 0;
        if(threadIdx.x < size)
            thread_data = neighborLen[threadIdx.x];
        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
        __syncthreads();
        if(threadIdx.x < size)
            neighborLen[threadIdx.x] = thread_data;
        if(threadIdx.x == 0)
            neighborLen[size] = block_aggregate;
        __syncthreads();

        int total_slots = ((block_aggregate+31)>>5);
        int slots_warp = ceil(float(total_slots)/(blockDim.x>>5));
        int start_index=FETCH_SIZE-1;
        for(int item = LANE_; item<FETCH_SIZE; item = item+32)
        {
            unsigned mask_start  = __ballot_sync(0xffffffff, neighborLen[item]> (threadIdx.x>>5)*32*slots_warp);
            if(mask_start!=0)
            {
                start_index = item-LANE_+ (__ffs(mask_start)-2);
                break;
            }
        }        

        int abs_neighbors = (threadIdx.x>>5)*32*slots_warp;
        int cur_node_idx=start_index;
        int cur_node_offset;        
        bool ifpush = false;
        VertexId neighbor = -1;
        for(int iter=0; iter<slots_warp; iter++)
        {
            if(abs_neighbors+LANE_ < block_aggregate)
            {
                while(abs_neighbors+LANE_ >= neighborLen[cur_node_idx+1])
                    cur_node_idx++;
                cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx]; // what if next node's neighborlen is 0
                neighbor = gc.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
                //printf("TID %d, tid %d, node %d, cur_node_idx %d, cur_node_offset %d\n", TID, threadIdx.x, nodes[cur_node_idx], cur_node_idx, cur_node_offset);
                VertexId color = gc.colors[neighbor];
                if(nodes[cur_node_idx] > 0) {
                    if(color > 12) printf("ERROR, node %d, neighbor %d neighbor color %d\n", nodes[cur_node_idx]-1, neighbor, color);
                    if(color != 0xffffffff && color < 32 && color >=0) {
                        atomicOr(forbiddenColors+cur_node_idx, (1<<(color%32)));
                    }
                    atomicAdd(accesstimes+cur_node_idx,1);
                    //printf("TID %d, tid %d, node %d, add 1 to cur_node_idx %d\n", TID, threadIdx.x, nodes[cur_node_idx], cur_node_idx);
                }
                else {
                    VertexId temp_node = nodes[cur_node_idx]*(-1)-1;
                    ifpush = ( (gc.colors[temp_node] == color) && 
                               ( (neighbor > temp_node && gc.ifpushed[neighbor] == false) || 
                                 (temp_node > neighbor && atomicAdd(ifpush_process+cur_node_idx, 1) == 0) ) );
                    neighbor = max(neighbor, temp_node);
                    atomicAdd(accesstimes+cur_node_idx,1);
                }
            }
            assert(!ifpush || (ifpush && nodes[cur_node_idx] < 0));
            assert(!ifpush || (ifpush && (neighbor+1) > 0));
            gc.worklists.push_cta(ifpush, neighbor+1);
            if(ifpush) gc.ifpushed[neighbor] = true;
            __syncthreads();
            abs_neighbors = abs_neighbors+32;
            neighbor = -1;
            ifpush = false;
        }
        __syncthreads();
        if(threadIdx.x < size) {
            assert(accesstimes[threadIdx.x] == neighborLen[threadIdx.x+1]-neighborLen[threadIdx.x]);
            assert(ifpush_process[threadIdx.x] <= neighborLen[threadIdx.x+1]-neighborLen[threadIdx.x]);
        }
        
        __syncthreads();
        ifpush = false;
        if(threadIdx.x < size && nodes[threadIdx.x] > 0) {
            unsigned mask = forbiddenColors[threadIdx.x];
            mask = (~mask);
            assert(mask!=0);
            gc.colors[nodes[threadIdx.x]-1] = __ffs(mask)-1;
            assert(__ffs(mask)-1 <=12);
            gc.ifpushed[nodes[threadIdx.x]-1] = false;
            ifpush = true;
            neighbor = nodes[threadIdx.x]*(-1);
        }
        gc.worklists.push_cta(ifpush, neighbor);
    }
};

//FETCH_SIZE == number of warps in the CTA
template<int FETCH_SIZE, typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCCTA_simple2 {
public:
    __forceinline__ __device__ void operator()(VertexId *nodes, int size, GC_Async<VertexId, SizeT, QueueT> gc)
    {
        if((threadIdx.x>>5) < size) {
            VertexId node = nodes[(threadIdx.x>>5)];
            if(node == 0) 
                printf("tid %d, warpid %d, size %d, task %d\n", TID, (threadIdx.x>>5), size, node);
            if(node > 0) // assign color
            {
                __shared__ uint32_t forbiddenColors[FETCH_SIZE*64];
                node = node-1;
                SizeT node_offset = gc.csr_offset[node];
                SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
                unsigned found_mask = 0;

                for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=2048)
                {
                    forbiddenColors[LANE_+(threadIdx.x>>5)*64] = 0;
                    forbiddenColors[LANE_+32+(threadIdx.x>>5)*64] = 0;
                    __syncwarp();

                    for(int item = LANE_; item<neighborlen; item=item+32)
                    {
                        VertexId neighbor = gc.csr_indices[node_offset+item];
                        VertexId color = gc.colors[neighbor];
                        VertexId color_forbid = color-colorRange;
                        if(color!=0xffffffff && color_forbid >= 0 && color_forbid < 2048)
                            atomicOr(forbiddenColors+(threadIdx.x>>5)*64+color_forbid/32, (1<<(color_forbid%32)));
                    }
                    __syncwarp();
                    for(int item = LANE_; item < 64; item = item+32)
                    {
                        unsigned mask = forbiddenColors[(threadIdx.x>>5)*64+item];
                        mask = (~mask);
                        found_mask = __ballot_sync(0xffffffff, (mask!=0));
                        if(found_mask!=0) {
                            if(LANE_ == __ffs(found_mask)-1) {
                                gc.colors[node] = colorRange+item*32+__ffs(mask)-1;
                                atomicAdd(gc.pushNext+node, 1);
                            }
                            break;
                        }
                    }
                    if(found_mask != 0)
                        break;
                }
                assert(found_mask != 0);
                __syncwarp();
                if(LANE_ == 0) {
                    gc.ifpushed[node] = false;
                    gc.worklists.push_warp((node+1)*(-1));
                    assert((node+1)*(-1)!=0);
                }
                __syncwarp();
            }
            else if(node < 0) // conflict detection
            {
                node = node*(-1)-1;
                VertexId color = gc.colors[node];
                SizeT node_offset = gc.csr_offset[node];
                SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
                bool node_notadd = true;
                for(VertexId item=LANE_; item-LANE_ < neighborlen; item=item+32)
                {
                    bool ifpushnew = false;
                    VertexId neighbor;
                    if(item < neighborlen) {
                        neighbor = gc.csr_indices[node_offset+item];
                        VertexId neighbor_color = gc.colors[neighbor];
                        ifpushnew = ((neighbor_color == color) && 
                                     (((node > neighbor) && node_notadd) || ((neighbor > node) && (gc.ifpushed[neighbor] == false)))
                                    ) ;
                        neighbor = max(neighbor, node);
                    }
                    if(node_notadd) {
                        unsigned mask = __ballot_sync(0xffffffff, (ifpushnew && (neighbor == node)));
                        if(__popc(mask) > 0)
                        {
                            node_notadd = false;
                            if(ifpushnew && (neighbor == node) && LANE_ != __ffs(mask)-1)
                                ifpushnew = false;
                        }
                    }
                    __syncwarp(0xffffffff);
                    if(ifpushnew) {
                        //atomicAdd(gc.pushNext+neighbor, 1);
                        gc.worklists.push_warp(neighbor+1);
                        gc.ifpushed[neighbor] = true;
                        assert(neighbor+1!=0);
                    }
                    __syncwarp(0xffffffff);
                }
            }
        }
    }
};

template<typename VertexId, typename SizeT, typename QueueT=uint32_t>
__global__ void GCWarp_kernel(QueueT start, int size, GC_Async<VertexId, SizeT, QueueT> gc)
{
    if(((TID)>>5) < size) {
        QueueT start_local = start + ((TID)>>5);
        VertexId node = gc.worklists.get_item(start_local, 0);
        if(node == 0 && LANE_ == 0)
            printf("queue[%d] task is 0\n", start_local);
        if(node > 0) // assign color
        {
            extern __shared__ uint32_t forbiddenColors[]; // blockDim.x/32*64
            node = node-1;
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            unsigned found_mask = 0;

            for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=2048)
            {
                forbiddenColors[LANE_+(threadIdx.x>>5)*64] = 0;
                forbiddenColors[LANE_+32+(threadIdx.x>>5)*64] = 0;
                __syncwarp();

                for(int item = LANE_; item<neighborlen; item=item+32)
                {
                    VertexId neighbor = gc.csr_indices[node_offset+item];
                    VertexId color = gc.colors[neighbor];
                    VertexId color_forbid = color-colorRange;
                    if(color!=0xffffffff && color_forbid >= 0 && color_forbid < 2048)
                        atomicOr(forbiddenColors+(threadIdx.x>>5)*64+color_forbid/32, (1<<(color_forbid%32)));
                }
                __syncwarp();
                for(int item = LANE_; item < 64; item = item+32)
                {
                    unsigned mask = forbiddenColors[(threadIdx.x>>5)*64+item];
                    mask = (~mask);
                    found_mask = __ballot_sync(0xffffffff, (mask!=0));
                    if(found_mask!=0) {
                        if(LANE_ == __ffs(found_mask)-1) {
                            gc.colors[node] = colorRange+item*32+__ffs(mask)-1;
                            atomicAdd(gc.pushNext+node, 1);
                        }
                        break;
                    }
                }
                if(found_mask != 0)
                    break;
            }
            assert(found_mask != 0);
            __syncwarp();
            if(LANE_ == 0) {
                gc.ifpushed[node] = false;
                gc.worklists.push_warp((node+1)*(-1));
                assert((node+1)*(-1)!=0);
            }
            __syncwarp();
        }
        else if(node < 0)  // conflict detection
        {
            node = node*(-1)-1;
            VertexId color = gc.colors[node];
            SizeT node_offset = gc.csr_offset[node];
            SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
            bool node_notadd = true;
            for(VertexId item=LANE_; item-LANE_ < neighborlen; item=item+32)
            {
                bool ifpushnew = false;
                VertexId neighbor;
                if(item < neighborlen) {
                    neighbor = gc.csr_indices[node_offset+item];
                    VertexId neighbor_color = gc.colors[neighbor];
                    ifpushnew = ((neighbor_color == color) && 
                                 (((node > neighbor) && node_notadd) || ((neighbor > node) && (gc.ifpushed[neighbor] == false)))
                                ) ;
                    neighbor = max(neighbor, node);
                }
                if(node_notadd) {
                    unsigned mask = __ballot_sync(0xffffffff, (ifpushnew && (neighbor == node)));
                    if(__popc(mask) > 0)
                    {
                        node_notadd = false;
                        if(ifpushnew && (neighbor == node) && LANE_ != __ffs(mask)-1)
                            ifpushnew = false;
                    }
                }
                __syncwarp(0xffffffff);
                if(ifpushnew) {
                    //atomicAdd(gc.pushNext+neighbor, 1);
                    gc.worklists.push_warp(neighbor+1);
                    gc.ifpushed[neighbor] = true;
                    assert(neighbor+1!=0);
                }
                __syncwarp(0xffffffff);
            }
        }
    }
}

template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename QueueT=uint32_t>
class GCWarp_discrete {
public:
    __forceinline__ void operator()(QueueT start, int size, uint32_t shareMem_size, cudaStream_t &stream, GC_Async<VertexId, SizeT, QueueT> gc)
    {
        int numWaprPerBlock = BLOCK_SIZE/32;
        int gridSize = (size+numWaprPerBlock-1)/numWaprPerBlock;
        //std::cout << "Block size "<< BLOCK_SIZE <<" Grid size " << gridSize << " share mem "<< shareMem_size << std::endl;
        GCWarp_kernel<<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>(start, size, gc);
    }
};

template<typename Functor, typename T, typename P>
__forceinline__ __device__ void func(Functor F, T t, P p)
{
    F(t, p);
}

template<typename Functor, typename T, typename P>
__forceinline__ __device__ void func(Functor F, T t, int size, P p)
{
    F(t, size, p);
}

template<typename Functor, typename T, typename P>
__forceinline__ __host__ void func(Functor F, T start, int size, uint32_t share, cudaStream_t &stream, P p)
{
    F(start, size, share, stream, p);
}

template<typename VertexId, typename SizeT, typename QueueT>
__global__ void detection(GC_Async<VertexId, SizeT, QueueT> gc, int *workload)
{
    for(VertexId vid = WARPID; vid < gc.nodes; vid+=(gridDim.x*blockDim.x/32))
    {
        VertexId node_color = gc.colors[vid];
        //if(node_color == 0xffffffff && LANE_ == 0)
        //    printf("node %d, color is %d\n", vid, node_color);
        //assert(node_color!=0xffffffff);
        SizeT node_offset = gc.csr_offset[vid];
        SizeT neighborlen = gc.csr_offset[vid+1]-node_offset;
        bool addvid = false;
        for(int item=LANE_; item<neighborlen; item=item+32)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(neighbor_color == node_color && neighbor < vid)
            {
                addvid = true;
                //if(gc.pushNext[vid] == 0)
                //    printf("vertex %d is supposed to be push to queue\n", vid);
            }
        }
        if(__popc(__ballot_sync(0xffffffff, addvid)) > 0 && LANE_ == 0)
            atomicAdd(workload, 1);
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
VertexId GC_Async<VertexId, SizeT, QueueT>::getWorkload()
{
    VertexId totalWork;
    CUDA_CHECK(cudaMemcpy(&totalWork, (void *)(worklists.start), sizeof(QueueT), cudaMemcpyDeviceToHost));
    VertexId *tasks = (VertexId *)malloc(sizeof(VertexId)*totalWork);
    CUDA_CHECK(cudaMemcpy(tasks, worklists.queue, sizeof(VertexId)*totalWork, cudaMemcpyDeviceToHost));
    VertexId workload = 0;
    printf("workload %d\n", totalWork);
    for(VertexId i=0; i<totalWork; i++)
        if(tasks[i] > 0)
            workload++;
        else if(tasks[i] == 0)
            printf("%dth item in the queue is 0\n", i);
    return workload;
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::outputVistTimes(std::string file_name = "freq.txt")
{
    VertexId totalWork;
    CUDA_CHECK(cudaMemcpy(&totalWork, (void *)(worklists.start), sizeof(QueueT), cudaMemcpyDeviceToHost));
    VertexId *tasks = (VertexId *)malloc(sizeof(VertexId)*totalWork);
    CUDA_CHECK(cudaMemcpy(tasks, worklists.queue, sizeof(VertexId)*totalWork, cudaMemcpyDeviceToHost)); 
    VertexId *nodeFreq = (VertexId *)malloc(sizeof(VertexId)*nodes);
    for(VertexId i=0; i<nodes; i++)
        nodeFreq[i] = 0;
    for(VertexId i=0; i<totalWork; i++)
    {
        if(tasks[i] > 0)
            nodeFreq[tasks[i]-1] = nodeFreq[tasks[i]-1]+1; 
        else if(tasks[i]==0)
            printf("%dth item on the queue is 0\n", i);
    }
    VertexId *h_pushNext = (VertexId *)malloc(sizeof(VertexId)*nodes);
    CUDA_CHECK(cudaMemcpy(h_pushNext, pushNext, sizeof(VertexId)*nodes, cudaMemcpyDeviceToHost));
    for(VertexId i=0; i<nodes; i++)
        if(h_pushNext[i]!=nodeFreq[i])
            printf("node %i, nodeFreq %d, pushNest %d\n", i, nodeFreq[i], h_pushNext[i]);
    std::ofstream myfile (file_name);
    if(myfile.is_open())
    {
        for(int count=0; count < nodes; count++)
            myfile << nodeFreq[count] << std::endl;
        myfile.close();
    }
    std::cout <<"Finish writing node freqency data out to file " << file_name << std::endl; 
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::outputNeighborLen(std::string file_name = "neighborLen.txt")
{
    VertexId *nodeLen= (VertexId *)malloc(sizeof(VertexId)*nodes);
    for(VertexId i=0; i<nodes; i++)
        nodeLen[i] = csr_offset[i+1]-csr_offset[i];

    std::ofstream myfile (file_name);
    if(myfile.is_open())
    {
        for(int count=0; count < nodes; count++)
            myfile << nodeLen[count] << std::endl;
        myfile.close();
    }
    std::cout <<"Finish writing node length data out to file " << file_name << std::endl; 
}

template<typename VertexId, typename SizeT, typename QueueT>
void GC_Async<VertexId, SizeT, QueueT>::GCStart(int numBlock, int numThread)
{
    //worklists.launchWarpPer32Items_minIter(numBlock, numThread, GCWarp<VertexId, SizeT>(), *this);
    worklists.launchWarpPer32Items_minIter(numBlock, numThread, GCWarp_op<VertexId, SizeT>(), *this);
    worklists.sync_all_wl();
    //int * h_pushNext = (int *)malloc(sizeof(int)*nodes);
    //CUDA_CHECK(cudaMemcpy(h_pushNext, pushNext, sizeof(int)*nodes, cudaMemcpyDeviceToHost));
    //int nextwork = 0;
    //int extrawork = 0;
    //for(int i=0; i<nodes; i++) {
    //    nextwork += h_pushNext[i];
    //    if(h_pushNext[i] > 1)
    //        extrawork += h_pushNext[i]-1;
    //}
    //printf("next workload %d, extra work %d, %d\n", nextwork, extrawork, nextwork-extrawork);
    //int *workload;
    //CUDA_CHECK(cudaMalloc(&workload, sizeof(int)));
    //CUDA_CHECK(cudaMemset(workload, 0, sizeof(int)));
    //CUDA_CHECK(cudaDeviceSynchronize());
    //detection<<<320, 512>>>(*this, workload);
    //CUDA_CHECK(cudaDeviceSynchronize());
    //CUDA_CHECK(cudaMemcpy(&nextwork, workload, sizeof(int), cudaMemcpyDeviceToHost));
    //CUDA_CHECK(cudaDeviceSynchronize());
    //printf("detect next workload %d\n", nextwork);
}

template<typename VertexId, typename SizeT, typename QueueT>
template<int FETCH_SIZE>
void GC_Async<VertexId, SizeT, QueueT>::GCStart_CTA(int numBlock, int numThread, bool ifmesh=0)
{
    
    //VertexId *color_check = (VertexId *)malloc(sizeof(VertexId)*nodes);
    //CUDA_CHECK(cudaMemcpy(color_check, colors, sizeof(VertexId)*nodes, cudaMemcpyDeviceToHost));
    //for(int i=0; i<nodes; i++)
    //    assert(color_check[i] == 0xffffffff);
    //free(color_check);
    //printf("PASS color check before launch queue\n");
    if(ifmesh)
        worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread,  0, GCCTA_mesh<FETCH_SIZE, VertexId, SizeT>(), *this);
    else 
        worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread,  0, GCCTA<FETCH_SIZE, VertexId, SizeT>(), *this);
    //worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread,  0, GCCTA_simple<FETCH_SIZE, 1, VertexId, SizeT>(), *this);
    //worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread,  0, GCCTA_simple2<FETCH_SIZE, VertexId, SizeT>(), *this);
    worklists.sync_all_wl();
    //bool *check_ifpush = (bool *)malloc(sizeof(bool)*nodes);
    //CUDA_CHECK(cudaMemcpy(check_ifpush, ifpushed, sizeof(bool)*nodes, cudaMemcpyDeviceToHost));
    //int need_process = 0;
    //for(int i=0; i<nodes; i++)
    //    if(check_ifpush[i] == true)
    //        need_process++;
    //        //printf("%dth item ifpushed is not false %d\n", i, check_ifpush[i]);
    //printf("need to process next turn %d\n", need_process);
    //free(check_ifpush);
    //int nextwork = 0;
    //int *workload;
    //CUDA_CHECK(cudaMalloc(&workload, sizeof(int)));
    //CUDA_CHECK(cudaMemset(workload, 0, sizeof(int)));
    //CUDA_CHECK(cudaDeviceSynchronize());
    //detection<<<320, 512>>>(*this, workload);
    //CUDA_CHECK(cudaDeviceSynchronize());
    //CUDA_CHECK(cudaMemcpy(&nextwork, workload, sizeof(int), cudaMemcpyDeviceToHost));
    //CUDA_CHECK(cudaDeviceSynchronize());
    //printf("detect next workload %d\n", nextwork);
    //int *check = (int *)malloc(sizeof(int)*nodes);
    //CUDA_CHECK(cudaMemcpy(check, pushNext, sizeof(int)*4847571, cudaMemcpyDeviceToHost));
    //for(int i=0; i<nodes; i++)
    //    if(check[i] != 1)
    //        printf("%dth item is visited %d times\n", i, check[i]);
    //free(check);
}

template<typename VertexId, typename SizeT, typename QueueT>
template<int FETCH_SIZE>
void GC_Async<VertexId, SizeT, QueueT>::GCStart_CTA_simple(int numBlock, int numThread)
{
    worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread,  0, GCCTA_simple<FETCH_SIZE, 1, VertexId, SizeT>(), *this);
    worklists.sync_all_wl();
}

template<typename VertexId, typename SizeT, typename QueueT>
template<int FETCH_SIZE, int BLOCK_SIZE>
void GC_Async<VertexId, SizeT, QueueT>::GCStart_discrete()
{
    uint32_t shareMem = BLOCK_SIZE/32*64*sizeof(uint32_t);
    worklists.template launchDiscrete_minIter<FETCH_SIZE, BLOCK_SIZE>(shareMem, GCWarp_discrete<FETCH_SIZE, BLOCK_SIZE, VertexId, SizeT, QueueT>(), *this);
}
#endif

