#ifndef BreadthFirstSearch
#define BreadthFirstSearch

#include <math.h>

#include "../../util/error_util.cuh"
#include "../../util/util.cuh"
#include "../../comm/csr.cuh"
#include "../queue/queue.cuh"
//#include "../queue/queue.cuh"

#include <cub/cub.cuh> 

template<typename VertexId>
struct BFSEntry
{
    VertexId node;
    VertexId dep;

    __host__ __device__ BFSEntry(){}
    __host__ __device__ BFSEntry(VertexId _node, VertexId _dep) {node = _node; dep = _dep;}
};

template<typename VertexId, typename SizeT, typename QueueT=uint32_t>
struct BFS
{
    VertexId nodes;
    SizeT edges;

    SizeT *csr_offset;
    VertexId *csr_indices;

    VertexId *depth;

    MaxCountQueue::Queue<VertexId, QueueT> worklists;

    BFS(Csr<VertexId, SizeT> &csr, uint32_t min_iter=800)
    {
        nodes = csr.nodes;
        edges = csr.edges;
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;
        worklists.init(QueueT(nodes*2), min_iter);
        std::cout << "wl_size: "<<worklists.get_capacity()<<std::endl;
        std::cout << "min_iter: "<< worklists.min_iter << std::endl;

        CUDA_CHECK(cudaMallocManaged(&depth, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(depth, (nodes+1), sizeof(VertexId)*nodes));
    }

    void release()
    {
        worklists.release();
        CUDA_CHECK(cudaFree(depth));
    }

    void reset()
    {
        worklists.reset();
    }

    void BFSInit(VertexId source, int numBlock, int numThread);
    void BFSInit(VertexId *source, int size, int numBlock, int numThread);

    template<int FETCH_SIZE, int BLOCK_SIZE>
    void BFSStart(QueueT start, int size, int shareMem_size, cudaStream_t stream);

};

template<typename VertexId, typename SizeT>
__global__ void pushSource(BFS<VertexId, SizeT> bfs, VertexId source)
{
    
    if( LANE_ ==0)
    {
        bfs.worklists.push_warp(source);
        bfs.depth[source ] = 0;
    }
}

template<typename VertexId, typename SizeT>
__global__ void pushSource_wrong(BFS<VertexId, SizeT> bfs, VertexId *source, int size)
{
    
    if( TID < size)
    {
//        bfs.worklists.push_warp(source[TID]);
        bfs.worklists.queue[TID] = source[TID];
    }
    if(TID ==0)
    {
        *bfs.worklists.end = size;
        *bfs.worklists.end_count = size;
        *bfs.worklists.end_max = size;
        *bfs.worklists.end_alloc= size;
    }
}

template<typename VertexId, typename SizeT>
__global__ void initDepth(BFS<VertexId, SizeT> bfs)
{
    for(int i=TID; i<bfs.nodes; i=i+gridDim.x*blockDim.x)
        bfs.depth[i] = (bfs.nodes+1);
}

template<typename Queue>
__global__ void checkEnd(Queue bq)
{
    if(*bq.end != *bq.end_alloc)
    {
        if(*bq.end_max == *bq.end_count && *bq.end_count == *bq.end_alloc)
            *bq.end = *bq.end_alloc;
        //else if(threadIdx.x == 0)
        //    printf("bqueue end update error: end %d, end_alloc %d, end_count %d, end_max %d\n", *bq.end, *bq.end_alloc, *bq.end_count, *bq.end_max);
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
void BFS<VertexId, SizeT, QueueT>::BFSInit(VertexId source, int numBlock, int numThread)
{
    int gridSize=320; int blockSize=512;
    CUDA_CHECK(cudaMallocManaged(&worklists.execute, sizeof(QueueT)*numBlock));
    initDepth<<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
//    pushSource_wrong<<<gridSize, blockSize>>>(*this, source);
    pushSource<<<1, 32>>>(*this, source);
    CUDA_CHECK(cudaDeviceSynchronize());
    MaxCountQueue::checkEnd<<<1, 32>>>(worklists);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename VertexId, typename SizeT, typename QueueT>
void BFS<VertexId, SizeT, QueueT>::BFSInit(VertexId *source, int size, int numBlock, int numThread)
{
    int gridSize=320; int blockSize=512;
    CUDA_CHECK(cudaMallocManaged(&worklists.execute, sizeof(QueueT)*numBlock));
    initDepth<<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
    pushSource_wrong<<<gridSize, blockSize>>>(*this, source, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    MaxCountQueue::checkEnd<<<1, 32>>>(worklists);
    CUDA_CHECK(cudaDeviceSynchronize());
}


template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename QueueT>
__global__ void BFSCTA(QueueT start, int size, BFS<VertexId, SizeT, QueueT> bfs)
{
    __shared__ SizeT node_offset[FETCH_SIZE];
    __shared__ SizeT neighborLen[FETCH_SIZE+1];
    __shared__ VertexId depths[FETCH_SIZE];
    QueueT start_local = start + blockIdx.x*FETCH_SIZE;
    int size_local = min(FETCH_SIZE, size-blockIdx.x*FETCH_SIZE);

    if(threadIdx.x <  size_local)
    {
	VertexId node = bfs.worklists.get_item(start_local+threadIdx.x);
        depths[threadIdx.x] = ((volatile VertexId *)bfs.depth)[node];
        node_offset[threadIdx.x] = bfs.csr_offset[node];
        neighborLen[threadIdx.x] = bfs.csr_offset[node+1]-node_offset[threadIdx.x];
    }
    __syncthreads();
    
    typedef cub::BlockScan<int, BLOCK_SIZE, cub::BLOCK_SCAN_RAKING,1,1,700> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    int thread_data = 0;
    if(threadIdx.x < size_local)
        thread_data = neighborLen[threadIdx.x];
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
    __syncthreads();
    if(threadIdx.x < size_local)
        neighborLen[threadIdx.x] = thread_data;
    if(threadIdx.x == 0)
        neighborLen[size_local] = block_aggregate;
    __syncthreads();

    int total_slots = ((block_aggregate+31)>>5);
    int slots_warp = ceil(float(total_slots)/(blockDim.x>>5));
    int start_index=-1;
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
            neighbor = bfs.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
            VertexId old_depth = atomicMin(bfs.depth+neighbor, depths[cur_node_idx]+1);
            if(old_depth > depths[cur_node_idx]+1)
                ifpush = true;
        }
  //      __syncthreads();
        bfs.worklists.push_cta(ifpush, neighbor);
  //      __syncthreads();
        abs_neighbors = abs_neighbors+32;
        neighbor = -1;
        ifpush = false;
    }
}

template<typename VertexId, typename SizeT, typename QueueT>
template<int FETCH_SIZE, int BLOCK_SIZE>
void BFS<VertexId, SizeT, QueueT>::BFSStart(QueueT start, int size, int shareMem_size, cudaStream_t stream)
{
    int gridSize = (size+FETCH_SIZE-1)/FETCH_SIZE;
//    std::cout << "Fetch size: "<< FETCH_SIZE << " Grid Size: "<< gridSize << " Block Size: "<< BLOCK_SIZE << " Share Mem: "<< shareMem_size << " Start: "<< start << " Size: "<< size << std::endl;
    BFSCTA<FETCH_SIZE, BLOCK_SIZE><<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>(start, size, *this);
}


#endif
