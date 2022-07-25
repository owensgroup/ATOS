#ifndef PAGE_RANK
#define PAGE_RANK
#include <limits>
#include "../../util/error_util.cuh"
#include "../../util/util.cuh"
#include "../../comm/csr.cuh"
#include "../queue/queue.cuh"

#include <cub/cub.cuh>

__device__ float inf[1];

template<typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t>
struct PageRank
{
    size_t nodes;             //total number of nodes
    size_t edges;                //total number of edges
    Rank * rank;              //rank value for each node, length: nodes
    Rank * res;               //residual value used to update rank value, length: nodes

    float lambda;               //probability to be activited
    float epsilon;              //stop threshold

    //csr used in the main loop
    SizeT * csr_offset;       //csr row offset array on device
    VertexId * csr_indices;   //csr column indices array on device

    //use a worklist push-based version:http://www.cs.utexas.edu/~inderjit/public_papers/scalable_pagerank_europar15.pdf
    MaxCountQueue::Queue<VertexId, QueueT> worklists;

    PageRank(Csr<VertexId, SizeT> &csr, float _lambda, float _epsilon, uint32_t min_iter=800)
    {
        nodes = csr.nodes;
        edges = csr.edges;
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;
        lambda = _lambda;
        epsilon = _epsilon;
        //worklists.init(QueueT(nodes*32), min_iter);
        worklists.init(QueueT(nodes*16), min_iter);
        std::cout << "wl_size: "<<worklists.get_capacity()<<std::endl;

        CUDA_CHECK(cudaMallocManaged(&rank, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMallocManaged(&res, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*nodes));
        float h_inf = std::numeric_limits<float>::infinity(); 
        CUDA_CHECK(cudaMemcpyToSymbol(inf, &h_inf, sizeof(float), 0, cudaMemcpyHostToDevice));
    }

    void release() {
	worklists.release();
        CUDA_CHECK(cudaFree(rank));
        CUDA_CHECK(cudaFree(res));
    }

    void reset() {
        worklists.reset();
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*nodes));
    }

    void PrInit(uint32_t numBlock, uint32_t numThread);

    template<int FETCH_SIZE, int BLOCK_SIZE>
    void PrStart(QueueT start, int size, int shareMem_size, cudaStream_t stream);

    void normalize();

    void OutputRank()
    {
        std::ofstream myfile;
        myfile.open ("rankResult.txt");
        for(VertexId i=0; i<nodes; i++)
            myfile << rank[i] << " ";
        myfile << "\n";
        myfile.close();
    }
};

template<typename VertexId, typename SizeT, typename Rank, typename QueueT>
__global__ void _PrInit(PageRank<VertexId, SizeT, Rank, QueueT> pr)
{
    for(uint32_t i=TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
    {
        pr.rank[i] = 1.0-pr.lambda;
        SizeT col_start = pr.csr_offset[i];
        VertexId source_len = pr.csr_offset[i+1]-col_start;
        for(uint32_t j=0; j<source_len; j++)
        {
            VertexId dest = pr.csr_indices[col_start+j];
            Rank add_item = (1.0-pr.lambda)*pr.lambda/source_len;
            atomicAdd(pr.res+dest, add_item);
        }
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT>
__global__ void _PushVertices(PageRank<VertexId, SizeT, Rank, QueueT> pr)
{
   if(TID == 0) printf("%p\n", &(pr.nodes));
   for(VertexId i=TID; i-threadIdx.x<pr.nodes; i+=blockDim.x*gridDim.x)
   {
	 bool ifpush = (i<pr.nodes);
	 pr.worklists.push_cta(ifpush, i);
   }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT>
__global__ void _PushVertices2(PageRank<VertexId, SizeT, Rank, QueueT> pr)
{
   for(VertexId i=TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
   {
       pr.worklists.queue[i] = i;
   }
   *pr.worklists.end = pr.nodes;
   *pr.worklists.end_alloc = pr.nodes;
   *pr.worklists.end_max = pr.nodes;
   *pr.worklists.end_count = pr.nodes;
}

template<typename Queue>
__global__ void checkEnd(Queue bq)
{
    if(*bq.end != *bq.end_alloc)
    {
        if(*bq.end_max == *bq.end_count && *bq.end_count == *bq.end_alloc)
            *bq.end = *bq.end_alloc;
        else if(threadIdx.x == 0)
            printf("bqueue end update error: end %d, end_alloc %d, end_count %d, end_max %d\n", *bq.end, *bq.end_alloc, *bq.end_count, *bq.end_max);
    }
}
 
template<typename VertexId, typename SizeT, typename Rank, typename QueueT>
void PageRank<VertexId, SizeT, Rank, QueueT>::PrInit(uint32_t numBlock, uint32_t numThread)
{
    int gridSize = 320; int blockSize = 512;
    CUDA_CHECK(cudaMallocManaged(&worklists.execute, sizeof(QueueT)*numBlock));
    _PrInit<VertexId, SizeT, Rank><<<gridSize, blockSize>>>(*this);

    //Push all vertices and its out edges list length into the worklist (vertexId, listlength)
    _PushVertices2<VertexId, SizeT, Rank, QueueT><<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
    checkEnd<<<1, 1>>>(worklists);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT>
__global__ void PageRankCTA(QueueT start, int size, PageRank<VertexId, SizeT, Rank, QueueT> pr)
{
    __shared__ SizeT node_offset[FETCH_SIZE];
    __shared__ SizeT neighborLen[FETCH_SIZE+1];
    __shared__ Rank res[FETCH_SIZE];
    QueueT start_local = start + blockIdx.x*FETCH_SIZE;
    int size_local = min(FETCH_SIZE, size-blockIdx.x*FETCH_SIZE);

    if(threadIdx.x < size_local)
    {
	VertexId node = pr.worklists.get_item(start_local+threadIdx.x);
        res[threadIdx.x] = atomicExch(pr.res+node, 0);
        node_offset[threadIdx.x] = pr.csr_offset[node];
        neighborLen[threadIdx.x] = pr.csr_offset[node+1]-node_offset[threadIdx.x];
        atomicAdd(pr.rank+node, res[threadIdx.x]);
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
    {
        res[threadIdx.x] = res[threadIdx.x]*pr.lambda/(Rank)neighborLen[threadIdx.x];
        neighborLen[threadIdx.x] = thread_data;
    }
    if(threadIdx.x == 0)
        neighborLen[size_local] = block_aggregate;
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
            cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx];
            neighbor = pr.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
            Rank old_res = atomicAdd(pr.res+neighbor, res[cur_node_idx]);
            if(old_res <= pr.epsilon && old_res+res[cur_node_idx] >= pr.epsilon)
                ifpush = true;
        }
        pr.worklists.push_cta(ifpush, neighbor);
        abs_neighbors = abs_neighbors+32;
        neighbor = -1;
        ifpush = false;
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT>
template<int FETCH_SIZE, int BLOCK_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT>::PrStart(QueueT start, int size, int shareMem_size, cudaStream_t stream)
{
    int gridSize = (size+FETCH_SIZE-1)/FETCH_SIZE;
    //std::cout << "Fetch size: "<< FETCH_SIZE << " Grid Size: "<< gridSize << " Block Size: "<< BLOCK_SIZE << " Share Mem: "<< shareMem_size << " Start: "<< start << " Size: "<< size << std::endl;
    PageRankCTA<FETCH_SIZE, BLOCK_SIZE><<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>(start, size, *this);
}


#endif
