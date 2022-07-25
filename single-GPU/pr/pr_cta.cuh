#ifndef PAGE_RANK
#define PAGE_RANK
#include <limits>
#include "../../util/error_util.cuh"
#include "../../util/util.cuh"
#include "../../comm/csr.cuh"
#include "../queue/queue.cuh"
//#include "../profile_queue/queue.cuh"

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
    uint32_t* ifact;
    VertexId size_ifact;
    VertexId real_size_ifact;
    uint32_t *checkres;

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
        //worklists.init(min(QueueT(1<<30), max(QueueT(1024), QueueT(nodes*8))), min_iter);
        worklists.init(QueueT(nodes)*32, min_iter);
        //worklists.init(QueueT(nodes)*18, min_iter);
        std::cout << "wl_size: "<<worklists.get_capacity()<<std::endl;

        CUDA_CHECK(cudaMallocManaged(&rank, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMallocManaged(&res, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*nodes));
        float h_inf = std::numeric_limits<float>::infinity(); 
        CUDA_CHECK(cudaMemcpyToSymbol(inf, &h_inf, sizeof(float), 0, cudaMemcpyHostToDevice));
        real_size_ifact = (nodes+31)/32;
        size_ifact = align_up_yx(real_size_ifact, 32);
        CUDA_CHECK(cudaMallocManaged(&ifact, sizeof(uint32_t)*(size_ifact)));
        CUDA_CHECK(cudaMemset(ifact+real_size_ifact, 0xffffffff, sizeof(uint32_t)*(size_ifact-real_size_ifact)));
        CUDA_CHECK(cudaMallocManaged(&checkres, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(checkres, 0, sizeof(uint32_t)));
    }

    void release() {
	    worklists.release();
        CUDA_CHECK(cudaFree(rank));
        CUDA_CHECK(cudaFree(res));
        CUDA_CHECK(cudaFree(ifact));
    }

    void reset() {
        worklists.reset();
        CUDA_CHECK(cudaMemset(rank, 0, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(checkres, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(ifact+real_size_ifact, 0xffffffff, sizeof(uint32_t)*(size_ifact-real_size_ifact)));
    }

    void PrInit(uint32_t numBlock, uint32_t numThread);

    template<int FETCH_SIZE, int ROUND_SIZE>
    void PrStart_CTA(int numBlock, int numThread, int shareMem_size, cudaStream_t stream1);

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
   for(VertexId i = TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
   {
       if(LANE_ == 0)
           pr.ifact[(i>>5)] = 0xffffffff;
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

   for(VertexId i = TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
   {
       if(LANE_ == 0)
           pr.ifact[(i>>5)] = 0xffffffff;
   }
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

__device__ int stop_counter=0;

template<int FETCH_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t>
class PageRankFuncCTA {
    public:
        __forceinline__ __device__ void operator()(VertexId *nodes, int size, PageRank<VertexId, SizeT, Rank, QueueT> pr)
        {
            __shared__ SizeT node_offset[FETCH_SIZE];
            __shared__ SizeT neighborLen[FETCH_SIZE+1];
            __shared__ Rank res[FETCH_SIZE];

            if(threadIdx.x < size)
            {
                VertexId node = nodes[threadIdx.x];
                res[threadIdx.x] = atomicExch(pr.res+node, 0);
                node_offset[threadIdx.x] = pr.csr_offset[node];
                neighborLen[threadIdx.x] = pr.csr_offset[node+1]-node_offset[threadIdx.x];
                atomicAnd(pr.ifact+(node>>5), (0xffffffff ^ (1 << (node&31))));
                atomicAdd(pr.rank+node, res[threadIdx.x]);
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
            {
                res[threadIdx.x] = res[threadIdx.x]*pr.lambda/(Rank)neighborLen[threadIdx.x];
                neighborLen[threadIdx.x] = thread_data;
            }
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
            for(int iter=0; iter<slots_warp; iter++)
            {
                if(abs_neighbors+LANE_ < block_aggregate)
                {
                    while(abs_neighbors+LANE_ >= neighborLen[cur_node_idx+1])
                        cur_node_idx++;
                    cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx];
                    VertexId neighbor = pr.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
                    atomicAdd(pr.res+neighbor, res[cur_node_idx]);
                }
                abs_neighbors = abs_neighbors+32;
            }
        }
};

template<int ROUND_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t>
class PushVertices {
    public:
        __forceinline__ __device__ void operator()(PageRank<VertexId, SizeT, Rank, QueueT> pr)
        {
            __shared__ QueueT alloc;
            if(threadIdx.x == 0)
                alloc = atomicAdd(pr.checkres, ROUND_SIZE*blockDim.x);
            __syncthreads();
            for(int round = 0; round < ROUND_SIZE; round++)
            {
                uint32_t my_mask = 0xffffffff;
                my_mask = pr.ifact[(alloc+round*blockDim.x+threadIdx.x)%(pr.size_ifact)];
                my_mask = ~my_mask;
                for(int j=0; j<32; j++)
                {
                    uint32_t cur_mask = __shfl_sync(0xffffffff, my_mask, j); 
           //         if(cur_mask == 0) continue;
                    bool ifpush = false;
                    VertexId item = 0;
                    if(((1<<(LANE_))&cur_mask)!=0)
                    {
                        VertexId pos = (((alloc+round*blockDim.x+((threadIdx.x>>5)<<5)+j)%pr.size_ifact));
                        if(((Rank volatile *)pr.res)[pos*32+LANE_] >= pr.epsilon)
                        {
                            uint32_t mask = __activemask();
                            if(__popc(mask & lanemask_lt()) == 0 )
                                atomicOr(pr.ifact+pos, mask);
                            __syncwarp(mask);
                            pr.worklists.push_warp(pos*32+LANE_);
 //                           ifpush = true;
 //                           item = pos*32+LANE_;
                        }
                    }
          //          __syncwarp();
//                    __syncthreads();
//                    pr.worklists.push_cta(ifpush, item);
                }
            }
        }
};


template<typename Functor, typename T, typename P>
__forceinline__ __device__ void func(Functor F, T t, int size, P p)
{
    F(t, size, p);
}

template<typename Functor, typename P>
__forceinline__ __device__ void func(Functor F, P p)
{
    F(p);
}

template< typename VertexId, typename SizeT, typename Rank, typename QueueT>
template<int FETCH_SIZE, int ROUND_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT>::PrStart_CTA(int numBlock, int numThread, int shareMem_size, cudaStream_t stream1)
{
    worklists.template launchCTA_minIter_2func<FETCH_SIZE>(numBlock, numThread, stream1, shareMem_size, PageRankFuncCTA<FETCH_SIZE, VertexId, SizeT, Rank>(), PushVertices<ROUND_SIZE, VertexId, SizeT, Rank> (), *this);
    CUDA_CHECK(cudaStreamSynchronize(stream1));
}


#endif
