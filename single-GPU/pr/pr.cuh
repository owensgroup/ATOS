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
    MaxCountQueue::Queues<VertexId, QueueT> worklists;

    PageRank(Csr<VertexId, SizeT> &csr, float _lambda, float _epsilon, uint32_t min_iter=800, int num_queues=1)
    {
        nodes = csr.nodes;
        edges = csr.edges;
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;
        lambda = _lambda;
        epsilon = _epsilon;
        //worklists.init(min(QueueT(1<<30), max(QueueT(1024), QueueT(nodes*8))), min_iter);
        worklists.init(QueueT(nodes)*16, num_queues, min_iter);
        //worklists.init(QueueT(nodes)*18, min_iter);
        std::cout << "wl_size: "<<worklists.get_capacity()<<std::endl;

        CUDA_CHECK(cudaMallocManaged(&rank, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMallocManaged(&res, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*nodes));
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

    template<int ROUND_SIZE>
    void PrStart_warp(int numBlock, int numThread);

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
    worklists.launchWarpPer32Items_minIter_preLaunch(numBlock, numThread);
    int gridSize = 320; int blockSize = 512;
    _PrInit<VertexId, SizeT, Rank><<<gridSize, blockSize>>>(*this);

    //Push all vertices and its out edges list length into the worklist (vertexId, listlength)
    _PushVertices2<VertexId, SizeT, Rank, QueueT><<<gridSize, blockSize>>>(*this);
    CUDA_CHECK(cudaDeviceSynchronize());
    checkEnd<<<1, 1>>>(worklists);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t>
class PageRankFuncWarp {
    public:
        __forceinline__ __device__ void operator()(VertexId node, PageRank<VertexId, SizeT, Rank, QueueT> pr)
        {
            assert(node != -1);
            Rank res_owner = 0;
            if(LANE_ == 0) {
                res_owner = atomicExch(pr.res+node, 0.0);
                atomicAnd(pr.ifact+(node>>5), (0xffffffff ^ (1 << (node&31))));
            }
            SizeT node_offset = pr.csr_offset[node];
            SizeT neighborLen = pr.csr_offset[node+1];
            res_owner = __shfl_sync(0xffffffff, res_owner, 0);
            if(res_owner > 0)
            {
                if(LANE_ == 0)
                    atomicAdd(pr.rank+node, res_owner);
                neighborLen = neighborLen-node_offset;
                res_owner = res_owner*pr.lambda/(Rank)(neighborLen);  
                __syncwarp();
                for(int item=LANE_; item<neighborLen; item=item+32) {
                    VertexId neighbor = pr.csr_indices[node_offset+item];
                    atomicAdd(pr.res+neighbor, res_owner);
                }
                __syncwarp();
            }
        }
};

template<int ROUND_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t>
class PushVertices {
    public:
        __forceinline__ __device__ void operator()(PageRank<VertexId, SizeT, Rank, QueueT> pr)
        {
            QueueT alloc;
            if(LANE_ == 0)
                alloc = atomicAdd(pr.checkres, ROUND_SIZE*32);
            alloc = __shfl_sync(0xffffffff, alloc, 0);
            for(int round = 0; round < ROUND_SIZE; round++)
            {
                uint32_t my_mask = 0xffffffff;
                my_mask = pr.ifact[(alloc+round*32+LANE_)%(pr.size_ifact)];
                my_mask = ~my_mask;
                for(int j=0; j<32; j++)
                {
                    uint32_t cur_mask = __shfl_sync(0xffffffff, my_mask, j); 
                    //if(cur_mask == 0) continue;
                    if(((1<<(LANE_))&cur_mask)!=0)
                    {
                        VertexId pos = ((alloc+round*32+j)%pr.size_ifact);
                        if(((Rank volatile *)pr.res)[pos*32+LANE_] >= pr.epsilon)
                        {
                            uint32_t mask = __activemask();
                            if(__popc(mask & lanemask_lt()) == 0 )
                                atomicOr(pr.ifact+pos, mask);
                            __syncwarp(mask);
                            pr.worklists.push_warp(pos*32+LANE_);
                        }
                    }
                    //__syncwarp();
                }
            }
            __syncwarp();
        }
};


template<typename Functor, typename T, typename P>
__forceinline__ __device__ void func(Functor F, T t, P p)
{
    F(t, p);
}

template<typename Functor, typename P>
__forceinline__ __device__ void func(Functor F, P p)
{
    F(p);
}

template< typename VertexId, typename SizeT, typename Rank, typename QueueT>
template<int ROUND_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT>::PrStart_warp(int numBlock, int numThread)
{
    worklists.launchWarpPer32Items_minIter_2func(numBlock, numThread, PageRankFuncWarp<VertexId, SizeT, Rank>(), PushVertices<ROUND_SIZE, VertexId, SizeT, Rank> (), *this);
    worklists.sync_all_wl();
}


#endif
