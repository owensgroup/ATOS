#ifndef PageRankApp
#define PageRankApp

#include <math.h>

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../comm/csr.cuh"
#include "../comm/distr_standard_queue.cuh"

#include <cub/cub.cuh>

template<typename VertexId, typename Rank>
struct PREntry {
    VertexId id=0xffffffff;
    Rank res=0xffffffff;
    __device__ __host__ PREntry() {id=0xffffffff; res=0xffffffff;}
    __device__ __host__ PREntry(VertexId _id, Rank _res): id(_id), res(_res) {}
    __device__ __host__ PREntry(const volatile PREntry<VertexId, Rank> &entry):
    id(entry.id), res(entry.res) {}
    __device__ __host__ PREntry(const PREntry<VertexId, Rank> &entry):
    id(entry.id), res(entry.res) {}
    __device__ __host__ PREntry<VertexId, Rank>& operator=(const PREntry<VertexId, Rank> &entry)
    {
        id = entry.id;
        res = entry.res;
        return *this;
    }
};

namespace dev{
	template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE=64>
    class PageRank {
    public:
        VertexId nodes;
        SizeT edges;
        VertexId totalNodes;
        VertexId startNode;
        VertexId endNode;
        const VertexId * __restrict__ partition_table;

        int my_pe;
        int n_pes;

        const SizeT * __restrict__ csr_offset;
        const VertexId * __restrict__ csr_indices;

        Rank * rank;
        Rank * res;

        Rank epsilon;
        Rank lambda;

        uint32_t* ifact;
        VertexId size_ifact;
        VertexId real_size_ifact;
        uint32_t *checkres;

        void *reserve=NULL;

		Atos::MAXCOUNT::dev::DistributedQueue<PREntry<VertexId, Rank>, VertexId, QueueT, PADDING_SIZE> wl;

        PageRank(int m_pe, int n_pe, VertexId n, SizeT e, VertexId total, VertexId s,
            VertexId end, VertexId *table, SizeT *offset, VertexId* indices, Rank epsi, Rank lamb,
            Rank * _rank, Rank * _res, Atos::MAXCOUNT::DistributedQueue
            <PREntry<VertexId, Rank>, VertexId, QueueT, PADDING_SIZE> &worklists,
            uint32_t *ifa=NULL, VertexId size_ifa=0, VertexId real_size_ifa=0, 
            uint32_t *cres=NULL, void * res=NULL): my_pe(m_pe), n_pes(n_pe), nodes(n), edges(e), totalNodes(total), 
            startNode(s), endNode(end), partition_table(table), csr_offset(offset), csr_indices(indices), 
            epsilon(epsi), lambda(lamb), rank(_rank), res(_res), 
            reserve(res), ifact(ifa), size_ifact(size_ifa), real_size_ifact(real_size_ifa),
            checkres(cres) { 
                wl = worklists.deviceObject();
            }

        __forceinline__ __device__ bool ifLocal(VertexId vid) const
        { return (vid >= startNode && vid < endNode); }

        __forceinline__ __device__ int findPE(VertexId &vid) const
        {
            for(int i=0; i<n_pes; i++)
            {
                if(vid >= __ldg(partition_table+i) && vid < __ldg(partition_table+i+1))
                {
                    return i;
                }
            }
            return -1;
        }
    };
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE=64>
class PageRank
{
public:
    VertexId nodes;
    SizeT edges;
    VertexId totalNodes;
    VertexId startNode;
    VertexId endNode;
    VertexId *partition_table;

    int my_pe;
    int n_pes;

    SizeT *csr_offset;
    VertexId *csr_indices;

    float lambda;               //probability to be activited
    float epsilon;              //stop threshold

    Rank * rank;              //rank value for each node, length: nodes
    Rank * res;               //residual value used to update rank value, length: nodes
    Rank ** res_ptr_table;  

	Atos::MAXCOUNT::DistributedQueue<PREntry<VertexId, Rank>, VertexId, QueueT, PADDING_SIZE> worklists;

	typedef dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> PRDeviceObjectType;
    typedef Atos::MAXCOUNT::dev::DistributedQueue<PREntry<VertexId, Rank>, VertexId, QueueT, PADDING_SIZE> WLDeviceObjectType;

    uint32_t* ifact;
    VertexId size_ifact;
    VertexId real_size_ifact;
    uint32_t *checkres;
    void *reserve=NULL;

    PageRank(Csr<VertexId, SizeT> &csr, int m_pe, int n_pe, int group_id, int group_size, int local_id, int local_size, VertexId *table, Rank lamb, Rank eps, 
        uint32_t num_q=1, QueueT l_cap=0, QueueT r_cap=0, int min_iter=-1):
    my_pe(m_pe), n_pes(n_pe), lambda(lamb), epsilon(eps)
    {
        CUDA_CHECK(cudaMalloc(&partition_table, sizeof(int)*(n_pes+1)));
        CUDA_CHECK(cudaMemcpy(partition_table, table, sizeof(int)*(n_pes+1), cudaMemcpyHostToDevice));
        totalNodes = table[n_pes];
        startNode = table[my_pe];
        endNode = table[my_pe+1];
        nodes = csr.nodes;
        edges = csr.edges;
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;
        assert(endNode-startNode == nodes);
        if(l_cap == 0 || r_cap == 0)
			 worklists.init(n_pes, my_pe, group_id, group_size, local_id, local_size, 32*nodes, (1<<25), num_q, min_iter);
        else worklists.init(n_pes, my_pe, group_id, group_size, local_id, local_size, l_cap, r_cap, num_q, min_iter);
		printf("worklist.min_iter %d\n", worklists.min_iter);

        // only local rank 
        CUDA_CHECK(cudaMalloc(&rank, sizeof(Rank)*nodes));
        CUDA_CHECK(cudaMemset(rank, 0, sizeof(Rank)*nodes));
		CUDA_CHECK(cudaMalloc(&res, sizeof(Rank)*totalNodes));
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*totalNodes));

        real_size_ifact = (endNode+31)/32 - startNode/32 ;
        size_ifact = align_up_yc(real_size_ifact, 32);
        printf("pe %d, start %d, end %d, nodes %d, real_size %d, size_ifact %d\n", my_pe, startNode, endNode, nodes, real_size_ifact, size_ifact);
        CUDA_CHECK(cudaMalloc(&ifact, sizeof(uint32_t)*(size_ifact)));
        if(real_size_ifact < size_ifact)
        CUDA_CHECK(cudaMemset(ifact+real_size_ifact, 0xffffffff, sizeof(uint32_t)*(size_ifact-real_size_ifact)));
        CUDA_CHECK(cudaMalloc(&checkres, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(checkres, 0, sizeof(uint32_t)));
    }

    ~PageRank() {
        std::cout << "PR destructor is called\n";
        Free();
    }

    __host__ void print() const {
        printf("PE \t totalNode \t startNode \t endNode \t nodes \t      wl capacity \t total num wl \t recv capacity \t  min iter of wl \t lambda \t epsilon\n");
        printf("%2d \t %9d \t %9d \t %7d \t %7d \t %7d \t %7d \t %7d \t %7d \t\t %1.4f \t %1.4f\n", my_pe, totalNodes, startNode,
        endNode, nodes, worklists.local_capacity, worklists.total_num_queues, worklists.recv_capacity, worklists.min_iter, lambda, epsilon);
    }

    __host__ void reset(cudaStream_t stream=0, QueueT execute_init_value=0) const {
        worklists.reset(stream, execute_init_value);
        CUDA_CHECK(cudaMemset(res, 0, sizeof(Rank)*totalNodes));
        CUDA_CHECK(cudaMemset(rank, 0, sizeof(Rank)*nodes));
    }

    PRDeviceObjectType DeviceObject() {
        return dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>(my_pe, n_pes, 
            nodes, edges, totalNodes, startNode, endNode, partition_table, csr_offset, 
            csr_indices, epsilon, lambda, rank, res, worklists, ifact,
            size_ifact, real_size_ifact, checkres, reserve);
    }

    void PageRankInit(cudaStream_t stream);
    void PageRankInitMerge(cudaStream_t stream);

    template<int FETCH_SIZE, int BLOCK_SIZE>
    void PageRankStart(QueueT start, int size, int shareMem_size, cudaStream_t stream);

	void PageRankMerge(QueueT start, int size, cudaStream_t stream, int recv_id, int stream_id, int *check_finish);

    void PushVertices(cudaStream_t stream);

private:
    void Free()
    {
        CUDA_CHECK(cudaFree(rank));
        CUDA_CHECK(cudaFree(res));
        CUDA_CHECK(cudaFree(partition_table));
    }
};

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__global__ void initResRank(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
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

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__global__ void initWL(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
{
    for(VertexId i=TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
    {
        pr.wl.local_queues[i] = i;
    }
    if(TID == 0) {
        *(pr.wl.end_alloc) = pr.nodes;
        *(pr.wl.end_max) = pr.nodes;
        *(pr.wl.end_count) = pr.nodes;
        *(pr.wl.end) = pr.nodes;
    }

    //for(VertexId i = TID; i<pr.nodes; i+=blockDim.x*gridDim.x)
    //{
    //    if(LANE_ == 0)
    //        pr.ifact[(i>>5)] = 0xffffffff;
    //}
    for(VertexId i=TID; i<pr.size_ifact; i+=blockDim.x*gridDim.x)
        pr.ifact[i] = 0xffffffff;
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__global__ void initResRemote(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
{
    for(VertexId i=TID; i-threadIdx.x<pr.totalNodes; i+=blockDim.x*gridDim.x)
    {
		bool ifpush = false;
		PREntry<VertexId, Rank> entry;
		int pe = -1;
        if(i < pr.totalNodes && (!pr.ifLocal(i)) && pr.res[i]!=0.0)
        {
			ifpush = true;
            pe = pr.findPE(i);
            assert(pe != -1);
            assert(pe != pr.my_pe);
            entry.res = atomicExch(pr.res+i, 0.0);
			entry.id = i;
        }
		pr.wl.remote_push_cta(ifpush, pe, entry);
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__global__ void initMergeRes(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
{
    for(int recv_q = 0; recv_q < pr.n_pes-1; recv_q++) {
        QueueT recv_end = pr.wl.get_queue_end(pr.wl.num_local_queues+recv_q);
        if(TID == 0)
            printf("pe %d, recv_q %d, recv_end %d, pr.wl.num_local_queues %d\n", pr.my_pe, recv_q, recv_end, pr.wl.num_local_queues);
        for(VertexId i=TID; i<recv_end; i+=blockDim.x*gridDim.x)
        {
            PREntry<VertexId, Rank> entry = pr.wl.get_recv_item(i, recv_q);
            assert(entry.id >= 0);
            assert(entry.id < pr.totalNodes);
            assert(pr.ifLocal(entry.id));
            assert(entry.res > 0);
            atomicAdd(pr.res+entry.id, entry.res);
        }
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PageRankInit(cudaStream_t stream)
{
    initResRank<<<160, 512, 0, stream>>>(this->DeviceObject());
	CUDA_CHECK(cudaStreamSynchronize(stream));
    initResRemote<<<160, 512, 0, stream>>>(this->DeviceObject());
	CUDA_CHECK(cudaStreamSynchronize(stream));
}


template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PageRankInitMerge(cudaStream_t stream)
{
    int gridSize=320; int blockSize=512;
    initMergeRes<<<gridSize, blockSize, 0, stream>>>(this->DeviceObject());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    worklists.reset(stream, 0xffffffff);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    initWL<<<1, 32, 0, stream>>>(this->DeviceObject());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__launch_bounds__(1024, 1)
__global__ void PRCTA_kernel(QueueT start, int size, dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
{
    __shared__ SizeT node_offset[FETCH_SIZE];
    __shared__ SizeT neighborLen[FETCH_SIZE+1];
    __shared__ Rank res[FETCH_SIZE];
    QueueT start_local = start + blockIdx.x*FETCH_SIZE;
    int size_local = min(FETCH_SIZE, size-blockIdx.x*FETCH_SIZE);

    VertexId node;
    if(threadIdx.x <  size_local)
    {
        //node = pr.wl.get_local_item(start_local+threadIdx.x);
		node = pr.wl.get_local_item_volatile(start_local+threadIdx.x);
        assert(node != -1);
        assert(pr.ifLocal(node+pr.startNode));
        assert(node >=0 && node < pr.nodes);
        res[threadIdx.x] = atomicExch(pr.res+node+pr.startNode, 0.0);
        if(((node+pr.startNode-((pr.startNode>>5)<<5))>>5)>=pr.size_ifact ) {
            printf("pe %d, node %d, wl idx %d, global node %d, ifact idx %d, update mask %x\n",
            pr.my_pe, node, start_local+threadIdx.x, node+pr.startNode, ((node+pr.startNode-((pr.startNode>>5)<<5))>>5), (0xffffffff ^ (1 << ((node+pr.startNode-((pr.startNode>>5)<<5))&31))));
            assert(false);
        }
        if(((node+pr.startNode-((pr.startNode>>5)<<5))>>5) < 0) {
            printf("pe %d, node %d, wl idx %d, global node %d, ifact idx %d, update mask %x\n",
            pr.my_pe, node, start_local+threadIdx.x, node+pr.startNode, ((node+pr.startNode-((pr.startNode>>5)<<5))>>5), (0xffffffff ^ (1 << ((node+pr.startNode-((pr.startNode>>5)<<5))&31))));
        }
        atomicAnd(pr.ifact+((node+pr.startNode-((pr.startNode>>5)<<5))>>5), (0xffffffff ^ (1 << ((node+pr.startNode-((pr.startNode>>5)<<5))&31))));
        atomicAdd(pr.rank+node, res[threadIdx.x]);
        node_offset[threadIdx.x] = pr.csr_offset[node];
        neighborLen[threadIdx.x] = pr.csr_offset[node+1]-node_offset[threadIdx.x];
        //if(node == 0)
        //    printf("pe %d, node %d, wl idx %d, global node %d, ifact idx %d, update mask %x\n",
        //    pr.my_pe, node, start_local+threadIdx.x, node+pr.startNode, ((node+pr.startNode-((pr.startNode>>5)<<5))>>5), (0xffffffff ^ (1 << ((node+pr.startNode-((pr.startNode>>5)<<5))&31))));
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
    for(int iter=0; iter<slots_warp; iter++)
    {
		bool ifpush = false;
		PREntry<VertexId, Rank> entry;
		int pe = -1;
        if(abs_neighbors+LANE_ < block_aggregate)
        {
            while(abs_neighbors+LANE_ >= neighborLen[cur_node_idx+1])
                cur_node_idx++;
            cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx];
            VertexId neighbor = pr.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
            if(pr.ifLocal(neighbor))
                atomicAdd(pr.res+neighbor, res[cur_node_idx]);
            else {
                Rank old_res = atomicAdd(pr.res+neighbor, res[cur_node_idx]);
                if(old_res <= pr.epsilon && old_res+res[cur_node_idx] >= pr.epsilon) {
                    pe = pr.findPE(neighbor);
					entry.id = neighbor;
					entry.res = atomicExch(pr.res+neighbor, 0.0); 
					ifpush = true;
                }
            }
        }
		pr.wl.remote_push_cta(ifpush, pe, entry);
        abs_neighbors = abs_neighbors+32;
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
template<int FETCH_SIZE, int BLOCK_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PageRankStart(QueueT start, int size,
    int shareMem_size, cudaStream_t stream)
{
    if(size == 0) return;
    int gridSize = (size+FETCH_SIZE-1)/FETCH_SIZE;
//    std::cout << "Fetch size: "<< FETCH_SIZE << " Grid Size: "<< gridSize << " Block Size: "<< BLOCK_SIZE << " Share Mem: "<< shareMem_size << " Start: "<< start << " Size: "<< size << std::endl;
    PRCTA_kernel<FETCH_SIZE, BLOCK_SIZE><<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>
    (start, size, this->DeviceObject());
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__launch_bounds__(1024, 1)
__global__ void PRMerge_kernel(QueueT start, int size, dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr, int recv_id, int stream_id, int *check_finish)
{
	int update_size = 0;
	for(int idx = TID; idx < size; idx+=blockDim.x*gridDim.x)
	{
		PREntry<VertexId, Rank> entry = pr.wl.get_recv_item(start+idx, recv_id);
		assert(pr.ifLocal(entry.id));
		assert(entry.id != -1);
		atomicAdd(pr.res+entry.id, entry.res);
		if(threadIdx.x == 0) {
			update_size += min(blockDim.x, size-idx);
		}
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		int old = atomicAdd(check_finish+stream_id*PADDING_SIZE, update_size);
		if(old+update_size == size) {
			check_finish[stream_id*PADDING_SIZE] = 0;	
			*(pr.wl.execute+pr.wl.size_per_queue*(recv_id+1)+stream_id) = start+size;
			//printf("pe %d, update q %d, with %d\n", pr.my_pe, recv_id+1, start+size);
			if(start == 0) *(pr.wl.start+(recv_id+1)*PADDING_SIZE) = start+size;
			//printf("pe %d, recv_q %d, stream_id %d, start %d (%p), update execute %d\n", pr.my_pe, recv_id, stream_id, *(pr.wl.start+PADDING_SIZE), pr.wl.start+PADDING_SIZE, start+size);
		}
	}
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PageRankMerge(QueueT start, int size,
    cudaStream_t stream, int recv_id, int stream_id, int *check_finish)
{
    if(size == 0) return;
    int gridSize = (size+511)/512;
    PRMerge_kernel<<<gridSize, 512, 0, stream>>>
    (start, size, this->DeviceObject(), recv_id, stream_id, check_finish);
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
__global__ void PushVertices_kernel(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
{
    for(VertexId idx=TID; idx < pr.size_ifact; idx+=gridDim.x*blockDim.x)
    {
        uint32_t my_mask = 0xffffffff;
        my_mask = pr.ifact[idx];
        my_mask = ~my_mask;
        for(int j=0; j<32; j++)
        {
            uint32_t cur_mask = __shfl_sync(0xffffffff, my_mask, j);
            if(((1<<LANE_)&cur_mask)!=0)
            {
                VertexId pos = idx-LANE_+j;
                Rank res = ((Rank volatile *)(pr.res+((pr.startNode>>5)<<5)))[pos*32+LANE_];
                if(res >= pr.epsilon)
                {
                    uint32_t mask = __activemask();
                    if(__popc(mask & lanemask_lt()) == 0)
                        atomicOr(pr.ifact+pos, mask);
                    __syncwarp(mask);
                    assert(pr.ifLocal(pos*32+LANE_+((pr.startNode>>5)<<5)));
                    pr.wl.local_push_warp(pos*32+LANE_+((pr.startNode>>5)<<5)-pr.startNode);
                    //atomicAdd(record, 1);
                }
            }
        }
    }
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PushVertices(cudaStream_t stream)
{
    int gridSize = (size_ifact+511)/512;
    //printf("pe %d, launch push vertices with %d X %d\n", this->my_pe, gridSize, 512);
    PushVertices_kernel<<<gridSize, 512, 0, stream>>>(this->DeviceObject());
}
/*
template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t, int PADDING_SIZE=64>
class PageRankCTA {
public:
    __forceinline__ __device__ void operator()(VertexId node, int size, dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
    {
        if(size == 0) return;
        __shared__ SizeT node_offset[FETCH_SIZE];
        __shared__ SizeT neighborLen[FETCH_SIZE+1];
        __shared__ Rank res[FETCH_SIZE];

        if(threadIdx.x < size)
        {
            assert(pr.ifLocal(node+pr.startNode));
            res[threadIdx.x] = atomicExch(pr.res+node+pr.startNode, 0.0);
            atomicAnd(pr.ifact+((node+pr.startNode-((pr.startNode>>5)<<5))>>5), (0xffffffff ^ (1 << ((node+pr.startNode-((pr.startNode>>5)<<5))&31))));
            atomicAdd(pr.rank+node, res[threadIdx.x]);
            node_offset[threadIdx.x] = pr.csr_offset[node];
            neighborLen[threadIdx.x] = pr.csr_offset[node+1]-node_offset[threadIdx.x];
        }
        __syncthreads();

        typedef cub::BlockScan<int, BLOCK_SIZE, cub::BLOCK_SCAN_RAKING,1,1,700> BlockScan;
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
                if(pr.ifLocal(neighbor))
                    atomicAdd(pr.res+neighbor, res[cur_node_idx]);
                else {
                    Rank old_res = atomicAdd(pr.res+neighbor, res[cur_node_idx]);
                    if(old_res <= pr.epsilon && old_res+res[cur_node_idx] >= pr.epsilon) {
                        int pe = pr.findPE(neighbor);
                        atomicAdd(pr.get_res_ptr(pe, neighbor), atomicExch(pr.res+neighbor, 0.0));
                    }
                }
            }
            abs_neighbors = abs_neighbors+32;
        }
    }
};

template<int ROUND_SIZE, typename VertexId, typename SizeT, typename Rank, typename QueueT=uint32_t, int PADDING_SIZE=64>
class PushVerticesCTA {
    public:
        __forceinline__ __device__ void operator()(dev::PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE> pr)
        {
            __shared__ QueueT alloc;
            if(threadIdx.x == 0) {
                alloc = atomicAdd(pr.checkres, ROUND_SIZE*blockDim.x);
                //if(pr.my_pe == 0)
                //printf("bid %d, alloc %d\n", blockIdx.x, alloc);
            }
            __syncthreads();
            for(int round = 0; round < ROUND_SIZE; round++)
            {
                uint32_t my_mask = 0xffffffff;
                my_mask = pr.ifact[(alloc+round*blockDim.x+threadIdx.x)%(pr.size_ifact)];
                my_mask = ~my_mask;
                for(int j=0; j<32; j++)
                {
                    uint32_t cur_mask = __shfl_sync(0xffffffff, my_mask, j); 
                    //if(cur_mask == 0) continue;
                    bool ifpush = false;
                    VertexId item = 0;
                    if(((1<<(LANE_))&cur_mask)!=0)
                    {
                        //VertexId pos = (((alloc+round*blockDim.x+((threadIdx.x>>5)<<5)+j)%pr.size_ifact));
                        VertexId pos = (alloc+round*blockDim.x+threadIdx.x-LANE_+j)%pr.size_ifact;
                        Rank res = *((Rank volatile *)(pr.res+((pr.startNode>>5)<<5)+pos*32+LANE_));
                        if(res >= pr.epsilon)
                        //if(((Rank volatile *)pr.res)[pos*32+LANE_] >= pr.epsilon)
                        {
                            uint32_t mask = __activemask();
                            if(__popc(mask & lanemask_lt()) == 0 )
                                atomicOr(pr.ifact+pos, mask);
                            __syncwarp(mask);
                            assert(pr.ifLocal(pos*32+LANE_+((pr.startNode>>5)<<5)));
                            //pr.wl.push_warp(pos*32+LANE_+((pr.startNode>>5)<<5)-pr.startNode);
                            ifpush = true;
                            item = pos*32+LANE_+((pr.startNode>>5)<<5)-pr.startNode;
                        }
                    }
                    //__syncwarp();
                    //__syncthreads();
                    pr.wl.push_cta(ifpush, item);
                }
                __syncwarp();
            }
        }
};

template<typename Functor, typename VertexId, typename XT>
__forceinline__ __device__ void func(Functor F, VertexId node, int size, XT x)
{
    F(node, size, x);
}

template<typename Functor, typename XT>
__forceinline__ __device__ void func(Functor F, XT x)
{
    F(x);
}

template<typename VertexId, typename SizeT, typename Rank, typename QueueT, int PADDING_SIZE>
template<int FETCH_SIZE, int BLOCK_SIZE, int ROUND_SIZE>
void PageRank<VertexId, SizeT, Rank, QueueT, PADDING_SIZE>::PageRankStart_persist(int numBlock, int numThread, int shareMem_size)
{
    worklists.template launchCTA_minIter_2Func<FETCH_SIZE>(numBlock, numThread, shareMem_size, 
        PageRankCTA<FETCH_SIZE, BLOCK_SIZE, VertexId, SizeT, Rank>(), 
        PushVerticesCTA<ROUND_SIZE, VertexId, SizeT, Rank> (), 
        this->DeviceObject());
}
*/
#endif
