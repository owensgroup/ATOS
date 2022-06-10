#ifndef BreadthFirstSearch
#define BreadthFirstSearch

#include <math.h>

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../comm/csr.cuh"
#include "../comm/distr_standard_queue.cuh"

#include <cub/cub.cuh> 

template<typename VertexId>
struct BFSEntry {
	VertexId id=0xffffffff;
	VertexId depth=0xffffffff;
	__device__ __host__ BFSEntry() {id=0xffffffff; depth=0xffffffff;}
	__device__ __host__ BFSEntry(VertexId _id, VertexId _depth): id(_id), depth(_depth) {}
	__device__ __host__ BFSEntry(const volatile BFSEntry<VertexId> &entry): 
	id(entry.id), depth(entry.depth) {}
	__device__ __host__ BFSEntry(const BFSEntry<VertexId> &entry): 
	id(entry.id), depth(entry.depth) {}
	__device__ __host__ BFSEntry<VertexId>& operator=(const BFSEntry<VertexId> &entry)
	{
		id = entry.id;
		depth = entry.depth;
		return *this;
	}
};

namespace dev {
    template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE=64>
    class BFS {
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

        VertexId *depth;

        Atos::MAXCOUNT::dev::DistributedQueue<BFSEntry<VertexId>, VertexId, QueueT, PADDING_SIZE> wl;

        BFS(int m_pe, int n_pe, VertexId n, SizeT e, VertexId total, VertexId s,
             VertexId end, VertexId *table, SizeT *offset, VertexId* indices, VertexId *d, 
			Atos::MAXCOUNT::DistributedQueue<BFSEntry<VertexId>, VertexId, QueueT, PADDING_SIZE> &worklists):
        my_pe(m_pe), n_pes(n_pe), nodes(n), edges(e), totalNodes(total),
        startNode(s), endNode(end), partition_table(table), csr_offset(offset),
        csr_indices(indices), depth(d) 
        {
            wl = worklists.deviceObject(); 
        }

        __forceinline__ __device__ bool ifLocal(VertexId vid) const 
        { return (vid >= startNode && vid < endNode); }

        __forceinline__ __device__ int findPE(VertexId vid) const
        {
            for(int i=0; i<n_pes; i++)
                if(vid >= __ldg(partition_table+i) && vid < __ldg(partition_table+i+1))
                {
                    return i;
                }
            return -1;
        }
    };
}

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE=64>
class BFS
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

    VertexId *depth;

    Atos::MAXCOUNT::DistributedQueue<BFSEntry<VertexId>, VertexId, QueueT, PADDING_SIZE> worklists;

    typedef dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE> BFSDeviceObjectType;
    typedef Atos::MAXCOUNT::dev::DistributedQueue<BFSEntry<VertexId>, VertexId, QueueT, PADDING_SIZE> WLDeviceObjectType;

    BFS(Csr<VertexId, SizeT> &csr, int m_pe, int n_pe, int group_id, int group_size, int local_id, int local_size, VertexId *table, 
        uint32_t num_q=1, QueueT l_cap=0, QueueT r_cap=0, int min_iter=-1):
    my_pe(m_pe), n_pes(n_pe)
    {
        CUDA_CHECK(cudaMalloc(&partition_table, sizeof(int)*(n_pes+1)));
        CUDA_CHECK(cudaMemcpy(partition_table, table, sizeof(int)*(n_pes+1), cudaMemcpyHostToDevice))
        totalNodes = table[n_pes];
        startNode = table[my_pe];
        endNode = table[my_pe+1];
        nodes = csr.nodes;
        edges = csr.edges;
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;
        assert(endNode-startNode == nodes);
        if(l_cap == 0 || r_cap == 0)
            worklists.init(n_pes, my_pe, group_id, group_size, local_id, local_size, 2*nodes, (1<<22), num_q, min_iter);
        else worklists.init(n_pes, my_pe, group_id, group_size, local_id, local_size, l_cap, r_cap, num_q, min_iter);

        CUDA_CHECK(cudaMallocManaged(&depth, sizeof(VertexId)*totalNodes));
        //depth = (VertexId *)nvshmem_malloc(sizeof(VertexId)*totalNodes);
	}

    ~BFS() { 
        #ifdef DEBUG
            std::cout << "BFS destructor is called\n";
        #endif
        Free(); 
    }

    __host__ void print() const {
        printf("PE \t totalNode \t startNode \t endNode \t nodes \t      wl capacity \t rev capacity \t min iter of wl\n");
        printf("%2d \t %9d \t %9d \t %7d \t %7d \t %7d \t %7d \t %7d\n", my_pe, totalNodes, startNode,
        endNode, nodes, worklists.local_capacity, worklists.recv_capacity, worklists.min_iter);
    }

    __host__ void reset(cudaStream_t stream=0) const {
        worklists.reset(stream);
    }

private:
    void Free()
    {
        CUDA_CHECK(cudaFree(depth));
        CUDA_CHECK(cudaFree(partition_table));
    }

public:
    BFSDeviceObjectType DeviceObject() {
        return dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE>(my_pe, n_pes, nodes, edges, 
            totalNodes, startNode, endNode, partition_table, csr_offset, 
            csr_indices, depth, worklists);
    }

    void BFSInit(VertexId source, int numBlock);

    template<int FETCH_SIZE, int BLOCK_SIZE>
    void BFSStart(QueueT start, int size, 
         bool ifRecv, int shareMem_size, cudaStream_t stream, int wl_id);

    template<int FETCH_SIZE, int BLOCK_SIZE>
	void BFSStart_persistent(int numBlock, int numThread, int shared_mem, cudaStream_t stream);
};

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
__global__ void pushSource(dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE> bfs, VertexId source)
{
    if( LANE_ ==0)
    {
        if( bfs.ifLocal(source))
            //bfs.wl.push_warp(source-bfs.startNode);
            bfs.wl.local_push_warp(source);
        bfs.depth[source] = 0;
    }
}

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
__global__ void initDepth(dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE> bfs)
{
    for(int i=TID; i<bfs.totalNodes; i=i+gridDim.x*blockDim.x)
        bfs.depth[i] = (bfs.totalNodes+1);
}

template<typename Queue>
__global__ void updateEnd(Queue bq)
{
	if(threadIdx.x < bq.total_num_queues) {
    uint32_t max = *(bq.end_max+threadIdx.x);
    if(max!=*(bq.end+threadIdx.x) && max == *(bq.end_count+threadIdx.x))
        atomicMax((uint32_t *)bq.end+threadIdx.x, max);
	}
}

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
void BFS<VertexId, SizeT, QueueT, PADDING_SIZE>::BFSInit(VertexId source, int numBlock)
{
    int gridSize=320; int blockSize=512;
    initDepth<<<gridSize, blockSize>>>(this->DeviceObject());
    CUDA_CHECK(cudaDeviceSynchronize());
    pushSource<<<1, 32>>>(this->DeviceObject(), source);
    CUDA_CHECK(cudaDeviceSynchronize());
    updateEnd<<<1, 32>>>(worklists.deviceObject());
	worklists.setExecute(numBlock, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
__launch_bounds__(1024, 1)
__global__ void BFSCTA_kernel(QueueT start, int size, dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE> bfs, bool ifRecv, int wl_id)
{
    __shared__ SizeT node_offset[FETCH_SIZE];
    __shared__ SizeT neighborLen[FETCH_SIZE+1];
    __shared__ VertexId depths[FETCH_SIZE];
    QueueT start_local = start + blockIdx.x*FETCH_SIZE;
    int size_local = min(FETCH_SIZE, size-blockIdx.x*FETCH_SIZE);
    if(threadIdx.x <  size_local)
    {
		VertexId node;
		VertexId depth = 0xfffffff;
		bool ifprocess = true;
        // process remote receive queue
        if(ifRecv) {
        	BFSEntry<VertexId> entry;
			entry = bfs.wl.get_recv_item(start_local+threadIdx.x, wl_id);
        	node = entry.id;
			depth = atomicMin(bfs.depth+node, entry.depth);	
			ifprocess = (depth > entry.depth);
			depth = entry.depth;
        }
        // process local current active queue
        else {
            node = bfs.wl.get_local_item(start_local+threadIdx.x);
			depth = ((volatile VertexId *)bfs.depth)[node];
        }
		assert(node!=-1);
		assert(bfs.ifLocal(node));
		assert(node>=0);
		assert(node<bfs.totalNodes);
		node = node-bfs.startNode;
        //if(!bfs.ifLocal(node+bfs.startNode) || node+bfs.startNode < 0 || node+bfs.startNode >= bfs.totalNodes)
        //    asm("trap;");
            //printf("pe %d, process node, not valid %d\n", runtime.my_pe, node);

		if(ifprocess) {
			assert(depth!=0xfffffff);
			assert(depth!=bfs.totalNodes+1);
		}
        depths[threadIdx.x] = depth; 
        node_offset[threadIdx.x] = bfs.csr_offset[node];
        neighborLen[threadIdx.x] = bfs.csr_offset[node+1]-node_offset[threadIdx.x];
		if(!ifprocess)
			neighborLen[threadIdx.x] = 0;
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
        bool iflocal = true;
        VertexId neighbor = -1;
        VertexId neighbor_depth = -1;

        if(abs_neighbors+LANE_ < block_aggregate)
        {
            while(abs_neighbors+LANE_ >= neighborLen[cur_node_idx+1])
                cur_node_idx++;
            neighbor_depth = depths[cur_node_idx]+1;
            cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx]; // what if next node's neighborlen is 0
            neighbor = bfs.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
            iflocal = bfs.ifLocal(neighbor);
            VertexId old_depth = atomicMin(bfs.depth+neighbor, neighbor_depth);
            if(old_depth > neighbor_depth)
				ifpush = true;
        }
        //__syncthreads();
        bfs.wl.local_push_cta((ifpush&iflocal), neighbor);
		bfs.wl.remote_push_cta(ifpush&(!iflocal), bfs.findPE(neighbor), BFSEntry<VertexId>(neighbor, neighbor_depth));

        //__syncthreads();
        abs_neighbors = abs_neighbors+32;
    }
	__syncthreads();
}

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
template<int FETCH_SIZE, int BLOCK_SIZE>
void BFS<VertexId, SizeT, QueueT, PADDING_SIZE>::BFSStart(QueueT start, int size, bool ifRecv, int shareMem_size, cudaStream_t stream, int wl_id)
{
    if(size == 0) return;
    int gridSize = (size+FETCH_SIZE-1)/FETCH_SIZE;
    //std::cout << "PE " << my_pe << " Fetch size: "<< FETCH_SIZE << " Grid Size: "<< gridSize << " Block Size: "<< BLOCK_SIZE << " Share Mem: "<< shareMem_size << " Rev? " << ifRecv << " wl_id " << wl_id << " Start: "<< start << " Size: "<< size << std::endl;
    BFSCTA_kernel<FETCH_SIZE, BLOCK_SIZE><<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>
    (start, size, this->DeviceObject(), ifRecv, wl_id);
}

template<int FETCH_SIZE, int BLOCK_SIZE, 
typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
class BFSCTA{
    public:
        __forceinline__ __device__ void operator()(Atos::MAXCOUNT::dev::Work<VertexId, BFSEntry<VertexId>> &node, int size, QueueT idx, bool ifRecv, 
            dev::BFS<VertexId, SizeT, QueueT, PADDING_SIZE> bfs)
        {
			if(size == 0) return;			
			__shared__ SizeT node_offset[FETCH_SIZE];
			__shared__ SizeT neighborLen[FETCH_SIZE+1];
			__shared__ VertexId depths[FETCH_SIZE];

			if(threadIdx.x < size) {
				VertexId node_id = 0xffffffff;
				VertexId depth = 0xfffffff;
				bool ifprocess = true;
        		// process remote receive queue
        		if(ifRecv) {
        			node_id = node.remoteWork.id;
					depth = atomicMin(bfs.depth+node_id, node.remoteWork.depth);	
					ifprocess = (depth > node.remoteWork.depth);
					depth = node.remoteWork.depth;
        		}
        		// process local current active queue
        		else {
        		    node_id = node.localWork; 
					depth = ((volatile VertexId *)bfs.depth)[node_id];
        		}
				assert(node_id!=0xffffffff);
				assert(bfs.ifLocal(node_id));
				assert(node_id>=0);
				assert(node_id<bfs.totalNodes);
				node_id = node_id-bfs.startNode;

				if(ifprocess) {
					assert(depth!=0xfffffff);
					assert(depth!=bfs.totalNodes+1);
				}

        		depths[threadIdx.x] = depth; 
        		node_offset[threadIdx.x] = bfs.csr_offset[node_id];
        		neighborLen[threadIdx.x] = bfs.csr_offset[node_id+1]-node_offset[threadIdx.x];
				if(!ifprocess)
					neighborLen[threadIdx.x] = 0;
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
    		for(int iter=0; iter<slots_warp; iter++)
    		{
    		    bool ifpush = false;
    		    bool iflocal = true;
    		    VertexId neighbor = -1;
    		    VertexId neighbor_depth = -1;

    		    if(abs_neighbors+LANE_ < block_aggregate)
    		    {
    		        while(abs_neighbors+LANE_ >= neighborLen[cur_node_idx+1])
    		            cur_node_idx++;
    		        neighbor_depth = depths[cur_node_idx]+1;
    		        cur_node_offset = abs_neighbors+LANE_-neighborLen[cur_node_idx]; // what if next node's neighborlen is 0
    		        neighbor = bfs.csr_indices[node_offset[cur_node_idx]+cur_node_offset];
    		        iflocal = bfs.ifLocal(neighbor);
    		        VertexId old_depth = atomicMin(bfs.depth+neighbor, neighbor_depth);
    		        if(old_depth > neighbor_depth)
						ifpush = true;
    		    }
    		    //__syncthreads();
    		    bfs.wl.local_push_cta((ifpush&iflocal), neighbor);
				bfs.wl.remote_push_cta(ifpush&(!iflocal), bfs.findPE(neighbor), BFSEntry<VertexId>(neighbor, neighbor_depth));

    		    //__syncthreads();
    		    abs_neighbors = abs_neighbors+32;
    		}
			__syncthreads();
		}
};

template<typename Functor, typename VertexId, typename QueueT, typename XT>
__forceinline__ __device__ void func(Functor F, VertexId node, int size, QueueT idx, bool ifR, XT x)
{
    F(node, size, idx, ifR, x);
}

template<typename VertexId, typename SizeT, typename QueueT, int PADDING_SIZE>
template<int FETCH_SIZE, int BLOCK_SIZE>
void BFS<VertexId, SizeT, QueueT, PADDING_SIZE>::BFSStart_persistent(int numBlock, int numThread, int shared_mem, cudaStream_t stream)
{
	if(worklists.min_iter == -1)
		worklists.template launchCTA<FETCH_SIZE>(numBlock, 320, stream, shared_mem,
        BFSCTA<FETCH_SIZE, 320, VertexId, SizeT, QueueT, PADDING_SIZE>(), this->DeviceObject());
	else
    	worklists.template launchCTA_minIter<FETCH_SIZE>(numBlock, numThread, stream, shared_mem,
        BFSCTA<FETCH_SIZE, BLOCK_SIZE, VertexId, SizeT, QueueT, PADDING_SIZE>(), this->DeviceObject());
}
#endif
