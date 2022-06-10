#ifndef BreadthFirstSearch
#define BreadthFirstSearch

#include <math.h>

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../comm/csr.cuh"
#include "../comm/priority_queue.cuh"

#include <cub/cub.cuh> 

namespace dev {
    template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT=uint32_t>
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
        VertexId * const * __restrict__ depth_ptr_table;

        MaxCountQueue::dev::PriorityQueues<VertexId, THRESHOLD_TYPE, QueueT> wl;

        BFS(int m_pe, int n_pe, VertexId n, SizeT e, VertexId total, VertexId s,
             VertexId end, VertexId *table, SizeT *offset, VertexId* indices, 
             VertexId *d, MaxCountQueue::PriorityQueues<VertexId, THRESHOLD_TYPE, QueueT> &worklists,
             VertexId ** depth_table):
        my_pe(m_pe), n_pes(n_pe), nodes(n), edges(e), totalNodes(total),
        startNode(s), endNode(end), partition_table(table), csr_offset(offset),
        csr_indices(indices), depth(d), depth_ptr_table(depth_table)
        {
            wl = worklists.DeviceObject(); 
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

        __forceinline__ __device__ VertexId * get_depth_ptr(int pe, uint64_t offset) const 
        {
            return depth_ptr_table[pe]+offset;
        }
    };
}

template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT=uint32_t>
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
    VertexId **depth_ptr_table;

    MaxCountQueue::PriorityQueues<VertexId, THRESHOLD_TYPE, QueueT> worklists;

    typedef dev::BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT> BFSDeviceObjectType;
    typedef MaxCountQueue::dev::PriorityQueues<VertexId, THRESHOLD_TYPE, QueueT> WLDeviceObjectType;

    BFS(Csr<VertexId, SizeT> &csr, int m_pe, int n_pe, VertexId *table, THRESHOLD_TYPE thrhd, THRESHOLD_TYPE delta, int iter, int s_iter, QueueT cap=0):
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
        uint32_t rt_capacity = 0;
        for(int i=1; i<=n_pes; i++)
            rt_capacity = max(rt_capacity, table[i]-table[i-1]);
        rt_capacity = rt_capacity*2;
        if(cap == 0)
            worklists.init(my_pe, n_pes, thrhd, delta, QueueT(nodes*4), rt_capacity, iter, s_iter);
        else worklists.init(my_pe, n_pes, thrhd, delta, cap, rt_capacity, iter, s_iter);

        #ifdef PROFILE
        CUDA_CHECK(cudaMalloc(&(worklists.execute), sizeof(uint32_t)*80));
        CUDA_CHECK(cudaMalloc(&(worklists.reserve), sizeof(uint32_t)*80));
        CUDA_CHECK(cudaMemset(worklists.execute, 0, sizeof(uint32_t)*80));
        CUDA_CHECK(cudaMemset(worklists.reserve, 0, sizeof(uint32_t)*80));
        #endif

        //CUDA_CHECK(cudaMallocManaged(&depth, sizeof(VertexId)*totalNodes));
        depth = (VertexId *)nvshmem_malloc(sizeof(VertexId)*totalNodes);
        CUDA_CHECK(cudaMalloc(&depth_ptr_table, sizeof(VertexId *)*n_pes));
        VertexId * depth_table[n_pes];
        for(int i=0; i<n_pes; i++)
            depth_table[i] = (VertexId *)nvshmem_ptr(depth, i);
        CUDA_CHECK(cudaMemcpy(depth_ptr_table, depth_table, sizeof(VertexId *)*n_pes,
        cudaMemcpyHostToDevice));
    }

    ~BFS() { 
        #ifdef DEBUG
            std::cout << "BFS destructor is called\n";
        #endif
        Free(); 
    }

    __host__ void print() const {
        printf("PE \t totalNode \t startNode \t endNode \t nodes \t      wl capacity \t num wl \t min iter of wl\n");
        printf("%2d \t %9d \t %9d \t %7d \t %7d \t %7d\n", my_pe, totalNodes, startNode,
        endNode, nodes, worklists.get_capacity());
        #ifdef DEBUG
            VertexId *check_table[n_pes];
            CUDA_CHECK(cudaMemcpy(check_table, depth_ptr_table, sizeof(VertexId *)*n_pes,
            cudaMemcpyDeviceToHost));
            printf("depth table: \n");
            for(int i=0; i<n_pes; i++)
                printf("%16d \t", i);
            printf("\n");
            for(int i=0; i<n_pes; i++)
                printf("%p \t", check_table[i]);
            printf("\n")
        #endif
    }

    __host__ void reset(cudaStream_t stream=0) const {
        worklists.reset(stream);
    }

private:
    void Free()
    {
        //CUDA_CHECK(cudaFree(depth));
        CUDA_CHECK(cudaFree(partition_table));
        CUDA_CHECK(cudaFree(depth_ptr_table));
    }

public:
    BFSDeviceObjectType DeviceObject() {
        return dev::BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT>(my_pe, n_pes, nodes, edges, 
            totalNodes, startNode, endNode, partition_table, csr_offset, 
            csr_indices, depth, worklists, depth_ptr_table);
    }

    void BFSInit(VertexId source, int numBlock);

    template<int FETCH_SIZE, int BLOCK_SIZE>
    void BFSStart(QueueT start, int size,  QueueT threshold, MaxCountQueue::Priority prio,
         int ifRecv,
         int shareMem_size, cudaStream_t stream, int * record);
};

template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT>
__global__ void pushSource(dev::BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT> bfs, VertexId source)
{
    if( LANE_ ==0)
    {
        if( bfs.ifLocal(source))
            bfs.wl.push_warp(MaxCountQueue::Priority::HIGH, source-bfs.startNode);
            //bfs.wl.push_warp(MaxCountQueue::Priority::HIGH, source);
        bfs.depth[source] = 0;
    }
}

template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT>
__global__ void initDepth(dev::BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT> bfs)
{
    for(int i=TID; i<bfs.totalNodes; i=i+gridDim.x*blockDim.x)
        bfs.depth[i] = (bfs.totalNodes+1);
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

template<typename Queue>
__global__ void updateEnd(Queue bq)
{
    uint32_t max = *(bq.end_max+threadIdx.x);
    if(max!=*(bq.end+threadIdx.x) && max == *(bq.end_count+threadIdx.x))
        atomicMax((uint32_t *)bq.end+threadIdx.x, max);
}

template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT>
void BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT>::BFSInit(VertexId source, int numBlock)
{
    int gridSize=320; int blockSize=512;
    initDepth<<<gridSize, blockSize>>>(this->DeviceObject());
    CUDA_CHECK(cudaDeviceSynchronize());
    pushSource<<<1, 32>>>(this->DeviceObject(), source);
    CUDA_CHECK(cudaDeviceSynchronize());
    checkEnd<<<1, 32>>>(worklists.DeviceObject());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int FETCH_SIZE, int BLOCK_SIZE, typename VertexId, typename SizeT, typename THRESHOLD_TYPE,
typename QueueT>
__launch_bounds__(1024, 1)
__global__ void BFSCTA_kernel(QueueT start, int size, dev::BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT> bfs,
    int ifRecv_id, QueueT threshold, MaxCountQueue::Priority prio, int *record)
{
    __shared__ SizeT node_offset[FETCH_SIZE];
    __shared__ SizeT neighborLen[FETCH_SIZE+1];
    __shared__ VertexId depths[FETCH_SIZE];
    QueueT start_local = start + blockIdx.x*FETCH_SIZE;
    int size_local = min(FETCH_SIZE, size-blockIdx.x*FETCH_SIZE);

    bool ifpush_low = false;
    VertexId node;

    if(threadIdx.x <  size_local)
    {
        // process remote receive queue
        if(ifRecv_id) {
            node = (bfs.wl.recv_queues+(ifRecv_id-1)*bfs.wl.recv_capacity)[start_local+threadIdx.x];
            int loop=0;
            while(node == -1) {
                loop++;
                node = ((volatile VertexId *)(bfs.wl.recv_queues+(ifRecv_id-1)*bfs.wl.recv_capacity))[start_local+threadIdx.x];
                if(loop > 32) {
                    //asm("trap;");
                    printf("pe %d, node %d, ifRecv_id %d, recv offset %d, end %d, loop<32 fail\n", bfs.my_pe, node, ifRecv_id, (ifRecv_id-1)*bfs.wl.recv_capacity, start+size);
                    assert(loop<32);
                }
            }
            node = node-bfs.startNode;
        }
        // process local current active queue
        else {
            node = bfs.wl.get_item_volatile(start_local+threadIdx.x, prio);
            if(node == -1) 
                //asm("trap;");
                assert(node!=-1);
        }
        if(!bfs.ifLocal(node+bfs.startNode) || node+bfs.startNode < 0 || node+bfs.startNode >= bfs.totalNodes)
        {    //asm("trap;");
            printf("pe %d, process node, not valid %d\n", bfs.my_pe, node);
            assert(false);
        }

        VertexId depth = ((volatile VertexId *)bfs.depth)[node+bfs.startNode]; 
        if(depth < threshold) {
            depths[threadIdx.x] = depth;
            node_offset[threadIdx.x] = bfs.csr_offset[node];
            neighborLen[threadIdx.x] = bfs.csr_offset[node+1]-node_offset[threadIdx.x];
        }
        else {
            neighborLen[threadIdx.x] = 0;
            ifpush_low = true;
            #ifdef PROFILE
            atomicAdd(record, 1);
            #endif
        }
    }
    bfs.wl.push_cta(static_cast<MaxCountQueue::Priority>(!prio), ifpush_low, node);

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
    __shared__ uint32_t comm_total_cta[8];
    //__shared__ VertexId comm_temp_storage[BLOCK_SIZE];
    for(int iter=0; iter<slots_warp; iter++)
    {
        bool ifpush = false;
        bool iflocal = true;
        int pe = -1;
        VertexId neighbor = -1;
        VertexId neighbor_depth = -1;
        MaxCountQueue::Priority prio_wl = prio;

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
            {
                prio_wl = (neighbor_depth < threshold)? prio_wl : static_cast<MaxCountQueue::Priority>(!prio);
                if(iflocal)
                    ifpush = true;
                else {
                    pe = bfs.findPE(neighbor);
                    if(pe == -1 ||  pe == bfs.my_pe )
                        asm("trap;");
                    old_depth = atomicMin(bfs.get_depth_ptr(pe, neighbor), neighbor_depth);
                    if(old_depth > neighbor_depth)
                        ifpush = true;
                }
            }
            //old_depth = atomicMin(bfs.get_depth_ptr(pe, neighbor), neighbor_depth);
            //printf("neighbor %d, local %d, remote %d, ifpush %d, prio %d, pe %d, depth %d, old_depth %d, threshold %d\n", neighbor, int(iflocal),
            //int(!iflocal), int(ifpush), int(prio_wl), pe, neighbor_depth, old_depth, threshold);
        }
        //__syncthreads();
        bfs.wl.push_cta(prio_wl, (ifpush&iflocal), neighbor-bfs.startNode);

        //--------------------insert to remote GPU---------------------//
        __shared__ uint32_t res[8];
        if(threadIdx.x < 8) {
            res[threadIdx.x] = 0xffffffff; comm_total_cta[threadIdx.x] = 0;
        }
        __syncthreads();
        uint32_t alloc;
        uint32_t rank;
        unsigned mask_warp = __ballot_sync(0xffffffff, ifpush&(!iflocal));
        if(ifpush&(!iflocal)) {
            pe = (pe <  bfs.my_pe)?pe:pe-1;
            unsigned mask = __match_any_sync(mask_warp, pe);
            uint32_t total = __popc(mask);
            rank = __popc(mask & lanemask_lt());
            if(!rank) {
                alloc = atomicAdd((uint32_t *)(comm_total_cta+pe), total);
                #ifdef PROFILE
                atomicAdd(bfs.wl.execute+get_smid(), total);
                atomicAdd(((uint32_t *)bfs.wl.reserve)+get_smid(), 1);
                #endif
            }
            alloc = __shfl_sync(mask, alloc, __ffs(mask)-1);
        }
        __syncthreads();
        if(threadIdx.x < bfs.n_pes-1  && comm_total_cta[threadIdx.x])
        {
            res[threadIdx.x] =  atomicAdd(
                (uint32_t *)(bfs.wl.send_remote_alloc_end+threadIdx.x), 
                comm_total_cta[threadIdx.x]);
                if(res[threadIdx.x]+comm_total_cta[threadIdx.x] > bfs.wl.recv_capacity) 
                    printf("overflow the rtq queue pe %d, tid %d, res %d, total_cta %d\n", bfs.wl.my_pe, threadIdx.x, res[threadIdx.x], comm_total_cta[threadIdx.x]);
            //    assert(res[threadIdx.x]+total_cta[threadIdx.x] < bfs.wl.recv_capacity);
        }                
        __syncthreads();

        if(ifpush&(!iflocal)) {
            uint32_t write_offset = res[pe]+alloc+rank;
            pe = (pe < bfs.my_pe)?pe:pe+1;
            int remote_pos = (bfs.my_pe < pe)?bfs.my_pe:bfs.my_pe-1;
            MaxCountQueue::nvshmem_p_wraper(bfs.wl.recv_queues+(remote_pos)*bfs.wl.recv_capacity+write_offset,
            neighbor, pe);
        }
        //-----------------------------------------------------------------------//

        //__syncthreads();
        abs_neighbors = abs_neighbors+32;
    }

    __syncthreads();
    if(threadIdx.x < bfs.n_pes-1 && comm_total_cta[threadIdx.x])
    {
        nvshmem_fence();
        int remote_pe = (threadIdx.x < bfs.my_pe)?threadIdx.x:threadIdx.x+1;
        uint32_t update_value = *((volatile uint32_t *)(bfs.wl.send_remote_alloc_end+threadIdx.x));
        uint32_t *signal_remote_rq_end = (uint32_t *)nvshmem_ptr(bfs.wl.sender_write_remote_end, remote_pe);
        //if(remote_pe == 2) printf("pe %d update GPU2's recev end with offset %d, local offset %d, value %d\n", bfs.my_pe, ((bfs.my_pe < remote_pe)?bfs.my_pe:(bfs.my_pe-1)),threadIdx.x, update_value);
        atomicMax(signal_remote_rq_end+(bfs.my_pe < remote_pe?bfs.my_pe:bfs.my_pe-1), update_value);
    }
}

template<typename VertexId, typename SizeT, typename THRESHOLD_TYPE, typename QueueT>
template<int FETCH_SIZE, int BLOCK_SIZE>
void BFS<VertexId, SizeT, THRESHOLD_TYPE, QueueT>::BFSStart(QueueT start, int size, QueueT threshold, MaxCountQueue::Priority prio, int ifRecv,
    int shareMem_size, cudaStream_t stream, int *record)
{
    if(size == 0) return;
    int gridSize = (size+FETCH_SIZE-1)/FETCH_SIZE;
//    std::cout << "Fetch size: "<< FETCH_SIZE << " Grid Size: "<< gridSize << " Block Size: "<< BLOCK_SIZE << " Share Mem: "<< shareMem_size << " Start: "<< start << " Size: "<< size << std::endl;
    BFSCTA_kernel<FETCH_SIZE, BLOCK_SIZE><<<gridSize, BLOCK_SIZE, shareMem_size, stream>>>
    (start, size, this->DeviceObject(), ifRecv, threshold, prio, record);
}
#endif
