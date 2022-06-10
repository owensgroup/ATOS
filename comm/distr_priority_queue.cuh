#ifndef DISTRIBUTED_PRIORITY_QUEUE
#define DISTRIBUTED_PRIORITY_QUEUE

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../util/nvshmem_util.cuh"

#include "distr_queue_base.cuh"

#include <assert.h>

namespace Atos {
	namespace MAXCOUNT {
		enum Priority {HIGH=0, LOW=1};
		namespace dev {
			// SHARE_BUFFER_SIZE > blockDim.x+ n*BATCH_SIZE, n>=1
        	template<typename RECV_T, typename LOCAL_T, typename THRESHOLD_TYPE, typename COUNTER_T, int PADDING_SIZE=1>
			class DistributedPriorityQueue : public Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> {
			public:
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::n_pes;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::my_pe;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::group_size;

				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_alloc;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_max;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_count;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::start_alloc;

				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::local_capacity;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::recv_capacity;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::local_queues;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::recv_queues;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::aggregate_queues;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::aggregate_maps;

				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::num_local_queues;
				using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::num_aggregate_queues;
				THRESHOLD_TYPE threshold;
    	    	THRESHOLD_TYPE threshold_delta;

				COUNTER_T *check_temp = NULL;

            	DistributedPriorityQueue() {}
            	DistributedPriorityQueue(const Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> &base,THRESHOLD_TYPE thrshd, 
				THRESHOLD_TYPE delta, COUNTER_T * check_t = NULL):
				Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>(base),
				threshold(thrshd), threshold_delta(delta), check_temp(check_t)
				{ }
	
				__forceinline__ __device__ bool same_group(int pe) const {return pe/group_size==my_pe/group_size;}
				__forceinline__ __device__ LOCAL_T get_local_item(COUNTER_T idx, Priority prio) const
            	{ return *(local_queues + uint64_t(prio)*local_capacity + idx);}

            	__forceinline__ __device__ LOCAL_T get_local_item_volatile(COUNTER_T idx, Priority prio) const
            	{ return *((volatile LOCAL_T *)(local_queues + uint64_t(prio)*local_capacity + idx));}

				__forceinline__ __device__ RECV_T get_recv_item(COUNTER_T index, int q_id) {
                	if(q_id >= n_pes-1)  asm("trap;");
                	if(index >= recv_capacity) asm("trap;");
                	return recv_queues[(size_t)(q_id)*recv_capacity+index];
				}

				__forceinline__ __device__ RECV_T get_recv_item_volatile(COUNTER_T index, int q_id) {
                	if(q_id >= n_pes-1)  asm("trap;");
                	if(index >= recv_capacity) asm("trap;");
                	return ((volatile RECV_T *)recv_queues)[(size_t)(q_id)*recv_capacity+index];
            	}

            	__forceinline__ __device__ COUNTER_T len(Priority prio) const
            	{return *(end+prio*PADDING_SIZE);}

				__forceinline__ __device__ void local_push_warp(Priority prio, LOCAL_T item) const {
                	unsigned mask = __activemask();
                	unsigned mask_prio = __match_any_sync(mask, prio);
                	uint32_t total = __popc(mask_prio);
                	uint32_t rank = __popc(mask_prio & lanemask_lt());
                	COUNTER_T alloc = 0xffffffff;
                	if(rank == 0)
                	    alloc = atomicAdd((COUNTER_T *)(end_alloc+prio*PADDING_SIZE), total);
                	alloc = __shfl_sync(mask_prio, alloc, __ffs(mask_prio)-1);

                	__syncwarp(mask);
                	if(alloc+total > local_capacity)
                	    asm("trap;");
                	*(local_queues + uint64_t(prio)*local_capacity + alloc+rank) = item;
                	__threadfence();

                	if(!rank) {
                	    COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+prio*PADDING_SIZE), (alloc+total));
                	    //__threadfence();
                	    if(maxNow < alloc+total) maxNow = alloc+total;
                	    COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+prio*PADDING_SIZE), total) + total;
                	    if(validC == maxNow)
                	        atomicMax((COUNTER_T *)(end+prio*PADDING_SIZE), maxNow);
                	}
                	__syncwarp(mask);
            	}

				__forceinline__ __device__ void local_push_cta(Priority prio, bool ifpush, LOCAL_T item) const
            	{
            	    __shared__ COUNTER_T total_cta[2];
            	    __shared__ COUNTER_T res[2];
            	    if(threadIdx.x < 2)
            	    {
            	        total_cta[threadIdx.x]=0;
            	        res[threadIdx.x] = 0xffffffff;
            	    }
            	    __syncthreads();

            	    uint32_t alloc=0xffffffff;
            	    uint32_t rank=0xffffffff;
            	    unsigned mask_warp = __ballot_sync(FULLMASK_32BITS, ifpush);
            	    if(ifpush) {
            	        unsigned mask = __match_any_sync(mask_warp, prio);
            	        uint32_t total = __popc(mask);
            	        rank = __popc(mask & lanemask_lt());
            	        if(rank == 0)
            	            alloc = atomicAdd((total_cta+prio), total);
            	        alloc = __shfl_sync(mask, alloc, __ffs(mask)-1);
            	    }
            	    __syncthreads();

            	    if(threadIdx.x < 2 && total_cta[threadIdx.x])
            	    {
            	        res[threadIdx.x] = atomicAdd( (COUNTER_T *)(end_alloc+threadIdx.x*PADDING_SIZE), total_cta[threadIdx.x]);
            	        assert(res[threadIdx.x]!=0xffffffff);
            	        assert(res[threadIdx.x]+total_cta[threadIdx.x]< local_capacity);
            	    }

            	    __syncthreads();
            	    if(ifpush)
            	    {
            	        //if(alloc==0xffffffff)
            	        //    asm("trap;");
            	        //if(rank == 0xffffffff)
            	        //    asm("trap;");
            	        //if(res[prio] == 0xffffffff)
            	        //    asm("trap;");
            	        assert(alloc!=0xffffffff);
            	        assert(rank != 0xffffffff);
            	        assert(res[prio]!=0xffffffff);
            	        *(local_queues+uint64_t(prio)*local_capacity+res[prio]+alloc+rank) = item;
            	        __threadfence();
            	    }

            	    __syncthreads();
            	    if(threadIdx.x < 2 && total_cta[threadIdx.x]) {
            	        COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+threadIdx.x*PADDING_SIZE),res[threadIdx.x]+total_cta[threadIdx.x]);
            	        //__threadfence();
            	        if(maxNow < res[threadIdx.x]+total_cta[threadIdx.x])
            	            maxNow = res[threadIdx.x]+total_cta[threadIdx.x];
            	        COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+threadIdx.x*PADDING_SIZE), total_cta[threadIdx.x])+total_cta[threadIdx.x];
            	        if(maxNow == validC)
            	            atomicMax((COUNTER_T *)(end+threadIdx.x*PADDING_SIZE), validC);
            	    }
            	}

				__forceinline__ __device__ void remote_push_cta(bool ifpush, int pe, const RECV_T& item) const {
                	for(int pe_range=0; pe_range < n_pes; pe_range+=32) {
                	    __shared__ COUNTER_T res[32];
                	    __shared__ int total_cta[32];
                	    if(threadIdx.x < 32) {
                	        res[threadIdx.x] = 0xffffffff; total_cta[threadIdx.x] = 0;
                	    }
                	    __syncthreads();

                	    bool _ifpush = ifpush && pe >= pe_range && pe < pe_range+32 && !same_group(pe);
                	    uint32_t alloc;
                	    uint32_t rank;
                	    unsigned mask_warp = __ballot_sync(0xffffffff, _ifpush);
                	    if(_ifpush) {
                	        unsigned mask = __match_any_sync(mask_warp, pe);
                	        uint32_t total = __popc(mask);
                	        rank = __popc(mask & lanemask_lt());
                	        if(!rank) alloc = atomicAdd(total_cta+pe-pe_range, total);
                	        alloc = __shfl_sync(mask, alloc, __ffs(mask)-1);
                	    }
                	    __syncthreads();
                	    if(threadIdx.x < 32 && total_cta[threadIdx.x]) {
                	        int aggregate_pe_offset = (pe_range+threadIdx.x) >= (my_pe/group_size+1)*group_size? pe_range+threadIdx.x-group_size:pe_range+threadIdx.x;
                	        assert((pe_range+threadIdx.x)/group_size != my_pe/group_size);
                	        res[threadIdx.x] = atomicAdd((COUNTER_T *)(end_alloc+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), total_cta[threadIdx.x]);
                	        assert(res[threadIdx.x]+total_cta[threadIdx.x] < recv_capacity);
                	        //if(res[threadIdx.x]+total_cta[threadIdx.x] >= recv_capacity)
                	        //  asm("trap;");
                	    }
                	    __syncthreads();
                	    if(_ifpush) {
                	        if(res[pe-pe_range]==0xffffffff)
                	            asm("trap;");
                	        size_t aggregate_pe_offset = pe >= (my_pe/group_size+1)*group_size? pe-group_size:pe;
                	        aggregate_queues[aggregate_pe_offset*recv_capacity+res[pe-pe_range]+alloc+rank] = item;
                	        __threadfence();
							aggregate_maps[aggregate_pe_offset*(align_up_yc(recv_capacity,32))*4+res[pe-pe_range]+alloc+rank] = 1;
							//aggregate_maps[aggregate_pe_offset*(align_up_yc(recv_capacity,32))+res[pe-pe_range]+alloc+rank] = 1;
                	        ifpush = false;
                	    }
                	    __syncthreads();
			
						if(threadIdx.x < 32 && total_cta[threadIdx.x])
                	    {
                	        if(res[threadIdx.x]==0xffffffff)
                	            asm("trap;");
                	        int aggregate_pe_offset = (pe_range+threadIdx.x) >= (my_pe/group_size+1)*group_size? pe_range+threadIdx.x-group_size:pe_range+threadIdx.x;
							atomicMax((COUNTER_T *)(end_max+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), res[threadIdx.x]+total_cta[threadIdx.x]);
							atomicAdd((COUNTER_T *)(end_count+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), total_cta[threadIdx.x]);
                	        //COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), res[threadIdx.x]+total_cta[threadIdx.x]);
                	        //if(maxNow < res[threadIdx.x]+total_cta[threadIdx.x]) maxNow = res[threadIdx.x]+total_cta[threadIdx.x];
                	        //COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), total_cta[threadIdx.x])+total_cta[threadIdx.x];
                	        //if(maxNow == validC)
                	        //    atomicMax((COUNTER_T *)(end+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE), validC);
                	    }
                	    __syncthreads();
                	}
            	}

            	__forceinline__ __device__ COUNTER_T grab_cta(int fetch_size, Priority prio) const
            	{
            	    if(fetch_size <= 0) return COUNTER_T(0xffffffff);
            	    __shared__ COUNTER_T old;
            	    if(!threadIdx.x) {
            	        old = atomicAdd((COUNTER_T *)(start_alloc+prio*PADDING_SIZE), fetch_size);
            	    }
            	    __syncthreads();
            	    return old;
            	}

            	__forceinline__ __device__ COUNTER_T grab_thread(int fetch_size, Priority prio) const
            	{
            	    return atomicAdd((COUNTER_T *)(start_alloc+prio*PADDING_SIZE), fetch_size);
            	}

            	__forceinline__ __device__ void update_end_cta() const
            	{
            	    if(threadIdx.x < 2) {
						uint64_t end_max_count = *((uint64_t *)(end_max+threadIdx.x*PADDING_SIZE));
						COUNTER_T maxNow = (COUNTER_T)(end_max_count);
						if(maxNow == COUNTER_T(end_max_count>>32) && maxNow != *(end+threadIdx.x*PADDING_SIZE))
            	            atomicMax((COUNTER_T *)(end+threadIdx.x*PADDING_SIZE), maxNow);
            	    }
				}
			}; 
		} // dev

		template<typename COUNTER_T>
		struct res_info {
			float elapsed_ms = 0;
			int num_kernels = 0;
			COUNTER_T workload = 0;

			res_info(float time, int kernels, COUNTER_T w):
			elapsed_ms(time), num_kernels(kernels), workload(w) {}
	
			void print() {
				printf("elapsed time(ms) %6.2f, %6d kernels, workload %8d\n", elapsed_ms, num_kernels, workload);
			}	
		};

	 	template<typename RECV_T, typename LOCAL_T, typename THRESHOLD_TYPE, typename COUNTER_T, int PADDING_SIZE=1>
		class DistributedPriorityQueue: public DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> {
		public:
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::n_pes;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::my_pe;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::stop;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::start;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::total_num_queues;
			using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::min_iter;
			THRESHOLD_TYPE threshold;
        	THRESHOLD_TYPE threshold_delta;

			cudaStream_t streams[34];
						
			COUNTER_T *check_temp = NULL;

        	DistributedPriorityQueue() {}

        	~DistributedPriorityQueue() { release();}

			__host__ void init(int _n_pes, int _my_pe, int _group_id, int _group_size, int local_id, int local_size, COUNTER_T l_capacity, COUNTER_T r_capacity,
			THRESHOLD_TYPE thrshd, THRESHOLD_TYPE delta,  int m_iter=-1, bool PRINT_INFO = false)
        	{
				DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::baseInit
				(_n_pes, _my_pe, _group_id, _group_size, local_id, local_size,l_capacity, r_capacity, m_iter, 2, PRINT_INFO);

				threshold = thrshd;
				threshold_delta = delta;

				for(int i=0; i<34; i++)
        		CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
				if(m_iter == -1) {
					printf("allocate nvshmem for check_temp pe %d\n", my_pe);
					check_temp = (COUNTER_T *)nvshmem_malloc(n_pes*sizeof(COUNTER_T)*7*PADDING_SIZE*total_num_queues);
				}
        	}

			dev::DistributedPriorityQueue<RECV_T, LOCAL_T, THRESHOLD_TYPE, COUNTER_T, PADDING_SIZE>
        	deviceObject() const {
            	return dev::DistributedPriorityQueue<RECV_T, LOCAL_T, THRESHOLD_TYPE, COUNTER_T, PADDING_SIZE>
				(DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::deviceObject(), threshold, threshold_delta, check_temp);
        	}

			template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
			res_info<COUNTER_T> launchCTA(THRESHOLD_TYPE threshold, THRESHOLD_TYPE delta, uint32_t sharedMem, Functor F, Args... arg);			

			template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
			res_info<COUNTER_T> launchCTA_minIter(THRESHOLD_TYPE threshold, THRESHOLD_TYPE delta, uint32_t sharedMem, Functor F, Args... arg);			

			private:
        	void release(bool PRINT_INFO = false) { printf("pe %d release priority queue\n", my_pe);}
		};

	// Only 1 block will be launched
	template<int PADDING_SIZE, typename COUNTER_T, typename QTYPE>
	__global__ void checkStop_master_kernel(QTYPE wl)
	{
	    __shared__ int exit;
	    extern __shared__ COUNTER_T sharespace[];
	    /*
	        pe_send_array, size (n_pes)*(n_pes)
	        diagnal is local queue 
	                  recv_pe
	            0 | 0 1 ...... n
	    send    1 | 0 1 ...... n
	    pe      2 | 0 1 ...... n
	            . |
	            . |
	            . |
	            n | 0 1 2 .... n 
	
	        pe_recv_array, size (n_pes)*(n_pes)
	        diagnal is local queue 
	                  send_pe
	            0 | 0 1 ...... n
	    recv    1 | 0 1 ...... n
	    pe      2 | 0 1 ...... n
	            . |
	            . |
	            . |
	            n | 0 1 2 .... n
	    */
	    COUNTER_T * pe_send_array = sharespace;
	    COUNTER_T * pe_recv_array = pe_send_array + wl.n_pes*wl.n_pes;
	
	    if(threadIdx.x == wl.my_pe) { exit = 0; *(wl.stop) = 1;}
		__syncthreads();
	    for(int i = TID; i <wl.n_pes*wl.n_pes; i+=blockDim.x*gridDim.x)
	    {
	        pe_send_array[i] = 0;
	        pe_recv_array[i] = 0;
	    }
	
		//int peer_stop = 0;
	    if(threadIdx.x < wl.n_pes-1) {
    		int peer = (threadIdx.x >= wl.my_pe)? threadIdx.x+1:threadIdx.x;
	        int peer_stop = nvshmem_int_atomic_fetch((int *)wl.stop, peer);
	        //printf("pe %d, peer %d, stop %d\n", wl.my_pe, peer, peer_stop);
			if(peer_stop != 1) exit = 1;
	    }

		__syncthreads();

		if(exit == 1) return;
	
		if(threadIdx.x < wl.n_pes && threadIdx.x != wl.my_pe) {
            nvshmem_getmem(wl.check_temp+threadIdx.x*7*PADDING_SIZE*wl.total_num_queues, (COUNTER_T *)wl.start, sizeof(COUNTER_T)*7*PADDING_SIZE*wl.total_num_queues, threadIdx.x);
        }
        __syncthreads();

        for(int idx = threadIdx.x; idx < 2*wl.n_pes*wl.n_pes; idx +=blockDim.x) {
            int pe = idx/(2*wl.n_pes);
            // recv = 0 send = 1
            int recvOrsend = (idx%(2*wl.n_pes))/(wl.n_pes);
            int peer_offset = (idx%(2*wl.n_pes))%(wl.n_pes);
            COUNTER_T *temp_write_share_ptr = (recvOrsend == 0)? pe_recv_array: pe_send_array;
            volatile COUNTER_T *temp_read_ptr = (pe == wl.my_pe)? wl.start: wl.check_temp;

            int offset = (pe!=wl.my_pe)*pe*7*PADDING_SIZE*wl.total_num_queues;
            offset = offset + (pe != peer_offset)*wl.num_local_queues*PADDING_SIZE + recvOrsend*2*wl.total_num_queues*PADDING_SIZE;
            offset = offset + recvOrsend*(pe!=peer_offset)*(wl.n_pes-1)*PADDING_SIZE;
            offset = offset + (peer_offset < pe)*peer_offset*PADDING_SIZE;
            offset = offset + (peer_offset > pe)*(peer_offset-1)*PADDING_SIZE;

            temp_write_share_ptr[pe*wl.n_pes+peer_offset] = *(temp_read_ptr+offset);

            //if(recvOrsend == 0)
            //  printf("pe %d recevi from %d: %d\n", pe, peer_offset, pe_recv_array[pe*wl.n_pes+peer_offset]);
            //else if(recvOrsend == 1)
            //  printf("pe %d send to %d: %d\n", pe, peer_offset, pe_send_array[pe*wl.n_pes+peer_offset]);

        }
        __syncthreads();

	    for(int idx = TID; idx < wl.n_pes*wl.n_pes; idx +=blockDim.x*gridDim.x) {
	        int send_pe = idx/wl.n_pes;
	        int recv_pe = idx%wl.n_pes;
	        if(pe_recv_array[recv_pe*wl.n_pes+send_pe] != pe_send_array[send_pe*wl.n_pes+recv_pe]) {
	            //printf("send_pe %d, send %d, recv_pe %d, recv %d, not equal\n", send_pe, pe_send_array[send_pe*wl.n_pes+recv_pe], recv_pe, pe_recv_array[recv_pe*wl.n_pes+send_pe]);
	            exit = 1;
	            break;
	        }
	    }
	
	    __syncthreads();
	    if(exit == 1)
	        return;
	
	    if(threadIdx.x < wl.n_pes) {
	        int ret = nvshmem_int_atomic_fetch((int *)wl.stop, threadIdx.x);
	        if(ret != 1)
	            exit = 1;
	    }
	    __syncthreads();
	    if(exit == 1)
	        return;
	
	    if(threadIdx.x < wl.n_pes) {
	        int ret = nvshmem_int_atomic_compare_swap((int *)wl.stop, 1, 3, threadIdx.x);
	        //printf("update peer %d stop to 3, ret %d\n", threadIdx.x, ret);
	        assert(ret == 1);
	    }
	}
   	
	template<typename RECV_T, typename LOCAL_T, typename THRESHOLD_TYPE, typename COUNTER_T, int PADDING_SIZE>
	void checkStop(DistributedPriorityQueue<RECV_T, LOCAL_T, THRESHOLD_TYPE, COUNTER_T, PADDING_SIZE> &wl, cudaStream_t stream)
	{
	    int share_mem = sizeof(COUNTER_T)*wl.n_pes*wl.n_pes*2;
	    int blockSize = align_up_yc(wl.n_pes, 32);
	    printf("pe %d, check stop condition, share_mem %d, blockSize %d\n", wl.my_pe, share_mem, blockSize);
	    checkStop_master_kernel<PADDING_SIZE, COUNTER_T><<<1, blockSize, share_mem, stream>>>(wl.deviceObject());
	}
 
	template<typename RECV_T, typename LOCAL_T, typename THRESHOLD_TYPE, typename COUNTER_T, int PADDING_SIZE>
	template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
	res_info<COUNTER_T> DistributedPriorityQueue<RECV_T, LOCAL_T, THRESHOLD_TYPE, COUNTER_T, PADDING_SIZE>::
	launchCTA(THRESHOLD_TYPE threshold, THRESHOLD_TYPE delta, uint32_t sharedMem, Functor F, Args... arg) {
		COUNTER_T end_host[n_pes+1] = {0};
		COUNTER_T start_host[n_pes+1] = {0};
		COUNTER_T size[n_pes+1] = {0};
		COUNTER_T temp_load[(n_pes+1)*PADDING_SIZE] = {0};
		int iter = 0;
		int log_iter = 0;
		THRESHOLD_TYPE threshold_temp = threshold;
        Priority pivot = Priority::HIGH;
		int stream_id = log_iter&31;
		int stop_flag = 0;

		GpuTimer timer;
        nvshmem_barrier_all();
        timer.Start();
        while(stop_flag != 3)
        {
            CUDA_CHECK(cudaMemcpyAsync(temp_load, end, sizeof(COUNTER_T)*(n_pes+1)*PADDING_SIZE, cudaMemcpyDeviceToHost, streams[32]));
            CUDA_CHECK(cudaStreamSynchronize(streams[32]));
            for(int q_id=0; q_id<n_pes+1; q_id++) end_host[q_id] = temp_load[q_id*PADDING_SIZE];
            for(int q_id=0; q_id<n_pes+1; q_id++) size[q_id] = end_host[q_id] - start_host[q_id];
			
            for(int recv_id = 2; recv_id < n_pes+1; recv_id++) {
                if(size[recv_id] > 0) {
                    if(stop_flag != 0) {
                        stop_flag = 0;
                        CUDA_CHECK(cudaMemcpyAsync(stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[33]));
                    }
					func(F, start_host[recv_id], size[recv_id], threshold_temp, pivot, recv_id-1, sharedMem, streams[stream_id], arg...);
                    log_iter++;
                    stream_id = log_iter&31;
                    start_host[recv_id] = end_host[recv_id];
                    iter = 0;
                }
            }
			// if high priority local queue has items
            if(size[pivot] > 0) {
                if(stop_flag != 0) {
                    stop_flag = 0;
                    CUDA_CHECK(cudaMemcpyAsync(stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[33]));
                }
				func(F, start_host[pivot], size[pivot], threshold_temp, pivot, 0, sharedMem, streams[stream_id], arg...);
                log_iter++;
                stream_id = log_iter&31;
                start_host[pivot] = end_host[pivot];
                iter = 0;
            }
            else if(size[!pivot] > 0) {
                if(stop_flag != 0) {
                   stop_flag = 0;
                   CUDA_CHECK(cudaMemcpyAsync(stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[33]));
                }
                pivot = static_cast<Priority>(!pivot);
                //threshold_temp += (threshold_increment+StartBatch*10);
                threshold_temp += delta;
				func(F, start_host[pivot], size[pivot], threshold_temp, pivot, 0, sharedMem, streams[stream_id], arg...);
                log_iter++;
                stream_id = log_iter&31;
                start_host[pivot] = end_host[pivot];
                iter = 0;
            }
			iter++;

            if(iter >= 10) {
                bool not_complete = false;
                for(int i=0; i<32; i++) {
                    if(cudaStreamQuery(streams[i]) == cudaErrorNotReady) {
                        not_complete = true;
                        break;
                    }
                }

                if(not_complete == false) {
                    for(int q=0; q<total_num_queues; q++)
                        CUDA_CHECK(cudaMemcpyAsync(start+q*PADDING_SIZE, start_host+q, sizeof(COUNTER_T), cudaMemcpyHostToDevice, streams[33]));
                    CUDA_CHECK(cudaMemcpyAsync(&stop_flag, stop, sizeof(int), cudaMemcpyDeviceToHost, streams[33]));
                    CUDA_CHECK(cudaStreamSynchronize(streams[33]));
					printf("pe %d, stop_flag %d\n", my_pe, stop_flag);
                    if(my_pe == 0 && stop_flag != 3) {
                        checkStop(*this, streams[33]);
					}
                    else if (my_pe != 0 && stop_flag == 0){
                        stop_flag = 1;
                        CUDA_CHECK(cudaMemcpyAsync(stop, &stop_flag, sizeof(int), cudaMemcpyHostToDevice, streams[33]));
                    }
                }

            }
		}
		timer.Stop();
		for(int i=0; i<34; i++)
			CUDA_CHECK(cudaStreamSynchronize(streams[i]));

		float elapsed = timer.ElapsedMillis();
		COUNTER_T workload = 0;
		for(int q=0; q<n_pes+1; q++) workload += end_host[q];
        SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f, %8d kernels, workload %8d\n", elapsed, log_iter, workload));		
		return res_info(elapsed, log_iter, workload);
	}

	template<typename RECV_T, typename LOCAL_T, typename THRESHOLD_TYPE, typename COUNTER_T, int PADDING_SIZE>
	template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
	res_info<COUNTER_T> DistributedPriorityQueue<RECV_T, LOCAL_T, THRESHOLD_TYPE, COUNTER_T, PADDING_SIZE>::launchCTA_minIter(THRESHOLD_TYPE threshold, THRESHOLD_TYPE delta, uint32_t sharedMem, Functor F, Args... arg) {
		COUNTER_T end_host[n_pes+1] = {0};
		COUNTER_T start_host[n_pes+1] = {0};
		COUNTER_T size[n_pes+1] = {0};
		COUNTER_T temp_load[(n_pes+1)*PADDING_SIZE] = {0};
		
		int iter = 0;
		int log_iter = 0;
		int threshold_temp = threshold;
		Atos::MAXCOUNT::Priority pivot = Atos::MAXCOUNT::Priority::HIGH;
		int stream_id = log_iter&31;	
		
		GpuTimer timer;
		nvshmem_barrier_all();
		
		timer.Start();
		while(iter < min_iter)
		{
			CUDA_CHECK(cudaMemcpyAsync(temp_load, end, sizeof(COUNTER_T)*(n_pes+1)*PADDING_SIZE, cudaMemcpyDeviceToHost, streams[32]));
			CUDA_CHECK(cudaStreamSynchronize(streams[32]));
			for(int q_id=0; q_id<n_pes+1; q_id++) end_host[q_id] = temp_load[q_id*PADDING_SIZE];
			for(int q_id=0; q_id<n_pes+1; q_id++) size[q_id] = end_host[q_id] - start_host[q_id];
			
			for(int recv_id = 2; recv_id < n_pes+1; recv_id++) {
                if(size[recv_id] > 0) {
					func(F, start_host[recv_id], size[recv_id], threshold_temp, pivot, recv_id-1, sharedMem, streams[stream_id], arg...);
                    log_iter++;
                    stream_id = log_iter&31;
                    start_host[recv_id] = end_host[recv_id];
                    iter = 0;
                }
            }
			// if high priority local queue has items
            if(size[pivot] > 0) {
				func(F, start_host[pivot], size[pivot], threshold_temp, pivot, 0, sharedMem, streams[stream_id], arg...);
                log_iter++;
                stream_id = log_iter&31;
                start_host[pivot] = end_host[pivot];
                iter = 0;
            }
            else if(size[!pivot] > 0) {
                pivot = static_cast<Priority>(!pivot);
                //threshold_temp += (threshold_increment+StartBatch*10);
                threshold_temp += delta;
				func(F, start_host[pivot], size[pivot], threshold_temp, pivot, 0, sharedMem, streams[stream_id], arg...);
                log_iter++;
                stream_id = log_iter&31;
                start_host[pivot] = end_host[pivot];
                iter = 0;
            }
			iter++;
		}
		timer.Stop();
		for(int q=0; q<total_num_queues; q++)
       		CUDA_CHECK(cudaMemcpyAsync(start+q*PADDING_SIZE, start_host+q, sizeof(COUNTER_T), cudaMemcpyHostToDevice, streams[33]));
		for(int i=0; i<34; i++)
			CUDA_CHECK(cudaStreamSynchronize(streams[i]));
		float elapsed = timer.ElapsedMillis();
		COUNTER_T workload = 0;
		for(int q=0; q<n_pes+1; q++) workload += end_host[q];
        SERIALIZE_PRINT(my_pe, n_pes, printf("time %8.2f, %8d kernels, workload %8d\n", elapsed, log_iter, workload));		
		return res_info(elapsed, log_iter, workload);
	}
	} // MAXCOUNT
} // Atos

#endif
