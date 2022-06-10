#ifndef DISTRIBUTED_QUEUE
#define DISTRIBUTED_QUEUE

#include <assert.h>

#include "../util/error_util.cuh"
#include "../util/util.cuh"
#include "../util/nvshmem_util.cuh"

#include "distr_queue_base.cuh"

namespace Atos {
	namespace MAXCOUNT{
    namespace dev{
		template<typename LOCAL_T, typename RECV_T>
		class Work {
		public:
			enum class ValueType { LOCAL, REMOTE, UNKNOWN };
           	__device__ Work(): mType(ValueType::UNKNOWN) {}
			__device__ Work(LOCAL_T l): mType(ValueType::LOCAL),localWork(l) {}
			__device__ Work(RECV_T  r): mType(ValueType::REMOTE), remoteWork(r) {}
	
			__device__ Work(const Work& val) { Copy(val); }	
			__device__ Work& operator=(const Work& val) { Copy(val); return *this;}
			
			ValueType mType;	
			union {
                LOCAL_T localWork;
                RECV_T remoteWork;
			};
		private:
			__device__ void Copy(const Work& val)
			{
			  switch (val.mType)
			  {
			  case ValueType::LOCAL:
			    localWork = val.localWork;
			    break;
			  case ValueType::REMOTE:
			    remoteWork = val.remoteWork;
			    break;
			  case ValueType::UNKNOWN:
			    break;
			  }
			
			  mType = val.mType;
			}	
        };
        // SHARE_BUFFER_SIZE > blockDim.x+ n*BATCH_SIZE, n>=1
        template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE=1>
        class DistributedQueue: public Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>
        {
        public:
			using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::n_pes;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::my_pe;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::group_size;

            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_alloc;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_max;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::end_count;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::start_alloc;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::start;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::stop;

            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::local_capacity;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::recv_capacity;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::local_queues;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::recv_queues;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::aggregate_queues;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::aggregate_maps;

            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::num_local_queues;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::num_aggregate_queues;
            using Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::min_iter;

            typedef Work<LOCAL_T, RECV_T> union_type;

            COUNTER_T * execute=NULL;
            COUNTER_T size_per_queue=0;
			COUNTER_T *workers_stop; //local end, recv ends, voting slots
			COUNTER_T *check_temp = NULL;

            DistributedQueue() {}
			DistributedQueue(const Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> &base, 
			COUNTER_T *_execute=NULL, COUNTER_T _size_per_queue=0, COUNTER_T * _workers_stop=NULL, COUNTER_T * _check_temp=NULL):
			Atos::MAXCOUNT::dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>(base),
			execute(_execute), size_per_queue(_size_per_queue), workers_stop(_workers_stop), check_temp(_check_temp)
			{ }
		
			__forceinline__ __device__ bool same_group(int pe) const {return pe/group_size==my_pe/group_size;}
            __forceinline__ __device__ COUNTER_T get_local_capacity() const {return local_capacity;}
            __forceinline__ __device__ COUNTER_T get_recv_capacity() const {return recv_capacity;}
            //TODO: Can use Work to unify the get_item APIs, may be do in the futrue
            __forceinline__ __device__ LOCAL_T get_local_item(COUNTER_T index, int q_id=0) {
                if(q_id >= num_local_queues) asm("trap;");
                if(index >= local_capacity) asm("trap;");
                return local_queues[(size_t)(q_id)*local_capacity+index];
            }
            __forceinline__ __device__ RECV_T get_recv_item(COUNTER_T index, int q_id) {
                if(q_id >= n_pes-1)  asm("trap;");
                return recv_queues[(size_t)(q_id)*recv_capacity+(index&(recv_capacity-1))];
            }
            __forceinline__ __device__ LOCAL_T get_local_item_volatile(COUNTER_T index, int q_id=0) {
                if(q_id >= num_local_queues) asm("trap;");
                if(index >= local_capacity) asm("trap;");
                return ((volatile LOCAL_T *)local_queues)[(size_t)(q_id)*local_capacity+index];
            }
            __forceinline__ __device__ RECV_T get_recv_item_volatile(COUNTER_T index, int q_id) {
                if(q_id >= n_pes-1)  asm("trap;");
                return ((volatile RECV_T *)recv_queues)[(size_t)(q_id)*recv_capacity+(index&(recv_capacity-1))];
            }

			// Only single queue. TODO: push APIs for multiple queues
            __forceinline__ __device__ void local_push_warp(const LOCAL_T& item) const {
                unsigned mask = __activemask();
                uint32_t total = __popc(mask);
                unsigned int rank = __popc(mask & lanemask_lt());
                int leader = __ffs(mask)-1;
                COUNTER_T alloc;
                if(rank == 0)
                    alloc = atomicAdd((COUNTER_T *)end_alloc, total);
                alloc = __shfl_sync(mask, alloc, leader);
                if(alloc+total >= local_capacity)
                    asm("trap;");

                local_queues[alloc+rank] = item;
				__threadfence();

                if(rank == 0) {
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)end_max, (alloc+total));
                    //__threadfence();
                    if(maxNow < alloc+total) maxNow = alloc+total;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)end_count, total)+total;
                    if(validC == maxNow)
                        atomicMax((COUNTER_T *)end, maxNow);
                }
                __syncwarp(mask); 
            }

            // only 1 local queue assumed
            __forceinline__ __device__ void local_push_cta(bool ifpush, const LOCAL_T& item) const {
                __shared__ COUNTER_T res;
                __shared__ int total_cta; 
                //init res and total_cta to 0
                if(threadIdx.x == 0)
                {
                    res = 0xffffffff; total_cta = 0;
                }
            
                __syncthreads();
            
                //each warp counts number of item need to be pushed and atomicAdd to the total_cta
                unsigned mask = __ballot_sync(0xffffffff, ifpush);
                uint32_t rank = __popc(mask & lanemask_lt());
                uint32_t alloc;
                if(ifpush && rank == 0) {
                    alloc = atomicAdd((int *)(&total_cta), __popc(mask));
                }
                alloc = __shfl_sync(0xffffffff, alloc, __ffs(mask)-1);
            
                //synchronize all threads of cta, make sure they add to total_cta
                __syncthreads();
            
                //first thread will atomicAdd to end_alloc to reserve a room for push
                if(threadIdx.x == 0 && total_cta)
                {
                    res = atomicAdd((COUNTER_T *)end_alloc, total_cta);
					assert(res+total_cta < local_capacity);
                    //if(res+total_cta >= local_capacity) 
                    //    asm("trap;");
                    //printf("queue is full res %d, total_cta %d\n", res, total_cta);
                }
            
                //synchronize all threads of cta, make sure res value is ready
                __syncthreads();

                if(ifpush)
                {
                    if(res==0xffffffff)
                        asm("trap;");
                    local_queues[res+alloc+rank] = item;
                    __threadfence();
                }
                __syncthreads();

				if(threadIdx.x == 0 && total_cta)
                {
                    if(res==0xffffffff)
                        asm("trap;");
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)end_max, res+total_cta);
                    if(maxNow < res+total_cta) maxNow = res+total_cta;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)end_count, total_cta)+total_cta;
                    if(maxNow == validC)
                        atomicMax((COUNTER_T *)end, validC);
                }
            }


			__forceinline__ __device__ void remote_push_cta(bool ifpush, int pe, const RECV_T& item) const {
				for(int pe_range=0; pe_range < n_pes; pe_range+=32)	{
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
						assert(res[threadIdx.x]+total_cta[threadIdx.x] < recv_capacity+*(start_alloc+(num_local_queues+n_pes-1+aggregate_pe_offset)*PADDING_SIZE));
						assert(res[threadIdx.x]+total_cta[threadIdx.x] < 4*(align_up_yc(recv_capacity, 32)));
						//if(res[threadIdx.x]+total_cta[threadIdx.x] >= recv_capacity)
						//	asm("trap;");
					}	
					__syncthreads();
					if(_ifpush) {
						if(res[pe-pe_range]==0xffffffff)
							asm("trap;");
						size_t aggregate_pe_offset = pe >= (my_pe/group_size+1)*group_size? pe-group_size:pe;
						aggregate_queues[aggregate_pe_offset*recv_capacity+((res[pe-pe_range]+alloc+rank)&(recv_capacity-1))] = item;
						__threadfence();
						aggregate_maps[aggregate_pe_offset*(align_up_yc(recv_capacity,32))*4+res[pe-pe_range]+alloc+rank] = 1;
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

            __forceinline__ __device__ COUNTER_T grab_cta(const uint32_t &total, int q_id=0) const {
                if(total == 0) return COUNTER_T(0xffffffff);
                __shared__ COUNTER_T index;
                if(threadIdx.x == 0)
                {
                    index = atomicAdd((COUNTER_T *)(start_alloc+q_id*PADDING_SIZE), total);
                }
                __syncthreads();
                return index;
            }

			__forceinline__ __device__ COUNTER_T grab_thread(const uint32_t &total, int q_id) const {
                if(total == 0) return COUNTER_T(0xffffffff);
                return atomicAdd((COUNTER_T *)(start_alloc+q_id*PADDING_SIZE), total);
            }
			__forceinline__ __device__ COUNTER_T get_queue_end(int q_id) const {
				return *(end+q_id*PADDING_SIZE);
			}

			__forceinline__ __device__ void update_end(int q_id) const {
				uint64_t end_max_count = *((uint64_t *)(end_max+q_id*PADDING_SIZE));
				COUNTER_T maxNow = COUNTER_T(end_max_count);
				if(COUNTER_T(end_max_count>>32) == maxNow && *(end+q_id*PADDING_SIZE) != maxNow)
                	atomicMax((COUNTER_T *)(end+q_id*PADDING_SIZE), maxNow);
            }

            __forceinline__ __device__ void update_end_execute(int q_id, COUNTER_T init_value = std::numeric_limits<COUNTER_T>::max()) const {
                __shared__ COUNTER_T warp_min[32];
                if(LANE_ == 0)
                    warp_min[threadIdx.x>>5] = init_value;
                __syncwarp();
                COUNTER_T loadValue = init_value;
                for(int idx = threadIdx.x; idx<gridDim.x; idx+=blockDim.x)
                {
                    loadValue = execute[q_id*size_per_queue+idx];
                    //printf("pe %d, wl_id %d, execute[%d] %d\n", my_pe, q_id, q_id, loadValue);
                    unsigned mask = __activemask();
                    for(int offset=16; offset>0; offset/=2)
                    {
                        uint32_t temp = __shfl_down_sync(mask, loadValue, offset);
                        __syncwarp(mask);
                        if(LANE_+offset < __popc(mask))
                           loadValue = min(loadValue, temp);
                        __syncwarp(mask);
                    }
                    if(LANE_ == 0)
                        warp_min[threadIdx.x>>5] = min(warp_min[threadIdx.x>>5],loadValue);
                }
                __syncthreads();
                if(threadIdx.x < ((blockDim.x+31)>>5))
                {
                    unsigned mask = __activemask();
                    loadValue = warp_min[threadIdx.x];
                    for(int offset=16; offset>0; offset/=2)
                    {
                        uint32_t temp = __shfl_down_sync(mask, loadValue, offset);
                        __syncwarp(mask);
                        if(LANE_+offset < __popc(mask))
                            loadValue = min(loadValue, temp);
                        __syncwarp(mask);
                    }
                }
                if(threadIdx.x == 0) {
                    atomicMax((COUNTER_T *)start+q_id*PADDING_SIZE, loadValue);
                    //printf("pe %d, update q_id %d, with %d\n", my_pe, q_id, loadValue);
                }
                __syncthreads();
            }
        };
    } // end namespace dev

    template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE=1>
    class DistributedQueue: public DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>
    {
    public:
		using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::n_pes;
		using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::num_local_queues;
		using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::total_num_queues;
		using Atos::MAXCOUNT::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::stop;
        COUNTER_T *execute = NULL;
        COUNTER_T size_per_queue = 0;
		COUNTER_T *workers_stop = NULL;
		COUNTER_T *check_temp = NULL;

        DistributedQueue() {}

        ~DistributedQueue() { release();}

        __host__ void init(int _n_pes, int _my_pe, int _group_id, int _group_size, int local_id, int local_size, COUNTER_T l_capacity, COUNTER_T r_capacity, 
        int l_queues = 1, int m_iter=-1, bool PRINT_INFO = false)
        {
			DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::baseInit
                (_n_pes, _my_pe, _group_id, _group_size, local_id, local_size,l_capacity, r_capacity, m_iter, l_queues, PRINT_INFO);
			if(m_iter == -1)
				check_temp = (COUNTER_T *)nvshmem_malloc(n_pes*sizeof(COUNTER_T)*7*PADDING_SIZE*total_num_queues);
        }


		__host__ void reset(cudaStream_t stream, int execute_init_value = 0) const
        {
            DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::reset(stream);
            if(workers_stop!=NULL) {
            CUDA_CHECK(cudaMemset(workers_stop, 0xffffffff, sizeof(COUNTER_T)*(num_local_queues+n_pes-1)));
            CUDA_CHECK(cudaMemset(workers_stop+num_local_queues+n_pes-1, 0, sizeof(int)*size_per_queue));
            }
            if(execute!=NULL)
            CUDA_CHECK(cudaMemset(execute, execute_init_value, sizeof(COUNTER_T)*size_per_queue*(num_local_queues+n_pes-1)));
        }

        __host__ void setExecute(COUNTER_T _size_per_queue, int init_value)
        {
            if(size_per_queue == 0) {
                size_per_queue = _size_per_queue;
                CUDA_CHECK(cudaMalloc(&execute, sizeof(COUNTER_T)*size_per_queue*(num_local_queues+n_pes-1)));
				CUDA_CHECK(cudaMalloc(&workers_stop, sizeof(COUNTER_T)*(num_local_queues+n_pes-1)+sizeof(int)*size_per_queue));
            }
            else if(size_per_queue < _size_per_queue) {
                size_per_queue = _size_per_queue;
                CUDA_CHECK(cudaFree(execute));
                CUDA_CHECK(cudaMalloc(&execute, sizeof(COUNTER_T)*size_per_queue*(num_local_queues+n_pes-1)));
				CUDA_CHECK(cudaFree(workers_stop));
                CUDA_CHECK(cudaMalloc(&workers_stop, sizeof(COUNTER_T)*(num_local_queues+n_pes-1)+sizeof(int)*size_per_queue));
            }
			CUDA_CHECK(cudaMemset(execute, init_value, sizeof(COUNTER_T)*size_per_queue*(num_local_queues+n_pes-1)));
			CUDA_CHECK(cudaMemset(workers_stop, 0xffffffff, sizeof(COUNTER_T)*(num_local_queues+n_pes-1)));
            CUDA_CHECK(cudaMemset(workers_stop+num_local_queues+n_pes-1, 0, sizeof(int)*size_per_queue));
        }

        dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> 
        deviceObject() const {
            return dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>
            (DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::deviceObject(), execute, size_per_queue, workers_stop, check_temp);
        }

		template<int FETCH_SIZE, typename Functor, typename... Args>
		__host__ void launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

		template<int FETCH_SIZE, typename Functor, typename... Args>
		__host__ void launchCTA(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

		__host__ void checkStop(cudaStream_t &stream, int *stop_flag);
    private:
        void release() { 
			if(size_per_queue!=0)
				CUDA_CHECK(cudaFree(execute));
			if(workers_stop)
                CUDA_CHECK(cudaFree(workers_stop));
		}
    };

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE, 
	int FETCH_SIZE, typename Functor, typename... Args >
    __launch_bounds__(1024,1)
    __global__ void _launchCTA_minIter(dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> q, Functor F, Args... args)
    {
		assert(q.num_local_queues+q.n_pes-1 <= 32);
		extern __shared__ COUNTER_T shareSpace[];
		// end for num_local_queues + n_pes-1 remote queues
        COUNTER_T *end = shareSpace;
		COUNTER_T *index = end + q.num_local_queues+q.n_pes-1;
		int *fs = (int *)(index + q.num_local_queues+q.n_pes-1);
   
        int iter = 0;
		if(threadIdx.x < q.num_local_queues+q.n_pes-1)
		{
			fs[threadIdx.x] = FETCH_SIZE;
			end[threadIdx.x] = q.get_queue_end(threadIdx.x);
        	index[threadIdx.x] = q.grab_thread(FETCH_SIZE, threadIdx.x);
			q.start[threadIdx.x*q.padding] = 0xfffffff;
		}
        __syncthreads();

        do {
			for(int wl_id = q.num_local_queues+q.n_pes-2; wl_id >=0; wl_id--) 
			{
            	while(index[wl_id] < end[wl_id])
            	{
        			dev::Work<LOCAL_T, RECV_T> node;
            	    if(threadIdx.x < min(end[wl_id]-index[wl_id], fs[wl_id]))
            	    {
						if(wl_id < q.num_local_queues) {
            	        	node = q.get_local_item_volatile(index[wl_id]+threadIdx.x);
						}
						else {
							node = q.get_recv_item_volatile(index[wl_id]+threadIdx.x, wl_id-q.num_local_queues);
						}
            	    }
            	    __syncthreads();

            	    func(F, node, min(end[wl_id]-index[wl_id], fs[wl_id]), index[wl_id], (wl_id >= q.num_local_queues), args...);
            	    __syncthreads();

					if(threadIdx.x == 0) {
						fs[wl_id] = min(end[wl_id]-index[wl_id], fs[wl_id]);
            	    	if(fs[wl_id]+index[wl_id] == align_up_yc(index[wl_id]+1, FETCH_SIZE))
            	    	{
            	    	    index[wl_id] = q.grab_thread(FETCH_SIZE, wl_id);
							fs[wl_id] = FETCH_SIZE;
            	    	}
            	    	else
            	    	{
            	    	    index[wl_id] = index[wl_id] + fs[wl_id];
            	    	    fs[wl_id] = align_up_yc(index[wl_id], FETCH_SIZE) - index[wl_id];
            	    	}
					}
            	    __syncthreads();
            	} //while
				if(threadIdx.x == 0) {
					q.update_end(wl_id);
					end[wl_id] = q.get_queue_end(wl_id);
				}
			} //for
			__syncthreads();

			bool ifempty = false;
        	if(LANE_ < q.num_local_queues+q.n_pes-1)
        	    ifempty = index[LANE_] >= end[LANE_];
			if(__ballot_sync(0xffffffff, ifempty) == (0xffffffff>>(32-(q.num_local_queues+q.n_pes-1))))
			{
				if(threadIdx.x < q.num_local_queues)
					q.update_end(threadIdx.x);

				if(threadIdx.x < q.num_local_queues+q.n_pes-1)
            	    end[threadIdx.x] = q.get_queue_end(threadIdx.x);
            	__syncthreads();
				ifempty = false;
				if(LANE_ < q.num_local_queues+q.n_pes-1)
					ifempty = index[LANE_] >= end[LANE_];
			}

			if(__ballot_sync(0xffffffff, ifempty) == (0xffffffff>>(32-(q.num_local_queues+q.n_pes-1))))
        	{
        	    if(*(q.stop) == gridDim.x*q.num_local_queues) {
        	        break;
				}
				__syncthreads();
        	    if(threadIdx.x == 0)
        	    {
        	        iter++;
        	        if(iter == q.min_iter)
        	            atomicAdd((COUNTER_T *)(q.stop), 1);
        	    }
        	    //q.update_end_execute();
        	    //__nanosleep(100);
        	}
			
            //__syncthreads();
        }
        while(true);
		__syncthreads();
        if(threadIdx.x < q.num_local_queues+q.n_pes-1)
			atomicMin((COUNTER_T *)(q.start+threadIdx.x*q.padding), index[threadIdx.x]);
    }

    template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
    __device__ bool checkStop_master_CTA(dev::DistributedQueue
            <RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> &wl,
            COUNTER_T *pe_send_array, COUNTER_T *pe_recv_array, COUNTER_T *check_temp);

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE, 
	int FETCH_SIZE, typename Functor, typename... Args >
    __launch_bounds__(640,1)
    __global__ void _launchCTA(dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> q, Functor F, Args... args)
    {
		assert(q.num_local_queues+q.n_pes-1 <= 32);
		extern __shared__ COUNTER_T shareSpace[];
		// end for num_local_queues + n_pes-1 remote queues
        COUNTER_T *end = shareSpace;
		COUNTER_T *index = end + q.num_local_queues+q.n_pes-1;
		int *fs = (int *)(index + q.num_local_queues+q.n_pes-1);

		// global master + bid 1 need (2+2*n_pes*n_pes)*sizeof(COUNTER_T),
		// local master + bid 0 need (num_local_queues+n_pes-1)*sizeof(COUNTER_T)+gridDim.x*sizeof(int)
		COUNTER_T *ends_record = (COUNTER_T *)(fs + q.num_local_queues+q.n_pes-1);
   
        int iter = 0;
		if(threadIdx.x < q.num_local_queues+q.n_pes-1)
		{
			fs[threadIdx.x] = FETCH_SIZE;
			end[threadIdx.x] = q.get_queue_end(threadIdx.x);
        	index[threadIdx.x] = q.grab_thread(FETCH_SIZE, threadIdx.x);
			q.execute[threadIdx.x*q.size_per_queue+blockIdx.x] = index[threadIdx.x];	
			ends_record[threadIdx.x] = 0;
		}
        __syncthreads();

        do {
			for(int wl_id = q.num_local_queues+q.n_pes-2; wl_id >=0; wl_id--) 
			{
            	while(index[wl_id] < end[wl_id])
            	{

					if(threadIdx.x == 0)
						*(q.stop) = 0;
					__syncwarp();

        			dev::Work<LOCAL_T, RECV_T> node;
            	    if(threadIdx.x < min(end[wl_id]-index[wl_id], fs[wl_id]))
            	    {
						if(wl_id < q.num_local_queues) {
            	        	node = q.get_local_item_volatile(index[wl_id]+threadIdx.x);
						}
						else {
							node = q.get_recv_item_volatile(index[wl_id]+threadIdx.x, wl_id-q.num_local_queues);
						}
            	    }
            	    __syncthreads();

            	    func(F, node, min(end[wl_id]-index[wl_id], fs[wl_id]), index[wl_id], (wl_id >= q.num_local_queues), args...);
            	    __syncthreads();

					if(threadIdx.x == 0) {
						fs[wl_id] = min(end[wl_id]-index[wl_id], fs[wl_id]);
            	    	if(fs[wl_id]+index[wl_id] == align_up_yc(index[wl_id]+1, FETCH_SIZE))
            	    	{
            	    	    index[wl_id] = q.grab_thread(FETCH_SIZE, wl_id);
							fs[wl_id] = FETCH_SIZE;
            	    	}
            	    	else
            	    	{
            	    	    index[wl_id] = index[wl_id] + fs[wl_id];
            	    	    fs[wl_id] = align_up_yc(index[wl_id], FETCH_SIZE) - index[wl_id];
            	    	}
						q.execute[wl_id*q.size_per_queue+blockIdx.x] = index[wl_id];
					}
            	    __syncthreads();
            	} //while
				if(threadIdx.x == 0) {
					q.update_end(wl_id);
					end[wl_id] = q.get_queue_end(wl_id);
				}
			} //for
			__syncthreads();

			bool ifempty = false;
        	if(LANE_ < q.num_local_queues+q.n_pes-1)
        	    ifempty = index[LANE_] >= end[LANE_];
			if(__ballot_sync(0xffffffff, ifempty) == (0xffffffff>>(32-(q.num_local_queues+q.n_pes-1))))
			{
				if(*((volatile int *)q.stop) == 3) break;

				for(int q_id = 0; q_id < q.num_local_queues+q.n_pes-1; q_id++)
        	    	q.update_end_execute(q_id);
				
				if(threadIdx.x < q.num_local_queues)
					q.update_end(threadIdx.x);

				if(threadIdx.x < q.num_local_queues+q.n_pes-1)
            	    end[threadIdx.x] = q.get_queue_end(threadIdx.x);

				//local master
				if(blockIdx.x == 0)
				{
					__shared__ int check_other_workers;
					__shared__ int check_stop;
					if(threadIdx.x == 0) { check_other_workers = 0; check_stop = 0; }
					__syncthreads();
					ifempty = false;
					COUNTER_T end_alloc = 0;
					COUNTER_T local_start = 0;
					if(threadIdx.x < q.num_local_queues)
						end_alloc = *((volatile COUNTER_T *)q.end_alloc+threadIdx.x*PADDING_SIZE);
					else if (threadIdx.x < q.num_local_queues+q.n_pes-1)
						end_alloc = *((volatile COUNTER_T *)q.end+threadIdx.x*PADDING_SIZE);

					if(threadIdx.x < q.num_local_queues+q.n_pes-1) {
						local_start = *((volatile COUNTER_T *)q.start+threadIdx.x*PADDING_SIZE);
						ifempty = ((end_alloc == ends_record[threadIdx.x]) && (local_start == end_alloc) && (end_alloc == end[threadIdx.x]));
						if(end_alloc!=ends_record[threadIdx.x]) {
							assert(end_alloc > ends_record[threadIdx.x]);
							ends_record[threadIdx.x] = end_alloc;
						}
					}
											
					if( (threadIdx.x>>5) == 0 && 
						(__ballot_sync(0xffffffff, ifempty) ==  (0xffffffff>>(32-(q.num_local_queues+q.n_pes-1)))) )
					{
						iter++;
						if(threadIdx.x == 0) {
							check_stop = *((volatile int *)q.stop);
							if(iter > 5 && check_stop == 0)
								check_other_workers = 1;
							if(iter%1000 == 1) printf("pe %d, check_stop %d, iter %d, check_others %d\n", q.my_pe, check_stop, iter, check_other_workers);
						}
					}
					else if( (threadIdx.x>>5) == 0) { iter = 0;	if(threadIdx.x == 0) *(q.stop) = 0; }
					__syncthreads();

					if(check_other_workers == 1)
					{
						if(threadIdx.x < q.num_local_queues+q.n_pes-1)
						{
							q.workers_stop[threadIdx.x] = ends_record[threadIdx.x];
							__threadfence();
						}
						__syncthreads();

						//for(int tbid = threadIdx.x; tbid < gridDim.x-1; tbid+=blockDim.x)
						//	*((int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1)+tbid+1) = -1;
						// for simplicity, gridDim.x < blockDim.x
						if(threadIdx.x < gridDim.x-1) {
							*((int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1)+threadIdx.x+1) = -1;
							int workers_aknlg = ((volatile int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1))[threadIdx.x+1];
							while( workers_aknlg == -1) {
								workers_aknlg = ((volatile int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1))[threadIdx.x+1];
							}
							atomicAdd(&check_other_workers, int(workers_aknlg == 1));
						}
						__syncthreads();

						if(check_other_workers == gridDim.x) {
							if(threadIdx.x == 0) {
								*(q.stop) = 1;
								printf("pe %d, set stop to 1\n", q.my_pe);
								check_stop = 1;
							}
							__syncthreads();
						}
						else {
							if(threadIdx.x == 0) *(q.stop) = 0;
						}
					}

					if(q.my_pe == 0 && check_stop == 1)
					{
						if(checkStop_master_CTA(q, ends_record+q.num_local_queues+q.n_pes-1,
								 ends_record+q.num_local_queues+q.n_pes-1+q.n_pes*q.n_pes, q.check_temp))
							break;
					}
				}
				//local workers
				else 
				{
					if(threadIdx.x == 0) {
						ends_record[0] = ((volatile int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1))[blockIdx.x];
					}
					__syncthreads();
					if((int)(ends_record[0]) == -1)
					{
						COUNTER_T end_alloc = 0;
						if(threadIdx.x < q.num_local_queues)
							end_alloc = *((volatile COUNTER_T *)q.end_alloc+threadIdx.x*PADDING_SIZE);
						else if (threadIdx.x < q.num_local_queues+q.n_pes-1)
							end_alloc = *((volatile COUNTER_T *)q.end+threadIdx.x*PADDING_SIZE);

						bool ifSame = false;
						if(threadIdx.x < q.num_local_queues+q.n_pes-1)
						{
							if(*((volatile COUNTER_T *)(q.workers_stop)+threadIdx.x) == end_alloc && (end[threadIdx.x] == end_alloc))
								ifSame = true;
						}
						if( (threadIdx.x>>5) == 0)
						{
                        	if(__ballot_sync(0xffffffff, ifSame) == (0xffffffff>>(32-(q.num_local_queues+q.n_pes-1))))
							{
								if(threadIdx.x == 0)
									((int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1))[blockIdx.x] = 1;
							}
							else
							{
								if(threadIdx.x == 0)
									((int *)(((COUNTER_T *)q.workers_stop)+q.num_local_queues+q.n_pes-1))[blockIdx.x] = 0;
							}
						}
					}
				}
			}
        }
        while(true);
    }

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
	__device__ bool checkStop_master_CTA(dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> &wl,
		COUNTER_T *pe_send_array, COUNTER_T *pe_recv_array, COUNTER_T *check_temp) {
		__shared__ int exit;
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
		if(threadIdx.x == 0)  exit = 0;
		__syncthreads();
    	for(int i = threadIdx.x; i <wl.n_pes*wl.n_pes; i+=blockDim.x)
    	{
    	    pe_send_array[i] = 0;
    	    pe_recv_array[i] = 0;
    	}

    	if(threadIdx.x < wl.n_pes) {
    	    //int peer_stop = nvshmem_int_atomic_fetch((int *)wl.stop, threadIdx.x);
			int peer_stop = nvshmem_int_g((int *)wl.stop, threadIdx.x);
    	    printf("pe %d, bid %d, first check peer %d, stop %d\n", wl.my_pe, blockIdx.x, threadIdx.x, peer_stop);
			if(peer_stop != 1)
				exit = 1;
    	}
		__syncthreads();

		if(exit == 1)
    	    return false;

    	if(threadIdx.x < wl.n_pes && threadIdx.x != wl.my_pe) {
    	    nvshmem_getmem(check_temp+threadIdx.x*7*PADDING_SIZE*wl.total_num_queues, (COUNTER_T *)wl.start, sizeof(COUNTER_T)*7*PADDING_SIZE*wl.total_num_queues, threadIdx.x);
    	}
		__syncthreads();

        for(int idx = threadIdx.x; idx < 2*wl.n_pes*wl.n_pes; idx +=blockDim.x) {
            int pe = idx/(2*wl.n_pes);
            // recv = 0 send = 1
            int recvOrsend = (idx%(2*wl.n_pes))/(wl.n_pes);
            int peer_offset = (idx%(2*wl.n_pes))%(wl.n_pes);
			COUNTER_T *temp_write_share_ptr = (recvOrsend == 0)? pe_recv_array: pe_send_array;
			volatile COUNTER_T *temp_read_ptr = (pe == wl.my_pe)? wl.start: check_temp;

			int offset = (pe!=wl.my_pe)*pe*7*PADDING_SIZE*wl.total_num_queues;
			offset = offset + (pe != peer_offset)*wl.num_local_queues*PADDING_SIZE + recvOrsend*2*wl.total_num_queues*PADDING_SIZE;
			offset = offset + recvOrsend*(pe!=peer_offset)*(wl.n_pes-1)*PADDING_SIZE;
			offset = offset + (peer_offset < pe)*peer_offset*PADDING_SIZE;
			offset = offset + (peer_offset > pe)*(peer_offset-1)*PADDING_SIZE;

			temp_write_share_ptr[pe*wl.n_pes+peer_offset] = *(temp_read_ptr+offset);	

			//if(recvOrsend == 0)
   	     	//	printf("pe %d recevi from %d: %d\n", pe, peer_offset, pe_recv_array[pe*wl.n_pes+peer_offset]);
   	     	//else if(recvOrsend == 1)
			//	printf("pe %d send to %d: %d\n", pe, peer_offset, pe_send_array[pe*wl.n_pes+peer_offset]);

        }
    	__syncthreads();

   	 	for(int idx = threadIdx.x; idx < wl.n_pes*wl.n_pes; idx +=blockDim.x) {
   	 	    int send_pe = idx/wl.n_pes;
   	 	    int recv_pe = idx%wl.n_pes;
   	 	    if(pe_recv_array[recv_pe*wl.n_pes+send_pe] != pe_send_array[send_pe*wl.n_pes+recv_pe]) {
   	 	        printf("send_pe %d, send %d, recv_pe %d, recv %d, not equal\n", send_pe, pe_send_array[send_pe*wl.n_pes+recv_pe], recv_pe, pe_recv_array[recv_pe*wl.n_pes+send_pe]);
   	 	        exit = 1;
   	 	        break;
   	 	    }
   	 	}

		__syncthreads();
    	if(exit == 1)
    	    return false;
    	
    	if(threadIdx.x < wl.n_pes) {
			int ret = nvshmem_int_atomic_fetch((int *)wl.stop, threadIdx.x);
			//int ret = nvshmem_int_atomic_compare_swap((int *)wl.stop, 1, 2, threadIdx.x);
    	    if(ret != 1) {
    	        printf("peer %d, stop %d\n", threadIdx.x, ret);
    	        exit = 1;
    	    }
    	}
    	__syncthreads();
    	if(exit == 1)
    	    return false;

	    if(threadIdx.x < wl.n_pes) {
        	int ret = nvshmem_int_atomic_compare_swap((int *)wl.stop, 1, 3, threadIdx.x);
        	//printf("update peer %d stop to 3, ret %d\n", threadIdx.x, ret);
        	assert(ret == 1);
    	}	

		return true;
	}

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
	__global__ void checkStop_master(dev::DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> wl)
	{
		extern __shared__ COUNTER_T sharespace[];
		if(checkStop_master_CTA(wl, sharespace, sharespace+wl.n_pes*wl.n_pes, wl.check_temp)) {
			if(threadIdx.x == 0)
			printf("Detect Termination\n");
		}
	}

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
	void DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::
	checkStop(cudaStream_t &stream, int *stop_flag)
	{
		int shared_mem = sizeof(COUNTER_T)*(n_pes*n_pes*2);
		checkStop_master<<<1, align_up_yc(n_pes, 32), shared_mem, stream>>>(this->deviceObject());
		CUDA_CHECK(cudaMemcpyAsync(stop_flag, stop, sizeof(int), cudaMemcpyDeviceToHost, stream));
	}

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
    template<int FETCH_SIZE, typename Functor, typename... Args>
    void DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::
	launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
    {
		int shared_mem = (num_local_queues+n_pes-1)*3*sizeof(COUNTER_T)+shareMem_size;
        std::cout << "Launch CTA numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shared_mem << std::endl;
        _launchCTA_minIter<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE, FETCH_SIZE>
		<<<numBlock, numThread, shared_mem, stream>>>(this->deviceObject(), f, arg...);
    }

	template<typename RECV_T, typename LOCAL_T, typename COUNTER_T, int PADDING_SIZE>
    template<int FETCH_SIZE, typename Functor, typename... Args>
    void DistributedQueue<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>::
	launchCTA(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
    {
		// global master + bid 1 need (2+2*n_pes*n_pes)*sizeof(COUNTER_T),
        // local master + bid 0 need (num_local_queues+n_pes-1)*sizeof(COUNTER_T)+gridDim.x*sizeof(int)	
		int shared_mem = (num_local_queues+n_pes-1)*3*sizeof(COUNTER_T)+shareMem_size;
		//shared_mem = shared_mem + max(sizeof(COUNTER_T)*(n_pes*n_pes*2+2), sizeof(COUNTER_T)*(num_local_queues+n_pes-1)+numBlock*sizeof(int));
		shared_mem = shared_mem + sizeof(COUNTER_T)*(n_pes*n_pes*2)+sizeof(COUNTER_T)*(num_local_queues+n_pes-1);
        std::cout << "Launch CTA numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shared_mem << std::endl;
        _launchCTA<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE, FETCH_SIZE>
		<<<numBlock, numThread, shared_mem, stream>>>(this->deviceObject(), f, arg...);
    }
	} //MAXCOUNT
}// end Atos
#endif
