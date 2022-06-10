#ifndef QUEUE
#define QUEUE

#include <inttypes.h>
#include <limits>
#include <assert.h>

#include "../util/error_util.cuh"
#include "../util/util.cuh"

#include <cub/cub.cuh>

template<typename T>
__device__ T reductionMin_warp(T * array, uint32_t size, T init_value = std::numeric_limits<T>::max())
{
    T minValue = init_value; 
    T loadValue = init_value;
    for(uint32_t i=LANE_; i<size; i=i+32)
    {
        loadValue = array[i];
        unsigned mask = __activemask();
        for(int offset=16; offset>0; offset/=2)
        {   
            T temp = __shfl_down_sync(mask, loadValue, offset);
            __syncwarp(mask);
            if(LANE_+offset < __popc(mask))
                loadValue = min(loadValue, temp);
            __syncwarp(mask);
        }
        __syncwarp(mask);
        minValue = min(minValue, loadValue);
    }
    __syncwarp();
    minValue = __shfl_sync(0xffffffff, minValue, 0);
    return minValue;
}

namespace MaxCountQueue {
    template <class To, class From, class Res = typename std::enable_if<
    (sizeof(To) == sizeof(From)),
    //(alignof(To) >= alignof(From)) &&
    //(alignof(To)%alignof(From) == 0) &&
    //std::is_trivially_copyable<From>::value &&
    //std::is_trivially_copyable<To>::value,
    To>::type>
    __device__ Res& bit_cast(From& src) noexcept {
       return *reinterpret_cast<To*>(&src);
    }

    //enum { uint32_t, uint64_t } unpack_type;
    template <typename U32_T, std::enable_if_t<sizeof(U32_T) == 4, bool> = true>
    __forceinline__ __device__ void nvshmem_p_wraper(U32_T *dest, U32_T data, int peer)
    {
        uint32_t item = bit_cast<uint32_t>(data);
        nvshmem_uint32_p((uint32_t *)dest, item, peer);
    }
    template <typename U64_T, std::enable_if_t<sizeof(U64_T) == 8, bool> = true>
    __forceinline__ __device__ void nvshmem_p_wraper(U64_T *dest, U64_T data, int peer)
    {
        uint64_t item = bit_cast<uint64_t>(data);
        nvshmem_uint64_p((uint64_t *)dest, item, peer);
    }

    namespace dev {
        template<typename T, typename COUNTER_T>
        struct Queue {
            const int my_pe;
            const int n_pes;

            T *queue;
            T *recv_queues;
            COUNTER_T capacity;
            COUNTER_T recv_capacity;
            volatile COUNTER_T *start, *start_alloc, *end, *end_alloc, *end_max, *end_count;
            volatile int *stop;

            COUNTER_T * send_remote_alloc_end;
            COUNTER_T * sender_write_remote_end;
            volatile COUNTER_T * recv_read_local_end;
            COUNTER_T * recv_pop_alloc_start;
            uint32_t min_iter;
            int num_queues = 1;
            int queue_id = 0;
            COUNTER_T *execute = NULL;
            void *reserve = NULL;

            Queue() {}
            Queue(int _my_pe, int _n_pes, T *q, T * r_q, COUNTER_T cap, COUNTER_T r_cap, COUNTER_T *s, COUNTER_T *s_alloc, COUNTER_T *e, COUNTER_T *e_alloc, COUNTER_T *e_max, 
                COUNTER_T *e_count, int * _stop, COUNTER_T * send_alloc_end, COUNTER_T * sender_write, COUNTER_T * recv_read, COUNTER_T * recv_pop_alloc, uint32_t iter, int num_q, int q_id, COUNTER_T *exe, void *res):
                my_pe(_my_pe), n_pes(_n_pes), queue(q), recv_queues(r_q), capacity(cap), recv_capacity(r_cap), send_remote_alloc_end(send_alloc_end), sender_write_remote_end(sender_write), recv_pop_alloc_start(recv_pop_alloc),
                min_iter(iter), num_queues(num_q), queue_id(q_id), execute(exe), reserve(res)
                {
                    start = (volatile COUNTER_T *)s;
                    start_alloc = (volatile COUNTER_T *)s_alloc;
                    end = (volatile COUNTER_T *)e;
                    end_alloc = (volatile COUNTER_T *)e_alloc;
                    end_max = (volatile COUNTER_T *)e_max;
                    end_count = (volatile COUNTER_T *)e_count;
                    stop = (volatile int *)_stop;

                    recv_read_local_end = (volatile COUNTER_T *)recv_read;
                }

            __device__ COUNTER_T get_capacity() {return capacity;}
            __forceinline__ __device__ T get_item(COUNTER_T index) { 
                if(index >= capacity) asm("trap;");
                return (queue)[index]; 
            }
            __forceinline__ __device__ T get_item_volatile(COUNTER_T index) { 
                if(index >= capacity) asm("trap;");
                return ((volatile T *)queue)[index]; 
            }
            __forceinline__ __device__ void push_warp(const T& item) const {
                unsigned mask = __activemask();
                uint32_t total = __popc(mask);
                unsigned int rank = __popc(mask & lanemask_lt());
                int leader = __ffs(mask)-1;
                COUNTER_T alloc;
                if(rank == 0)
                    alloc = atomicAdd((COUNTER_T *)end_alloc, total);
                alloc = __shfl_sync(mask, alloc, leader);
                if(alloc+total >= capacity)
                    asm("trap;");

                queue[alloc+rank] = item;
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
            __forceinline__ __device__ void push_cta(bool ifpush, const T& item) const {
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
                    if(res+total_cta >= capacity) 
                        asm("trap;");
                    //printf("queue is full res %d, total_cta %d\n", res, total_cta);
                }
            
                //synchronize all threads of cta, make sure res value is ready
                __syncthreads();
                
                if(ifpush)
                {
                    if(res==0xffffffff);
                        asm("trap;");
                    queue[res+alloc+rank] = item;
                    __threadfence();
                }
            
                __syncthreads();
                // first thread update end_max, end_count and maybe end
                if(threadIdx.x == 0 && total_cta)
                {
                    if(res==0xffffffff);
                        asm("trap;");
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)end_max, res+total_cta);
                    if(maxNow < res+total_cta) maxNow = res+total_cta;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)end_count, total_cta)+total_cta;
                    if(maxNow == validC)
                        atomicMax((COUNTER_T *)end, validC);
                } 
            }
            __forceinline__ __device__ void update_end() const {
                if(LANE_ == 0) {
                    COUNTER_T maxNow = *(end_count);
                    if(*(end)!= maxNow && *(end_max) == maxNow)
                    {
                        atomicMax((COUNTER_T *)end, maxNow);
                    }
                }
            }
            __forceinline__ __device__ void update_end_execute(COUNTER_T init_value = std::numeric_limits<COUNTER_T>::max()) const {
                __shared__ COUNTER_T warp_min[32];
                if(LANE_ == 0)
                    warp_min[threadIdx.x>>5] = init_value;
                __syncwarp();
                T loadValue = init_value;
                for(int idx = threadIdx.x; idx<gridDim.x; idx+=blockDim.x)
                {
                    loadValue = execute[idx];
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
                if(threadIdx.x == 0)
                    atomicMax((COUNTER_T *)end, loadValue);
                __syncthreads();
            }
            __forceinline__ __device__ COUNTER_T grab_one() const {
                unsigned mask = __activemask();
                uint32_t total = __popc(mask);
                unsigned int rank = __popc(mask & lanemask_lt());
                int leader = __ffs(mask)-1;
                COUNTER_T alloc;
                if(rank == 0)
                        alloc = atomicAdd((COUNTER_T*)start_alloc, total);
                __syncwarp(mask);
                alloc = __shfl_sync(mask, alloc, leader);
                return alloc+rank;
            }

            __forceinline__ __device__ COUNTER_T grab_thread(const uint32_t &total) const {
                if(total == 0) return COUNTER_T(MAX_SZ);
                return atomicAdd((COUNTER_T *)start_alloc, total); 
            }

            __forceinline__ __device__ COUNTER_T grab_warp(const uint32_t& total) const {
                if(total==0) return COUNTER_T(MAX_SZ);
                COUNTER_T index;
                if(LANE_ == 0 ) {
                    index = atomicAdd((COUNTER_T*)start_alloc, total); 
                    if(index + total >= capacity)
                        asm("trap;");
                }
                __syncwarp();
                index = __shfl_sync(0xffffffff, index, 0);
                return index;   
            }
            __forceinline__ __device__ COUNTER_T grab_cta(const uint32_t &total) const {
                if(total == 0) return COUNTER_T(MAX_SZ);
                __shared__ COUNTER_T index;
                if(threadIdx.x ==0)
                {
                    index = atomicAdd((COUNTER_T *)start_alloc, total);
                    if(index + total >= capacity)
                        asm("trap;");
                }
                __syncthreads();
                return index;
            }
        };

        template<typename T, typename COUNTER_T, int PADDING_SIZE=64>
        struct Queues {
            int my_pe;
            int n_pes;

            T *queue;
            T * recv_queues;
            COUNTER_T capacity;
            COUNTER_T recv_capacity;
            volatile COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
            volatile int * stop;

            COUNTER_T * send_remote_alloc_end;
            COUNTER_T * sender_write_remote_end;
            COUNTER_T * recv_read_local_end;
            COUNTER_T * recv_pop_alloc_start;

            uint32_t min_iter;
            uint32_t num_queues;
            COUNTER_T *execute;

            Queues() {}
            Queues(int _my_pe, int _n_pes, T *q, T * r_q, COUNTER_T cap, COUNTER_T r_cap, COUNTER_T *s, COUNTER_T *s_alloc, COUNTER_T *e, COUNTER_T *e_alloc,
            COUNTER_T *e_max, COUNTER_T *e_count, int *_stop, COUNTER_T * send_alloc_end, COUNTER_T *sender_write, COUNTER_T * recv_read, COUNTER_T * recv_pop,
            uint32_t iter, uint32_t num_q, COUNTER_T *exe):
            my_pe(_my_pe), n_pes(_n_pes), queue(q), recv_queues(r_q), capacity(cap), recv_capacity(r_cap), send_remote_alloc_end(send_alloc_end),
            sender_write_remote_end(sender_write), recv_read_local_end(recv_read), recv_pop_alloc_start(recv_pop),
            min_iter(iter), num_queues(num_q), execute(exe)
            {
                start = (volatile COUNTER_T *)s;
                start_alloc = (volatile COUNTER_T *)s_alloc;
                end = (volatile COUNTER_T *)e;
                end_alloc = (volatile COUNTER_T *)e_alloc;
                end_max = (volatile COUNTER_T *)e_max;
                end_count = (volatile COUNTER_T *)e_count;
                stop = (volatile int *)_stop;
            }            

            __device__ T get_item_volatile(COUNTER_T idx, int q_id) {
                if(idx >= capacity)
                    asm("trap;");
                return ((volatile T *)queue)[idx];
            }
            __device__ void push_warp(T item)
            {
                unsigned mask = __activemask();
                uint32_t total = __popc(mask);
                unsigned int rank = __popc(mask & lanemask_lt());
                int leader = __ffs(mask)-1;
                uint64_t q_id = WARPID%num_queues;
                COUNTER_T alloc;
                if(rank == 0){
                    alloc = atomicAdd((COUNTER_T *)(end_alloc+q_id*PADDING_SIZE), total);
                }
                alloc = __shfl_sync(mask, alloc, leader);
                assert(alloc + total <= capacity);
                queue[q_id*capacity + (alloc+rank)] = item;
                __threadfence();

                if(rank == 0) {
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+q_id*PADDING_SIZE), (alloc+total));
                    //__threadfence();
                    if(maxNow < alloc+total) maxNow = alloc+total;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+q_id*PADDING_SIZE), total)+total;
                    if(validC == maxNow)
                    {
                        __threadfence();
                        atomicMax((COUNTER_T *)(end+q_id*PADDING_SIZE), maxNow);
                    }
                }
                __syncwarp(mask);
            }

            __forceinline__ __device__ void push_cta(bool ifpush, const T& item) const
            {
                __shared__ COUNTER_T res;
                __shared__ int total_cta; 
                //init res and total_cta to 0
                if(threadIdx.x == 0)
                {
                    res = 0xffffffff; total_cta = 0;
                }
    
                __syncthreads();
    
                //each warp counts number of item need to be pushed and atomicAdd to the total_cta
                unsigned mask = __ballot_sync(0xffffffff, ifpush);    // ballot makes different in soc_liverJournal BFS
                uint32_t rank = __popc(mask & lanemask_lt());
                uint32_t alloc;
                if(ifpush && rank == 0) 
                    alloc = atomicAdd(&total_cta, __popc(mask));
                alloc = __shfl_sync(0xffffffff, alloc, __ffs(mask)-1);
    
                //synchronize all threads of cta, make sure they add to total_cta
                __syncthreads();
    
                //first thread will atomicAdd to end_alloc to reserve a room for push
                uint64_t q_id = blockIdx.x%num_queues;
                if(threadIdx.x == 0 && total_cta)
                {
                    res = atomicAdd((COUNTER_T *)(end_alloc+q_id*PADDING_SIZE), total_cta);
                    //if(res+total_cta >= capacity) printf("queue is full res %d, total_cta %d\n", res, total_cta);
                    assert(res+total_cta < capacity);
                }
    
                //synchronize all threads of cta, make sure res value is ready
                __syncthreads();

                if(ifpush)
                {
                    assert(res!=0xffffffff);
                    queue[q_id*capacity+res+alloc+rank] = item;
                    __threadfence();
                }
    
                __syncthreads();
                // first thread update end_max, end_count and maybe end
                if(threadIdx.x == 0 && total_cta)
                {
                    assert(res!=0xffffffff);
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+q_id*PADDING_SIZE), res+total_cta);
                    if(maxNow < res+total_cta) maxNow = res+total_cta;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+q_id*PADDING_SIZE), total_cta)+total_cta;
                    if(maxNow == validC)
                        atomicMax((COUNTER_T *)(end+q_id*PADDING_SIZE), validC);
                }
    
            }
        };
    } // dev
    template<typename T, typename COUNTER_T, int PADDING_SIZE=1>
    struct Queue
    {
        int my_pe;
        int n_pes;

        T * queue;
        T * recv_queues;
        COUNTER_T capacity;
        COUNTER_T recv_capacity;
        COUNTER_T *counters; // 10 counters, last one for reservation
        int num_counters = 10;
        COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
        int *stop;

        COUNTER_T * send_remote_alloc_end;
        COUNTER_T * sender_write_remote_end;
        COUNTER_T * recv_read_local_end;
        COUNTER_T * recv_pop_alloc_start;
        uint32_t min_iter;
        int num_queues = 1;
        int queue_id = 0;
        COUNTER_T *execute=NULL;
        void *reserve=NULL;

        Queue() {}

        ~Queue() { release(); }

        __host__ void init(COUNTER_T _capacity, COUNTER_T _r_cap, int _my_pe, int _n_pes, uint32_t _min_iter=800)
        {
            my_pe = _my_pe;
            n_pes = _n_pes;
            capacity = _capacity; 
            recv_capacity = _r_cap;

            CUDA_CHECK(cudaMalloc(&queue, sizeof(T)*capacity)); 
            CUDA_CHECK(cudaMemset((void *)queue, -1, sizeof(T)*capacity));

            recv_queues = (T *)nvshmem_malloc(sizeof(T)*recv_capacity*(n_pes-1));
            CUDA_CHECK(cudaMemset((void *)recv_queues, -1, sizeof(T)*recv_capacity*(n_pes-1)));

            CUDA_CHECK(cudaMalloc(&counters, sizeof(COUNTER_T)*num_counters*PADDING_SIZE));
            CUDA_CHECK(cudaMemset((void *)counters, 0, sizeof(COUNTER_T)*num_counters*PADDING_SIZE));
            start = (counters);
            start_alloc = (counters+1*PADDING_SIZE);
            end_alloc = (counters+2*PADDING_SIZE);
            end = (counters+3*PADDING_SIZE);
            end_max = (counters+4*PADDING_SIZE);
            end_count = (counters+5*PADDING_SIZE);
            stop = (int *)(counters+6*PADDING_SIZE);

            CUDA_CHECK(cudaMalloc(&send_remote_alloc_end, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMemset(send_remote_alloc_end, 0, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMalloc(&recv_pop_alloc_start, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMemset(recv_pop_alloc_start, 0, sizeof(COUNTER_T)*(n_pes-1)));

            sender_write_remote_end = (COUNTER_T *)nvshmem_malloc(sizeof(COUNTER_T)*(n_pes-1));
            CUDA_CHECK(cudaMset(sender_write_remote_end, 0, sizeof(COUNTER_T)*(n_pes-1)));
            recv_read_local_end = (COUNTER_T *)nvshmem_ptr(sender_write_remote_end, my_pe);

    	    min_iter = _min_iter;
        }

        __host__ void init(int _my_pe, int _n_pes, COUNTER_T _capacity, COUNTER_T _r_cap, T *_queue, T * r_q, COUNTER_T *_start, COUNTER_T *_end, COUNTER_T *_start_alloc,
                COUNTER_T* _end_alloc, COUNTER_T* _end_max, COUNTER_T* _end_count, int *_stop, COUNTER_T *send_alloc_end, COUNTER_T *sender_write, COUNTER_T *recv_read,
                COUNTER_T *recv_pop_alloc, int _num_queues, int _q_id, uint32_t _min_iter=800)
        {
            my_pe = _my_pe;
            n_pes = _n_pes;
            capacity = _capacity; 
            recv_capacity = _r_cap;
            queue = _queue;
            recv_queues = r_q;
            start = _start;
            end = _end;
            start_alloc = _start_alloc;
            end_alloc = _end_alloc;
            end_max = _end_max;
            end_count = _end_count;
            stop = _stop;
            send_remote_alloc_end = send_alloc_end;
            sender_write_remote_end = sender_write;
            recv_read_local_end = recv_read;
            recv_pop_alloc_start = recv_pop_alloc;
            min_iter = _min_iter;
            num_queues = _num_queues;
            queue_id = _q_id;
        }

        __host__ void reset(cudaStream_t stream=0)
        {
            CUDA_CHECK(cudaMemsetAsync((void *)queue, -1, sizeof(T)*capacity, stream));
            CUDA_CHECK(cudaMemsetAsync((void *)recv_queues, -1, sizeof(T)*recv_capacity*(n_pes-1), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)start, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)start_alloc, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)end, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)end_alloc, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)end_max, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)end_count, 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)(stop), 0, sizeof(COUNTER_T), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)send_remote_alloc_end, 0, sizeof(COUNTER_T)*(n_pes-1), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)sender_write_remote_end, 0, sizeof(COUNTER_T)*(n_pes-1), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)recv_pop_alloc_start, 0, sizeof(COUNTER_T)*(n_pes-1), stream));
        }

        dev::Queue<T, COUNTER_T> DeviceObject() const {
            return dev::Queue<T, COUNTER_T>(my_pe, n_pes, queue, recv_queues, capacity, recv_capacity, start, start_alloc, end, end_alloc, end_max, end_count,
            stop, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end, recv_pop_alloc_start, min_iter, num_queues, queue_id, execute, reserve);
        }

        __host__ COUNTER_T get_capacity() {return capacity;}


        template<typename Functor, typename... Args>
        __host__ void launchWarpPer32Items_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

        template<int FETCH_SIZE, typename Functor, typename... Args>
        __host__ void launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

        template<int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args>
        __host__ void launchCTA_minIter_2Func(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor1 f1, Functor2 f2, Args... arg);

        template<int FETCH_SIZE, typename Functor, typename... Args>
        __host__ void launchCTA_minIter_RT(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

        __host__ void release()
        {
            if(queue!=NULL)
            cudaFree(queue);
            if(counters!=NULL)
            cudaFree((void *)counters);
    	    if(execute!=NULL)
            cudaFree(execute);
            if(reserve!=NULL)
            cudaFree(reserve);
        }

        __host__ void print() const
        {
            COUNTER_T check[6];
            CUDA_CHECK(cudaMemcpy(check, start, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check+1, start_alloc, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check+2, end, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check+3, end_alloc, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check+4, end_max, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check+5, end_count, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
            if(check[2]== check[3] && check[3] == check[4] && check[4] == check[5] && check[2] != check[0])
                printf("Queue %d, capacity: %zd, end: %zd\tstart: %zd\n", queue_id, size_t(capacity), size_t(check[2]),
                        size_t(check[0]));
            else if(check[2] == check[3] && check[3] == check[4] && check[4] == check[5] && check[2] == check[0])
                printf("Queue %d, capacity: %zd, end=start: %zd\n", queue_id, size_t(capacity), size_t(check[2]));
            else printf("Queue %d, capacity: %zd\nend: %zd\nend_alloc: %zd\nend_max: %zd\nend_count: %zd\nstart: %zd\nstart_alloc: %zd\n",
                    queue_id, size_t(capacity), size_t(check[2]), size_t(check[3]), size_t(check[4]), size_t(check[5]), size_t(check[0]), size_t(check[1]));
            //printf("Queue %d, queue %p, end_alloc %p, end %p, end_max %p, end_count %p, start_alloc %p, start %p\n", queue_id, (void *)queue, (void *)end_alloc,
            //        (void *)end, (void *)end_max, (void *)end_count, (void *)start_alloc, (void *)start);
        }
    };

    template<typename Functor, typename T, typename COUNTER_T, typename... Args>
    __forceinline__ __device__ void func(Functor F, T t, Args... args) // recursive variadic function
    {
       func(F, args...);
    }

    template<typename T, typename COUNTER_T, typename Functor, typename... Args>
    __launch_bounds__(1024,1)
    __global__ void _launchWarpPer32Items_minIter(dev::Queue<T, COUNTER_T> q, Functor F, Args... args)
    {

        COUNTER_T index = q.grab_warp(32);
        __syncwarp(0xffffffff);
        COUNTER_T e = *(q.end);
        uint32_t iter = 0;
        do {
            while(index < e)
            {
                unsigned activeMask = __ballot_sync(0xffffffff, (index+LANE_)<min(e, align_up_yc(index+1, 32)));
    	        activeMask = __popc(activeMask);
                T task;
                if(index+LANE_ < min(e,align_up_yc(index+1, 32)))
    	        {
                    task = q.get_item_volatile(index+LANE_);
    		        assert(task!= -1);
    	        }
                __syncwarp(0xffffffff);
                for(int i=0; i<activeMask; i++)
                {
                    T cur_task = __shfl_sync(0xffffffff, task, i);
                    __syncwarp(0xffffffff);
                    func(F, cur_task, args...);
                    __syncwarp(0xffffffff);
                }

                if(activeMask+ index == align_up_yc(index+1, 32))
                {
                    index = q.grab_warp(32);
                } 
                else
                {
                    e = *(q.end);
                    index = index+activeMask;
                }
                __syncwarp(0xffffffff);
            } //while
            e = *(q.end);
            if(index >= e)
            {
                if(*(q.stop) == blockDim.x*gridDim.x/32*q.num_queues)
                    break;
                iter++;
                if(iter == q.min_iter && LANE_ == 0)
                    atomicAdd((COUNTER_T *)(q.stop), 1);
                q.update_end();
                __syncwarp(0xffffffff);
    	        __nanosleep(60);
            }
            __syncwarp(0xffffffff);
        }
        while(true);

        if(LANE_ == 0) {
            q.execute[WARPID] = index;
        }
    }

    template<typename T, typename COUNTER_T, int FETCH_SIZE, typename Functor, typename... Args >
    __launch_bounds__(1024,1)
    __global__ void _launchCTA_minIter(dev::Queue<T, COUNTER_T> q, Functor F, Args... args)
    {
        __shared__ COUNTER_T end;
        //__shared__ T nodes[FETCH_SIZE];
        T node = T(-1);
        #ifdef TIMING
            long long int start_time, cpt_start;
            long long int cpt_time=0;
            if(threadIdx.x==0)
                start_time = clock64();
        #endif

        int fs = FETCH_SIZE;
        int iter = 0;
        COUNTER_T index = q.grab_cta(FETCH_SIZE);
        if(threadIdx.x == 0) {
    	    end = *(q.end);
    	    q.execute[blockIdx.x] = index;
        }
        __syncthreads();

        do {
            while(index < end)
         	{
                fs = min(end-index, fs);
         	    if(threadIdx.x < fs)
         	    {
                    node = q.get_item_volatile(index+threadIdx.x);
                    assert(node!=-1);
         	    }
         	    __syncthreads();

                #ifdef TIMING
                    if(threadIdx.x == 0)
                        cpt_start = clock64();
                #endif

         	    func(F, node, fs, args...);
                __syncthreads();

                #ifdef TIMING
                    if(threadIdx.x == 0)
                        cpt_time = cpt_time + (clock64()-cpt_start);
                #endif

         	    if(fs+index == align_up_yc(index+1, FETCH_SIZE))
         	    {
         		    index = q.grab_cta(FETCH_SIZE);
         		    fs = FETCH_SIZE;
    		        if(threadIdx.x == 0)
    			        q.execute[blockIdx.x] = index;
                }
                else
                {
                    index = index + fs;
                    fs = align_up_yc(index, FETCH_SIZE) - index;
                    if(threadIdx.x == 0)
                    {
                        end = *(q.end);
         		        q.update_end();
    		            q.execute[blockIdx.x] = index;
         		    }   
    		        //q.update_end_execute();
                }
         	    __syncthreads();

            } //while
            __syncthreads();

         	if(threadIdx.x == 0)
            {
                end = *(q.end);
            }
         	__syncthreads();

            if(index >= end)
            {
                if(*(q.stop) == gridDim.x*q.num_queues)
                    break;
                __syncthreads();
                if(threadIdx.x == 0)
                {
                    iter++;
                    if(iter == q.min_iter)
                        atomicAdd((COUNTER_T *)(q.stop), 1);
                }
                q.update_end_execute();
                //if(threadIdx.x == 0)
                //    q.update_end();
                __nanosleep(100);
            }
         	__syncthreads();
        }
        while(true);
        if(threadIdx.x == 0) {
            q.execute[blockIdx.x] = index;
            #ifdef TIMING
                ((double *)q.reserve)[blockIdx.x*2] = double(cpt_time/clockrate);
                ((double *)q.reserve)[blockIdx.x*2+1] = (double(clock64()-start_time)/clockrate);
            #endif
        }
    }

    template<typename T, typename COUNTER_T, int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args >
    __launch_bounds__(1024,1)
    __global__ void _launchCTA_minIter_2Func(dev::Queue<T, COUNTER_T> q, Functor1 F1, Functor2 F2, Args... args)
    {
        __shared__ COUNTER_T end;
        T node = T(-1);
    
        int fs = FETCH_SIZE;
        int iter = 0;
        COUNTER_T index = q.grab_cta(FETCH_SIZE);
        if(threadIdx.x == 0) {
    	    end = *(q.end);
            //printf("end %p, nodes %p\n", &end, nodes);
        }
        __syncthreads();
    
        do {
            while(index < end)
         	{
                fs = min(end-index, fs);
         	    if(threadIdx.x < fs)
         	    {
         		    node = q.get_item_volatile(index+threadIdx.x);
         		    assert(node!=-1);
         	    }
         	    __syncthreads();
             
         	    func(F1, node, fs, args...);
         	    __syncthreads();
             
         	    if(fs+index == align_up_yc(index+1, FETCH_SIZE))
         	    {
         		    index = q.grab_cta(FETCH_SIZE);
         		    fs = FETCH_SIZE;
                }
                else
                {
                    index = index + fs;
                    fs = align_up_yc(index, FETCH_SIZE) - index;
                    if(threadIdx.x == 0)
                    {
                        end = *(q.end);
         		        q.update_end();
         		    }
                }
         	    __syncthreads();
            } //while

            __syncthreads();
        
         	if(threadIdx.x == 0)
            {
                end = *(q.end);
            }
         	__syncthreads();
        
            if(index >= end)
            {
                //if(*(q.stop)==gridDim.x*q.num_queues) {
                func(F2, args...);
                //}
                if(*(q.stop) == gridDim.x*q.num_queues)
                    break;
    	        __syncthreads();
                if(threadIdx.x == 0)
                {
                    iter++;
                    if(iter == q.min_iter)
                        atomicAdd((COUNTER_T *)(q.stop), 1);
                    q.update_end();
                }
              //  __nanosleep(60);
            }
         	__syncthreads();
        }
        while(true);
        if(threadIdx.x == 0)
        {
            q.execute[blockIdx.x] = index;
        }
    }

    template<typename T, typename COUNTER_T, int FETCH_SIZE, typename Functor, typename... Args >
    __launch_bounds__(1024,1)
    __global__ void _launchCTA_minIter_RT(dev::Queue<T, COUNTER_T> q, Functor F, Args... args)
    {
        // local worklist, remote receive queue
        extern __shared__ COUNTER_T shared_space[];
        COUNTER_T * end = (COUNTER_T *) shared_space;
        COUNTER_T * index = (COUNTER_T *)(shared_space) + q.n_pes;
        int * fs = (int *)((COUNTER_T*)(shared_space) + 2*q.n_pes);

        T node = T(-1);

        int iter = 0;
        if(threadIdx.x < q.n_pes)
            fs[threadIdx.x] = FETCH_SIZE;
        if(threadIdx.x == 0) {
    	    end[0] = *(q.end);
            index[0] = q.grab_thread(FETCH_SIZE);
    	    q.execute[blockIdx.x] = index[0];
        }
        else if(threadIdx.x < q.n_pes) {
            end[threadIdx.x] = *((volatile uint32_t *)(q.recv_read_local_end+threadIdx.x-1));
            index[threadIdx.x] = atomicAdd(q.recv_pop_alloc_start+threadIdx.x-1, FETCH_SIZE);
        }
        __syncthreads();
        
        do {
            for(int wl_id = q.n_pes-1; wl_id >= 0; wl_id--) {
                while(index[wl_id] < end[wl_id])
         	    {
         	        if(threadIdx.x < min(end[wl_id]-index[wl_id], fs[wl_id]))
         	        {
                        if(wl_id == 0) {
                            node = q.get_item_volatile(index[0]+threadIdx.x);
                            int loop1 = 0;
                            while(node == -1) {
                                assert(loop1 < 32);
                                node = q.get_item_volatile(index[0]+threadIdx.x);
                                loop1++;
                            }
                        }
                        else {
                            node = q.recv_queues[(wl_id-1)*q.recv_capacity+index[wl_id]+threadIdx.x];
                            int loop = 0;
                            while(node == -1) {
                                assert(loop < 1024);
                                node = *((volatile T *)(q.recv_queues+(wl_id-1)*q.recv_capacity+index[wl_id]+threadIdx.x));
                                loop++;
                            }
                        }
                        assert(node!=-1);
         	        }
         	        __syncthreads();

         	        func(F, node, min(end[wl_id]-index[wl_id], fs[wl_id]), args...);
                    __syncthreads();

                    if(threadIdx.x == 0) {
                        fs[wl_id] = min(end[wl_id]-index[wl_id], fs[wl_id]);
         	            if(fs[wl_id]+index[wl_id] == align_up_yc(index[wl_id]+1, FETCH_SIZE))
         	            {
                            if(wl_id == 0) {
         	    	            index[0] = q.grab_thread(FETCH_SIZE);
    			                q.execute[blockIdx.x] = index[0];
                            }
                            else 
                                index[wl_id] = atomicAdd(q.recv_pop_alloc_start+wl_id-1, FETCH_SIZE);
                            fs[wl_id] = FETCH_SIZE;
                        }
                        else
                        {
                            index[wl_id] = index[wl_id] + fs[wl_id];
                            fs[wl_id] = align_up_yc(index[wl_id], FETCH_SIZE) - index[wl_id];
                            if(wl_id == 0)
    		                    q.execute[blockIdx.x] = index[0];
                        }   
                    }
         	        __syncthreads();

                } //while
                __syncthreads();

         	    if(threadIdx.x == 0)
                {
                    end[0] = *(q.end);
                    COUNTER_T maxNow = *(q.end_max);
                    if(end[0] != maxNow && maxNow == *(q.end_count)){
                        end[0] = maxNow;
                        atomicMax((COUNTER_T *)(q.end), maxNow);
                    }
                }
                else if(threadIdx.x <  q.n_pes)
                    end[threadIdx.x] = *((volatile uint32_t *)(q.recv_read_local_end+threadIdx.x-1));

         	    __syncthreads();

            
                bool ifempty = false;
                if(LANE_ < q.n_pes)
                    ifempty = index[LANE_] >= end[LANE_];

                if(__ballot_sync(0xffffffff, ifempty) == (0xffffffff>>(32-q.n_pes)))
                {
                    if(*(q.stop) == gridDim.x*q.num_queues)
                        break;
                    if(threadIdx.x == 0) {
                        iter++;
                        if(iter == q.min_iter)
                            atomicAdd((int *)(q.stop), 1);
                    }

                    q.update_end_execute();
                }
            }
        }
        while(*(q.stop) < gridDim.x*q.num_queues);
        if(threadIdx.x == 0) {
            q.execute[blockIdx.x] = index[0];
            atomicMin((uint32_t *)(q.reserve), index[1]);
        }
        else if(threadIdx.x < q.n_pes-1)
            atomicMin(((uint32_t *)q.reserve)+threadIdx.x, index[1+threadIdx.x]);
    }

    template<typename T>
    __global__ void ReductionMin(T *array, uint32_t size, T *out)
    {
        T res = reductionMin_warp(array, size);
        if(LANE_ == 0)
            *out = res;
        //    printf("size %d, min %d\n", size, int(res));
    }

    template<typename T, typename COUNTER_T, int PADDING_SIZE>
    template<typename Functor, typename... Args>
    void Queue<T, COUNTER_T, PADDING_SIZE>::launchWarpPer32Items_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
    {
        std::cout << "Launch Warp numBlock "<< numBlock << " numThread "<< numThread<<" dynamic shared memory size(bytes) "<< shareMem_size << std::endl;
        _launchWarpPer32Items_minIter<<<numBlock, numThread, shareMem_size, stream>>>(this->DeviceObject(), f, arg...);
    }

    template<typename T, typename COUNTER_T, int PADDING_SIZE>
    template<int FETCH_SIZE, typename Functor, typename... Args>
    void Queue<T, COUNTER_T, PADDING_SIZE>::launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
    {
        std::cout << "Launch CTA numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shareMem_size << std::endl;
        _launchCTA_minIter<T, COUNTER_T, FETCH_SIZE><<<numBlock, numThread, shareMem_size, stream>>>(this->DeviceObject(), f, arg...);
    }

    template<typename T, typename COUNTER_T, int PADDING_SIZE>
    template<int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args>
    void Queue<T, COUNTER_T, PADDING_SIZE>::launchCTA_minIter_2Func(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor1 f1, Functor2 f2, Args... arg)
    {
        std::cout << "numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shareMem_size << std::endl;
        _launchCTA_minIter_2Func<T, COUNTER_T, FETCH_SIZE><<<numBlock, numThread, shareMem_size, stream>>>(this->DeviceObject(), f1, f2, arg...);
    }

    template<typename T, typename COUNTER_T, int PADDING_SIZE>
    template<int FETCH_SIZE, typename Functor, typename... Args>
    void Queue<T, COUNTER_T, PADDING_SIZE>::launchCTA_minIter_RT(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
    {
        int shared_mem = n_pes*3*sizeof(COUNTER_T)+shareMem_size;
        std::cout << "Yxinxin Launch CTA numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shared_mem << std::endl;
        _launchCTA_minIter_RT<T, COUNTER_T, FETCH_SIZE><<<numBlock, numThread, shared_mem, stream>>>(this->DeviceObject(), f, arg...);
    }

    template<typename T, typename COUNTER_T=uint32_t, int PADDING_SIZE=64>
    struct Queues
    {
        int my_pe;
        int n_pes;

        T * queue;
        T * recv_queues;
        COUNTER_T capacity;
        COUNTER_T recv_capacity;
        uint32_t num_queues;
        COUNTER_T *counters; // 10 counters, last one for reservation
        int num_counters = 7;
        COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
        int *stop;

        COUNTER_T *send_remote_alloc_end;
        COUNTER_T *sender_write_remote_end;
        COUNTER_T *recv_read_local_end;
        COUNTER_T *recv_pop_alloc_start;

        uint32_t min_iter;
        COUNTER_T *execute;
        MaxCountQueue::Queue<T, COUNTER_T> *worklist;
        cudaStream_t *streams;
        int warpSize_wl_pop=-1;

        Queues() {}
        ~Queues() {release();}

        __host__ void init(COUNTER_T _capacity, COUNTER_T _r_cap, int _my_pe, int _n_pes, uint32_t _num_q=1, uint32_t _min_iter=800)
        {
            my_pe = _my_pe;
            n_pes = _n_pes;
            capacity = _capacity; 
            recv_capacity = _r_cap;
            num_queues = _num_q;

            CUDA_CHECK(cudaMalloc(&queue, sizeof(T)*capacity*num_queues)); 
            CUDA_CHECK(cudaMemset((void *)queue, -1, sizeof(T)*capacity*num_queues));

            recv_queues = (T *)nvshmem_malloc(sizeof(T)*recv_capacity*(n_pes-1));
            CUDA_CHECK(cudaMemset((void *)recv_queues, -1, sizeof(T)*recv_capacity*(n_pes-1)))

            CUDA_CHECK(cudaMalloc(&counters, sizeof(COUNTER_T)*num_counters*num_queues*PADDING_SIZE));
            CUDA_CHECK(cudaMemset((void *)counters, 0, sizeof(COUNTER_T)*num_counters*num_queues*PADDING_SIZE));
            start = (counters);
            start_alloc = (counters+1*num_queues*PADDING_SIZE);
            end_alloc = (counters+2*num_queues*PADDING_SIZE);
            end = (counters+3*num_queues*PADDING_SIZE);
            end_max = (counters+4*num_queues*PADDING_SIZE);
            end_count = (counters+5*num_queues*PADDING_SIZE);
            stop = (int *)(counters+6*num_queues*PADDING_SIZE);

            CUDA_CHECK(cudaMalloc(&send_remote_alloc_end, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMemset(send_remote_alloc_end, 0, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMalloc(&recv_pop_alloc_start, sizeof(COUNTER_T)*(n_pes-1)));
            CUDA_CHECK(cudaMemset(recv_pop_alloc_start, 0, sizeof(COUNTER_T)*(n_pes-1)));

            sender_write_remote_end = (COUNTER_T *)nvshmem_malloc(sizeof(COUNTER_T)*(n_pes-1));
            CUDA_CHECK(cudaMemset(sender_write_remote_end, 0, sizeof(COUNTER_T)*(n_pes-1)));
            recv_read_local_end = (COUNTER_T *)nvshmem_ptr(sender_write_remote_end, my_pe);

	        min_iter = _min_iter;

            worklist = (MaxCountQueue::Queue<T, COUNTER_T> *)malloc(sizeof(MaxCountQueue::Queue<T, COUNTER_T>)*num_queues);
            streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*num_queues);

            printf("alloc recv_qiueues %p, recv_capacity %d, my_pe %d, send_remote_alloc_end %p, sender_write %p, recv_read %p, recv_pop %p\n", 
            recv_queues, recv_capacity, my_pe, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end, recv_pop_alloc_start);
            for(uint64_t i=0; i<num_queues; i++)
            {
                worklist[i].init(my_pe, n_pes, capacity, recv_capacity, queue+i*capacity, recv_queues, start+i*PADDING_SIZE, end+i*PADDING_SIZE, 
                    start_alloc+i*PADDING_SIZE, end_alloc+i*PADDING_SIZE, end_max+i*PADDING_SIZE, 
                    end_count+i*PADDING_SIZE, stop, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end, recv_pop_alloc_start,
                    num_queues, i, min_iter);
                CUDA_CHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
            }
        }

        __host__ void release()
        {
            if(queue != NULL)
                CUDA_CHECK(cudaFree(queue));
            if(counters != NULL)
                CUDA_CHECK(cudaFree((void *)counters));
            if(streams != NULL)
                free(streams);
            if(warpSize_wl_pop != -1)
                for(int i=0; i<num_queues; i++)
                    CUDA_CHECK(cudaFree(worklist[i].execute));
            if(worklist!= NULL)
                free(worklist);
        }

        __host__ void reset(cudaStream_t stream=0) const
        {
            for(int i=0; i<num_queues; i++)
            {
                worklist[i].reset(stream);
                if(warpSize_wl_pop!=-1)
    	            CUDA_CHECK(cudaMemsetAsync(worklist[i].execute, 0, sizeof(COUNTER_T)*warpSize_wl_pop, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        dev::Queues<T, COUNTER_T, PADDING_SIZE> DeviceObject() {
            return dev::Queues<T, COUNTER_T, PADDING_SIZE>(my_pe, n_pes, queue, recv_queues, capacity, recv_capacity, start, start_alloc, end, end_alloc, 
            end_max, end_count, stop, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end, recv_pop_alloc_start, min_iter, num_queues, execute);
        }

        __host__ COUNTER_T get_capacity() const {return capacity;}

        __host__ void launchWarpPer32Items_minIter_preLaunch(int numBlock, int numThread)
        {
            if(warpSize_wl_pop == -1)
            {
                warpSize_wl_pop = numBlock/num_queues*numThread/32;
                for(int i=0; i<num_queues; i++)
                {
                    CUDA_CHECK(cudaMallocManaged(&(worklist[i].execute), sizeof(COUNTER_T)*warpSize_wl_pop));
                    CUDA_CHECK(cudaMemset(worklist[i].execute, 0, sizeof(COUNTER_T)*warpSize_wl_pop));
                }
            }
        }

        __host__ void launchCTA_minIter_preLaunch(int numBlock, int numThread)
        {
            if(warpSize_wl_pop == -1)
            {
                warpSize_wl_pop = numBlock/num_queues;
                for(int i=0; i<num_queues; i++)
                {
                    CUDA_CHECK(cudaMallocManaged(&(worklist[i].execute), sizeof(COUNTER_T)*warpSize_wl_pop));
                    CUDA_CHECK(cudaMemset(worklist[i].execute, 0, sizeof(COUNTER_T)*warpSize_wl_pop));
                }
            }
        }

        template<typename Functor, typename... Args>
        __host__ void launchThreadPerItem_minIter(int numBlock, int numThread, int shared_mem, Functor f, Args... arg)
        {
            for(int i=0; i<num_queues; i++)
            {
                worklist[i].launchThreadPerItem_minIter(numBlock/num_queues, numThread, streams[i], shared_mem, f, arg...);
            }
        }

        template<typename Functor, typename... Args>
        __host__ void launchWarpPer32Items_minIter(int numBlock, int numThread, int shared_mem, Functor f, Args... arg)
        {
            for(int i=0; i<num_queues; i++)
            {
                worklist[i].launchWarpPer32Items_minIter(numBlock/num_queues, numThread, streams[i], shared_mem, f, arg...);
            }
        }

        template<int FETCH_SIZE, typename Functor, typename... Args>
        __host__ void launchCTA_minIter(int numBlock, int numThread, int shared_mem, Functor f, Args... arg)
        {
            for(int i=0; i<num_queues; i++)
            {
                worklist[i].template launchCTA_minIter<FETCH_SIZE>(numBlock/num_queues, numThread, streams[i], shared_mem, f, arg...);
            }
        }

        template<int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args>
        __host__ void launchCTA_minIter_2Func(int numBlock, int numThread, int shared_mem, Functor1 f1, Functor2 f2, Args... arg)
        {
            for(int i=0; i<num_queues; i++)
            {
                worklist[i].template launchCTA_minIter_2Func<FETCH_SIZE>(numBlock/num_queues, numThread, streams[i], shared_mem, f1, f2, arg...);
            }
        }

        template<int FETCH_SIZE, typename Functor, typename... Args>
        __host__ void launchCTA_minIter_RT(int numBlock, int numThread, int shared_mem, Functor f, Args... arg)
        {
            assert(num_queues == 1);
            worklist[0].template launchCTA_minIter_RT<FETCH_SIZE>(numBlock/num_queues, numThread, streams[0], shared_mem, f, arg...);
        }

        __host__ void sync_all_wl()
        {
            for(int i=0; i<num_queues; i++)
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        __host__ void print()
        {
            uint64_t total_proc = 0;
            for(int i=0; i<num_queues; i++)
            {
                std::cout << i<<"th queue:\n";
                //assert(warpSize_wl_pop>0);
                if(warpSize_wl_pop > 0) {
                    ReductionMin<<<1, 32>>>(worklist[i].execute, warpSize_wl_pop, (COUNTER_T *)(worklist[i].start));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                worklist[i].print();
                //total_proc = total_proc + *(worklist[i].start);
            }
            //std::cout << "Processed "<< total_proc << " vertices\n";
        }
    }; //Queues

    template<typename T, typename C, int PADDING>
    __global__ void checkEnd(MaxCountQueue::dev::Queues<T, C, PADDING> q)
    {
        if(TID < q.num_queues)
            if(*(q.end+TID*PADDING) != *(q.end_alloc+TID*PADDING))
            {
                if(*(q.end_max+TID*PADDING) == *(q.end_count+TID*PADDING) 
                && *(q.end_count+TID*PADDING) == *(q.end_alloc+TID*PADDING))
                    *(q.end+TID*PADDING) = *(q.end_alloc+TID*PADDING);
                else 
                    printf("queue end update error: end[%d] %d, end_alloc[%d] %d, end_count[%d] %d, end_max[%d] %d\n", 
                    TID, *(q.end+TID*PADDING), TID, *(q.end_alloc+TID*PADDING), TID, *(q.end_count+TID*PADDING), TID, *(q.end_max+TID*PADDING));
            }
    }

    template<typename T, typename C>
    __global__ void checkEnd(MaxCountQueue::Queue<T, C> q)
    {
            if(*(q.end) != *(q.end_alloc))
            {
                if(*(q.end_max) == *(q.end_count) && *(q.end_count) == *(q.end_alloc))
                    *(q.end) = *(q.end_alloc);
                else 
                    printf("queue end update error: end[%d] %d, end_alloc[%d] %d, end_count[%d] %d, end_max[%d] %d\n", 
                    TID, *(q.end), TID, *(q.end_alloc), TID, *(q.end_count), TID, *(q.end_max));
            }
    }
} //MaxCountQueue
#endif
