#ifndef PRIORITY_QUEUE
#define PRIORITY_QUEUE

#include "../util/error_util.cuh"
#include "../util/util.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

    enum Priority {HIGH=0, LOW=1};  // guess a default value for priority type would be 0?

    namespace dev{
        template<typename T, typename THRESHOLD_TYPE, typename COUNTER_T=uint32_t>
        class PriorityQueues
        {
        public:
            int my_pe;
            int n_pes;

            T *queues;
            T * recv_queues;
            COUNTER_T capacity;
            COUNTER_T recv_capacity;
            volatile COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
            volatile int *stop;

            COUNTER_T *send_remote_alloc_end;
            COUNTER_T *sender_write_remote_end;
            volatile COUNTER_T *recv_read_local_end;

            COUNTER_T *execute = NULL;
            void *reserve;
            THRESHOLD_TYPE threshold;
            THRESHOLD_TYPE threshold_delta;
            volatile COUNTER_T * pivot_count;
            int min_iter;
            int sync_iter;
        
            PriorityQueues() {}
            
            PriorityQueues(int _my_pe, int _n_pes, COUNTER_T cap, COUNTER_T r_cap, T *queue, T * r_q, volatile COUNTER_T *counters, COUNTER_T * send_alloc,
            COUNTER_T *sender_write, COUNTER_T * recv_read, COUNTER_T *exe, void *res, THRESHOLD_TYPE thrshd, THRESHOLD_TYPE delta, int iter, int s_iter):
            my_pe(_my_pe), n_pes(_n_pes), capacity(cap), recv_capacity(r_cap), queues(queue), recv_queues(r_q), send_remote_alloc_end(send_alloc),
            sender_write_remote_end(sender_write), execute(exe), reserve(res), threshold(thrshd), threshold_delta(delta), min_iter(iter), sync_iter(s_iter)
            {
                // TODO no padding?
                start = counters;
                start_alloc = counters+1*2;
                end_alloc = counters+2*2;
                end = counters+3*2;
                end_max = counters+4*2;
                end_count = counters+5*2;
                stop = (int *)(counters+6*2);
                pivot_count = counters+7*2;
                //reserve = (void *)(counters+9*2);

                recv_read_local_end = recv_read;
                //printf("device queue, my_pe %d, n_pes %d, recv_queues %p, send_alloc %p, sender_write %p, recv_read %p\n", 
                //my_pe, n_pes, recv_queues, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end);
            }

            __forceinline__ __device__ T get_item(COUNTER_T idx, Priority prio) const
            { return *(queues + uint64_t(prio)*capacity + idx);}

            __forceinline__ __device__ T get_item_volatile(COUNTER_T idx, Priority prio) const
            { return *((volatile T *)(queues + uint64_t(prio)*capacity + idx));}

            __forceinline__ __device__ COUNTER_T len(Priority prio) const
            {return *(end+prio);}

            __forceinline__ __device__ void push_warp(Priority prio, T item) const {
                unsigned mask = __activemask();
                unsigned mask_prio = __match_any_sync(mask, prio);
                uint32_t total = __popc(mask_prio);
                uint32_t rank = __popc(mask_prio & lanemask_lt());
                COUNTER_T alloc = 0xffffffff;
                if(rank == 0)
                    alloc = atomicAdd((COUNTER_T *)(end_alloc+prio), total);
                alloc = __shfl_sync(mask_prio, alloc, __ffs(mask_prio)-1);

                __syncwarp(mask);
                if(alloc+total > capacity)
                    asm("trap;");
                *(queues + uint64_t(prio)*capacity + alloc+rank) = item;
                __threadfence();

                if(!rank) {
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+prio), (alloc+total));
                    //__threadfence();
                    if(maxNow < alloc+total) maxNow = alloc+total;
                    COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+prio), total) + total;
                    if(validC == maxNow)
                        atomicMax((COUNTER_T *)(end+prio), maxNow);
                }
                __syncwarp(mask);
            }

            __forceinline__ __device__ void push_cta(Priority prio, bool ifpush, T item) const
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
                    res[threadIdx.x] = atomicAdd( (COUNTER_T *)(end_alloc+threadIdx.x), total_cta[threadIdx.x]);
                    assert(res[threadIdx.x]!=0xffffffff);
                    assert(res[threadIdx.x]+total_cta[threadIdx.x]< capacity);
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
                    *(queues+uint64_t(prio)*capacity+res[prio]+alloc+rank) = item;
                    __threadfence();
                }

                __syncthreads();
                if(threadIdx.x < 2 && total_cta[threadIdx.x]) {
                    COUNTER_T maxNow = atomicMax((COUNTER_T *)(end_max+threadIdx.x),res[threadIdx.x]+total_cta[threadIdx.x]);
                    //__threadfence();
                    if(maxNow < res[threadIdx.x]+total_cta[threadIdx.x]) 
                        maxNow = res[threadIdx.x]+total_cta[threadIdx.x];
                    COUNTER_T validC = atomicAdd((COUNTER_T *)(end_count+threadIdx.x), total_cta[threadIdx.x])+total_cta[threadIdx.x];
                    if(maxNow == validC)
                        atomicMax((COUNTER_T *)(end+threadIdx.x), validC);
                }
            }

            __forceinline__ __device__ COUNTER_T grab_cta(int fetch_size, Priority prio) const
            {
                if(fetch_size <= 0) return FULLMASK_32BITS;
                __shared__ COUNTER_T old;
                if(!threadIdx.x) {
                    old = atomicAdd((COUNTER_T *)(start_alloc+prio), fetch_size);
                }
                __syncthreads();
                return old;
            }

            __forceinline__ __device__ COUNTER_T grab(int fetch_size, Priority prio) const
            {
                return atomicAdd((COUNTER_T *)(start_alloc+prio), fetch_size);
            }

            __forceinline__ __device__ void update_end_cta() const
            {
                if(threadIdx.x < 2) {
                    COUNTER_T maxNow = *(end_max+threadIdx.x);
                    if(*(end+threadIdx.x)!= maxNow && *(end_count+threadIdx.x) == maxNow)
                        atomicMax((COUNTER_T *)(end+threadIdx.x), maxNow);
                }
            }
        }; // priorityqueue

    }// dev

    template<typename T, typename THRESHOLD_TYPE, typename COUNTER_T=uint32_t>
    class PriorityQueues
    {
    public:
        int my_pe;
        int n_pes;

        T * queues;
        T * recv_queues;
        COUNTER_T capacity;
        COUNTER_T recv_capacity;
        volatile COUNTER_T *counters; // 10 counters, last one for reservation
        int num_counters = 10;
        volatile COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
        volatile int *stop;

        COUNTER_T * send_remote_alloc_end;
        COUNTER_T *sender_write_remote_end;
        COUNTER_T *recv_read_local_end;

        COUNTER_T *execute=NULL;
        void *reserve=NULL;
        COUNTER_T * pivot_count;
        THRESHOLD_TYPE threshold;
        THRESHOLD_TYPE threshold_delta;
        int min_iter;
        int sync_iter;

        PriorityQueues(COUNTER_T cap=0, COUNTER_T r_cap=0): capacity(cap), recv_capacity(r_cap)
        {   
            #ifdef DEBUG
                std::cout << "PriorityQueues constructor is called\n"; 
            #endif
            Alloc(); 
        }

        ~PriorityQueues() { 
            #ifdef DEBUG
                std::cout << "PriorityQueues destructor is called\n"; 
            #endif
            Free(); 
        }

        __host__ void init(int _my_pe, int _n_pes, THRESHOLD_TYPE thrshd, THRESHOLD_TYPE delta, COUNTER_T _capacity, COUNTER_T _r_cap, int iter, int s_iter)
        {
            my_pe = _my_pe;
            n_pes = _n_pes;
            if(capacity > 0) {
                std::cout << "Fail to initiate Priority queue, it has been initiated!\n";
                return;
            }
            capacity = _capacity;
            recv_capacity = _r_cap;
            threshold = thrshd;
            threshold_delta = delta;
            min_iter = iter;
            sync_iter = s_iter;

            Alloc();
        }

        void reset(cudaStream_t stream=0) const {
            CUDA_CHECK(cudaMemsetAsync((void *)queues, FULLMASK_32BITS, 
            sizeof(T)*capacity*2, stream));
            if(recv_queues)
            CUDA_CHECK(cudaMemsetAsync((void *)recv_queues, 0xffffffff, sizeof(T)*recv_capacity*(n_pes-1), stream));
            CUDA_CHECK(cudaMemsetAsync((void *)counters, 0, 
            sizeof(COUNTER_T)*2*num_counters, stream));
            if(send_remote_alloc_end)
            CUDA_CHECK(cudaMemsetAsync((void *)send_remote_alloc_end, 0, sizeof(COUNTER_T)*(n_pes-1), stream));
            if(sender_write_remote_end)
            CUDA_CHECK(cudaMemsetAsync((void *)sender_write_remote_end, 0, sizeof(COUNTER_T)*(n_pes-1), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        __host__ __device__ COUNTER_T get_capacity() const {return capacity;}

        __host__ COUNTER_T len(Priority prio, cudaStream_t stream=0) const {
            COUNTER_T h_end;
            CUDA_CHECK(cudaMemcpyAsync(&h_end, (void *)(end+prio), 
            sizeof(COUNTER_T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return h_end;
        }

        typedef dev::PriorityQueues<T, THRESHOLD_TYPE, COUNTER_T> DeviceObjectType;

        DeviceObjectType DeviceObject() {
            return dev::PriorityQueues<T, THRESHOLD_TYPE, COUNTER_T>
            (my_pe, n_pes, capacity, recv_capacity, queues, recv_queues, counters, send_remote_alloc_end,
            sender_write_remote_end, recv_read_local_end, execute, reserve, threshold, threshold_delta, min_iter, sync_iter);
        }

        void print() const {
            COUNTER_T check_counters[6];
            printf("high: start \t start_alloc \t   end_alloc \t      end \t   end_max \t end_count\n");
            for(int i=0; i<1; i++) {
                CUDA_CHECK(cudaMemcpy(check_counters, (void *)(start+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+1, (void *)(end+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+2, (void *)(start_alloc+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+3, (void *)(end_alloc+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+4, (void *)(end_max+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+5, (void *)(end_count+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                printf("%8d \t %8d \t %8d \t %8d \t %8d \t %8d \n",
                check_counters[0], check_counters[2], check_counters[3], check_counters[1], check_counters[4],
                check_counters[5]);
            }
            printf("low:  start \t start_alloc \t   end_alloc \t      end \t   end_max \t end_count\n");
            for(int i=1; i<2; i++) {
                CUDA_CHECK(cudaMemcpy(check_counters, (void *)(start+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+1, (void *)(end+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+2, (void *)(start_alloc+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+3, (void *)(end_alloc+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+4, (void *)(end_max+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(check_counters+5, (void *)(end_count+i), sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
                printf("%8d \t %8d \t %8d \t %8d \t %8d \t %8d \n",
                check_counters[0], check_counters[2], check_counters[3], check_counters[1], check_counters[4],
                check_counters[5]);
            }
        }

    private:
        void Alloc() {
            if(capacity == 0) return;
            CUDA_CHECK(cudaMalloc(&queues, sizeof(T)*capacity*2));
            CUDA_CHECK(cudaMemset((void *)queues, FULLMASK_32BITS, sizeof(T)*capacity*2));

            if(n_pes >= 2) {
                recv_queues = (T *)nvshmem_malloc(sizeof(T)*recv_capacity*(n_pes-1));
                CUDA_CHECK(cudaMemset(recv_queues, -1, sizeof(T)*recv_capacity*(n_pes-1)));
            } else recv_queues = NULL;

            CUDA_CHECK(cudaMallocManaged(&counters, sizeof(COUNTER_T)*2*num_counters));
            //CUDA_CHECK(cudaMalloc(&counters, sizeof(COUNTER_T)*2*num_counters));
            CUDA_CHECK(cudaMemset((void *)counters, 0, sizeof(COUNTER_T)*2*num_counters));

            start = counters;
            start_alloc = counters+1*2;
            end_alloc = counters+2*2;
            end = counters+3*2;
            end_max = counters+4*2;
            end_count = counters+5*2;
            stop = (int *)(counters+6*2);
            pivot_count = (uint32_t *)(counters+6*2+2);
            //reserve = (void *)(counters+6*2+6);

            if(n_pes >= 2) {
                CUDA_CHECK(cudaMalloc(&send_remote_alloc_end, sizeof(COUNTER_T)*(n_pes-1)));
                CUDA_CHECK(cudaMemset(send_remote_alloc_end, 0, sizeof(COUNTER_T)*(n_pes-1)));

                sender_write_remote_end = (COUNTER_T *)nvshmem_malloc(sizeof(COUNTER_T)*(n_pes-1));
                CUDA_CHECK(cudaMemset(sender_write_remote_end, 0, sizeof(COUNTER_T)*(n_pes-1)));
                recv_read_local_end = (COUNTER_T *)nvshmem_ptr(sender_write_remote_end, my_pe);
            } else {
                send_remote_alloc_end = NULL;
                sender_write_remote_end = NULL;
                recv_read_local_end = NULL;
            }
            printf("my_pe %d, n_pes %d, recv_queues %p, r_cap %d, send_alloc %p, sender_write %p, recv_read %p\n", 
            my_pe, n_pes, recv_queues, recv_capacity, send_remote_alloc_end, sender_write_remote_end, recv_read_local_end);
        }

        void Free() {
            if(capacity == 0) return;
            CUDA_CHECK(cudaFree(queues));
            CUDA_CHECK(cudaFree((void *)counters));
            if(execute!=NULL)
                CUDA_CHECK(cudaFree(execute));
        }
    }; // priorityqueue
} // MaxCountQueue

#endif