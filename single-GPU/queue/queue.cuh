#ifndef QUEUE
#define QUEUE

#include <inttypes.h>
#include <limits>
#include <assert.h>

#include "../../util/error_util.cuh"
#include "../../util/util.cuh"

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
            uint32_t temp = __shfl_down_sync(mask, loadValue, offset);
            __syncwarp(mask);
            if(LANE_+offset < __popc(mask))
                loadValue = min(loadValue, temp);
            __syncwarp(mask);
        }
        __syncwarp(mask);
        minValue = min(minValue, loadValue);
//        if(LANE_ == 0)
//        printf("min: %d\n", minValue);
    }
    __syncwarp();
    minValue = __shfl_sync(0xffffffff, minValue, 0);
    return minValue;
}

namespace MaxCountQueue {
    #define PADDING_SIZE (64)  // 32 gets bad performance
template<typename T, typename COUNTER_T>
struct Queue
{
    T * queue;
    COUNTER_T capacity;
    volatile COUNTER_T *counters; // 10 counters, last one for reservation
    int num_counters = 10;
    volatile COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
    volatile COUNTER_T *stop;
    uint32_t min_iter;    
    uint32_t queue_id=0;      // remove queue_id change the structure layout and gets bad performance
    int num_queues = 1;
    COUNTER_T *execute=NULL;
    cudaStream_t *discrete_streams=NULL;


    Queue() {}

    __host__ void init(COUNTER_T _capacity, uint32_t _min_iter=800)
    {
        capacity = _capacity; 

        CUDA_CHECK(cudaMalloc(&queue, sizeof(T)*capacity)); 
        CUDA_CHECK(cudaMemset((void *)queue, -1, sizeof(T)*capacity));

        CUDA_CHECK(cudaMallocManaged(&counters, sizeof(COUNTER_T)*num_counters*PADDING_SIZE));
        CUDA_CHECK(cudaMemset((void *)counters, 0, sizeof(COUNTER_T)*num_counters*PADDING_SIZE));
        start = (counters);
        start_alloc = (counters+1*PADDING_SIZE);
        end_alloc = (counters+2*PADDING_SIZE);
        end = (counters+3*PADDING_SIZE);
        end_max = (counters+4*PADDING_SIZE);
        end_count = (counters+5*PADDING_SIZE);
        stop = (counters+6*PADDING_SIZE);
	    min_iter = _min_iter;
    }

    __host__ void init(COUNTER_T _capacity, T *_queue, volatile COUNTER_T *_start, volatile COUNTER_T *_end, volatile COUNTER_T *_start_alloc,
            volatile COUNTER_T* _end_alloc, volatile COUNTER_T* _end_max, volatile COUNTER_T* _end_count, volatile COUNTER_T *_stop, int _num_queues, uint32_t _queue_id, uint32_t _min_iter=800)
    {
        
        capacity = _capacity; 
        queue = _queue;
        start = _start;
        end = _end;
        start_alloc = _start_alloc;
        end_alloc = _end_alloc;
        end_max = _end_max;
        end_count = _end_count;
        stop = _stop;
        min_iter = _min_iter;
        queue_id = _queue_id;
	num_queues = _num_queues;
    }

    __host__ void reset()
    {
        CUDA_CHECK(cudaMemset((void *)queue, -1, sizeof(T)*capacity));
        CUDA_CHECK(cudaMemset((void *)start, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)start_alloc, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)end, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)end_alloc, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)end_max, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)end_count, 0, sizeof(COUNTER_T)));
        CUDA_CHECK(cudaMemset((void *)stop, 0, sizeof(COUNTER_T)));
    }
    
    __host__ __device__ COUNTER_T get_capacity() {return capacity;}
    __forceinline__ __device__ T get_item(COUNTER_T index) { return ((volatile T *)queue)[index]; }
    __forceinline__ __device__ void push_warp(const T& item) const;
    __forceinline__ __device__ void push_cta(bool ifpush, const T& item) const;
    __forceinline__ __device__ void update_end() const;
    __forceinline__ __device__ void update_end_execute(COUNTER_T init_value) const;
    __forceinline__ __device__ COUNTER_T grab_one() const;
    __device__ COUNTER_T grab_warp(const uint32_t& total) const;
    __device__ COUNTER_T grab_cta(const uint32_t &total) const;

    template<typename Functor, typename... Args>
    __host__ void launchThreadPerItem(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg);

    template<typename Functor, typename... Args>
    __host__ void launchThreadPerItem_minIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg);

    template<typename Functor, typename... Args>
    __host__ void launchWarpPerItem(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg);

    template<typename Functor, typename... Args>
    __host__ void launchWarpPer32Items_maxIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg);

    template<typename Functor, typename... Args>
    __host__ void launchWarpPer32Items_minIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg);

    template<typename Functor1, typename Functor2, typename... Args>
    __host__ void launchWarpPer32Items_minIter_2func(int numBlock, int numThread, cudaStream_t stream, Functor1 f1, Functor2 f2, Args... arg);

    template<int FETCH_SIZE, typename Functor, typename... Args>
    __host__ void launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg);

    template<int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args>
    __host__ void launchCTA_minIter_2func(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor1 f1, Functor2 f2, Args... arg);

    __host__ void launchDiscrete_prepare();

    template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
    __host__ void launchDiscrete_minIter(uint32_t sharedMem, Functor F, Args... arg);

    __host__ void release()
    {
        if(queue!=NULL)
        cudaFree(queue);
        if(counters!=NULL)
        cudaFree((void *)counters);
	if(execute!=NULL)
	cudaFree(execute);
    }

    __host__ void print() const
    {
        COUNTER_T h_end, h_end_alloc, h_end_max, h_end_count, h_start, h_start_alloc;
        CUDA_CHECK(cudaMemcpy(&h_end, (COUNTER_T *)end, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_end_alloc, (COUNTER_T *)end_alloc, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_end_max, (COUNTER_T *)end_max, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_end_count, (COUNTER_T *)end_count, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_start, (COUNTER_T *)start, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_start_alloc, (COUNTER_T *)start_alloc, sizeof(COUNTER_T), cudaMemcpyDeviceToHost));
        if(h_end == h_end_alloc && h_end_alloc == h_end_max && h_end_max == h_end_count && h_end != h_start)
            printf("Queue %d, capacity: %zd, end: %zd\tstart: %zd\n", queue_id, size_t(capacity), size_t(h_end),
                    size_t(h_start));
        else if(h_end == h_end_alloc && h_end_alloc == h_end_max && h_end_max == h_end_count && h_end == h_start)
            printf("Queue %d, capacity: %zd, end=start: %zd\n", queue_id, size_t(capacity), size_t(h_end));
        else printf("Queue %d, capacity: %zd\nend: %zd\nend_alloc: %zd\nend_max: %zd\nend_count: %zd\nstart: %zd\nstart_alloc: %zd\n",
                queue_id, size_t(capacity), size_t(h_end), size_t(h_end_alloc), size_t(h_end_max), size_t(h_end_count), size_t(h_start), size_t(h_start_alloc));
    }
};

template<typename T, typename COUNTER_T>
__forceinline__ __device__ void Queue<T, COUNTER_T>::push_warp(const T& item) const
{
    unsigned mask = __activemask();
    uint32_t total = __popc(mask);
    unsigned int rank = __popc(mask & lanemask_lt());
    int leader = __ffs(mask)-1;
    COUNTER_T alloc;
    if(rank == 0){
        alloc = atomicAdd((COUNTER_T *)end_alloc, total);
    }
    alloc = __shfl_sync(mask, alloc, leader);
    assert(alloc + total <= capacity);
    queue[(alloc+rank)] = item;
    __threadfence();

    if(rank == 0) {
        COUNTER_T maxNow = atomicMax((COUNTER_T *)end_max, (alloc+total));
        //__threadfence();
        if(maxNow < alloc+total) maxNow = alloc+total;
        COUNTER_T validC = atomicAdd((COUNTER_T *)end_count, total)+total;
        if(validC == maxNow)
        {
            __threadfence();
            atomicMax((COUNTER_T *)end, maxNow);
        }
    }
    __syncwarp(mask);
}

template<typename T, typename COUNTER_T>
__forceinline__ __device__ void Queue<T, COUNTER_T>::push_cta(bool ifpush, const T& item) const
{
    __shared__ COUNTER_T res;
    __shared__ int total_cta; 
    //init res and total_cta to 0
    if(threadIdx.x == 0)
    {
        res = 0xffffffff; total_cta = 0;
 //       printf("res %p, total_cta %p\n", &res, &total_cta);
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
        //if(res+total_cta >= capacity) printf("queue is full res %d, total_cta %d\n", res, total_cta);
        assert(res+total_cta < capacity);
    }

    //synchronize all threads of cta, make sure res value is ready
    __syncthreads();
    
    if(ifpush)
    {
        assert(res!=0xffffffff);
        queue[res+alloc+rank] = item;
        __threadfence();
    }

    __syncthreads();
    // first thread update end_max, end_count and maybe end
    if(threadIdx.x == 0 && total_cta)
    {
        assert(res!=0xffffffff);
        COUNTER_T maxNow = atomicMax((COUNTER_T *)end_max, res+total_cta);
        if(maxNow < res+total_cta) maxNow = res+total_cta;
        COUNTER_T validC = atomicAdd((COUNTER_T *)end_count, total_cta)+total_cta;
        if(maxNow == validC)
            atomicMax((COUNTER_T *)end, validC);
    }

}

template<typename T, typename COUNTER_T>
__forceinline__ __device__ void Queue<T, COUNTER_T>::update_end() const
{
    if(LANE_ == 0) {
        COUNTER_T maxNow = *(end_count);
        if(*(end)!= maxNow && *(end_max) == maxNow)
        {
            atomicMax((COUNTER_T *)end, maxNow);
        }
    }
}

template<typename T, typename COUNTER_T>
__forceinline__ __device__ void Queue<T, COUNTER_T>::update_end_execute(COUNTER_T init_value = std::numeric_limits<COUNTER_T>::max()) const
{
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

template<typename T, typename COUNTER_T>
__forceinline__ __device__ COUNTER_T Queue<T, COUNTER_T>::grab_one() const
{
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

//assume whole warp participate
template<typename T, typename COUNTER_T>
__device__ COUNTER_T Queue<T, COUNTER_T>::grab_warp(const uint32_t& total) const
{
   if(total==0) return COUNTER_T(MAX_SZ);
   COUNTER_T index;
   if(LANE_ == 0 ) {
       index = atomicAdd((COUNTER_T*)start_alloc, total); 
//       printf("%d, %d, index %d\n", WARPID, LANE_, (uint32_t)index);
   }
   __syncwarp();
   index = __shfl_sync(0xffffffff, index, 0);
   return index;
}

template<typename T, typename COUNTER_T>
__device__ COUNTER_T Queue<T, COUNTER_T>::grab_cta(const uint32_t &total) const
{
    if(total == 0) return COUNTER_T(MAX_SZ);
    __shared__ COUNTER_T index;
    if(threadIdx.x ==0)
    {
        index = atomicAdd((COUNTER_T *)start_alloc, total);
    }
    __syncthreads();
    return index;
}

template<typename Functor, typename T, typename COUNTER_T, typename... Args>
__forceinline__ __device__ void func(Functor F, T t, Args... args) // recursive variadic function
{
   func(F, args...);
}

template<typename T, typename COUNTER_T, typename Functor, typename... Args>
__global__ void _launchThreadPerItem(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
    COUNTER_T index=TID;
    bool secondtime=0;
    uint32_t iter = 0;
    do {
        COUNTER_T end = *(q.end);
        if(index<end && secondtime == 0)
        {
            T task = q.get_item(index);
            assert(task!=-1);
            func(F, task, args...);
        }
        __syncwarp();
        bool left = index<end?1:0;
        unsigned cross = __ballot_sync(0xffffffff, left);
        if(cross == 0xffffffff) {
            index = index+gridDim.x*blockDim.x;
            secondtime = 0;
        }
        else {
            secondtime = left;
        }
        __syncwarp();
    }
    while(iter++ < q.min_iter);
    unsigned mask = __ballot_sync(0xffffffff, secondtime!=0);
    if(LANE_ == 0)
        q.execute[WARPID] = index+__popc(mask);

}

template<typename T, typename COUNTER_T, typename Functor, typename... Args>
__launch_bounds__(512,2)
__global__ void _launchThreadPerItem_minIter(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
    COUNTER_T index=TID;
    COUNTER_T end = *(q.end);
    uint32_t iter = 0;
    __shared__ COUNTER_T warp_min[32];
    if(LANE_==0) warp_min[threadIdx.x>>5] = 2147483647;
    do {
        while(index < end)
        {
            T task = q.get_item(index);
            assert(task!=-1);
            func(F, task, args...);
	    __syncwarp();
	    index = q.grab_one();
        }
        __syncwarp();
        end = *(q.end);
        if(index >= end)
        {
           if(*(q.stop) == blockDim.x*gridDim.x/32*q.num_queues)
               break;
           iter++;
           if(iter == q.min_iter)
		 if(LANE_ < q.num_queues)
                   atomicAdd((COUNTER_T *)(q.stop+LANE_*64), 1);
           q.update_end();
	   __syncwarp();
        }
    }
    while(true);
    atomicMin(warp_min+(threadIdx.x >>5), index);
    __syncwarp();
    if(LANE_ == 0)
        q.execute[WARPID] = warp_min[threadIdx.x>>5];

}

template<typename T, typename COUNTER_T, typename Functor, typename... Args>
__global__ void _launchWarpPerItem(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
   COUNTER_T index = q.grab_warp(1);
   __syncwarp();
   uint32_t iter = 0;
   do {
       if(index < *(q.end))
       {
           T task = q.get_item(index);
	   assert(task != -1);
           __syncwarp();
           func(F, task, args...);
           __syncwarp();
           index = q.grab_warp(1);
           __syncwarp();
        }
    }
    while(*q.stop == 0 && iter++ < q.min_iter);
    if(LANE_ == 0)
        q.execute[WARPID] = index;
}

template<typename T, typename COUNTER_T, typename Functor, typename... Args>
__global__ void _launchWarpPer32Items_maxIter(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
    COUNTER_T index = q.grab_warp(32);
    uint32_t iter = 0;
    uint32_t exec = 0;
    do {
        if(index < *(q.end))
        {
            unsigned activeMask = 0;
            T task;
            if(index+LANE_ < *(q.end) && index+LANE_ >= index+exec)
            {
                task = q.get_item(index+LANE_);
		assert(task != -1);
                activeMask = __activemask();
            }
            __syncwarp();
            activeMask = __shfl_sync(0xffffffff, activeMask, exec);
            for(int i=0; i<__popc(activeMask); i++)
            {
                T cur_task = __shfl_sync(0xffffffff, task, i+exec);
                __syncwarp();
                func(F, cur_task, args...);
                __syncwarp();
            }
            exec = exec + __popc(activeMask);
            if(exec == 32)
            {
                exec = 0;
                __syncwarp();
                index = q.grab_warp(32);
            }
            __syncwarp();
        }
        __syncwarp();
    }
    while(*(q.stop)==0 && iter++ < q.min_iter);

    if(LANE_ == 0)
    q.execute[WARPID] = index+exec;
}

template<typename T, typename COUNTER_T, typename Functor, typename... Args>
//__launch_bounds__(512,2)
__global__ void _launchWarpPer32Items_minIter(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
    
    COUNTER_T index = q.grab_warp(32);
    __syncwarp(0xffffffff);
    COUNTER_T e = *(q.end);
    uint32_t iter = 0;
    do {
        while(index < e)
        {
            unsigned activeMask = __ballot_sync(0xffffffff, (index+LANE_)<min(e, align_up_yx(index+1, 32)) );
	    activeMask = __popc(activeMask);
            T task;
            if(index+LANE_ < min(e,align_up_yx(index+1, 32)))
	    {
                task = q.get_item(index+LANE_);
		//assert(task!= -1);
	    }
            
            __syncwarp(0xffffffff);

            for(int i=0; i<activeMask; i++)
            {
                T cur_task = __shfl_sync(0xffffffff, task, i);
                __syncwarp(0xffffffff);
                func(F, cur_task, args...);
                __syncwarp(0xffffffff);
            }

            if(activeMask+ index == align_up_yx(index+1, 32))
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
        __syncwarp(0xffffffff);

        e = *(q.end);
        __syncwarp(0xffffffff);
        if(index >= e)
        {
            if(*(q.stop) == blockDim.x*gridDim.x/32*q.num_queues)
                break;
            __syncwarp(0xffffffff);
            iter++;
            if(iter == q.min_iter) // if just lane0 increment stop, get lower performance on road_ca, road_usa
		if(LANE_ == 0)
                    atomicAdd((COUNTER_T *)(q.stop), 1);
            q.update_end();
            __syncwarp(0xffffffff);
	        __nanosleep(600);
        }
//        __syncwarp(0xffffffff);
    }
    while(true);

    if(LANE_ == 0)
    {
    q.execute[WARPID] = index;
//    printf("warp %d, index %d\n", WARPID, index);
    }
}

template<typename T, typename COUNTER_T, typename Functor1, typename Functor2, typename... Args>
//__launch_bounds__(512,2)
__global__ void _launchWarpPer32Items_minIter_2func(Queue<T, COUNTER_T> q, Functor1 F1, Functor2 F2, Args... args)
{

    COUNTER_T index = q.grab_warp(32);
    __syncwarp(0xffffffff);
    COUNTER_T e = *(q.end);
    uint32_t iter = 0;
    do {
        while(index < e)
        {
            unsigned activeMask = __ballot_sync(0xffffffff, (index+LANE_)<min(e, align_up_yx(index+1, 32)) );
	        activeMask = __popc(activeMask);
            T task;
            if(index+LANE_ < min(e,align_up_yx(index+1, 32)))
	        {
                task = q.get_item(index+LANE_);
	        }

            __syncwarp(0xffffffff);

            for(int i=0; i<activeMask; i++)
            {
                T cur_task = __shfl_sync(0xffffffff, task, i);
                __syncwarp(0xffffffff);
                func(F1, cur_task, args...);
                __syncwarp(0xffffffff);
            }

            if(activeMask+ index == align_up_yx(index+1, 32))
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
        __syncwarp(0xffffffff);

        e = *(q.end);
        __syncwarp(0xffffffff);
        if(index >= e)
        {
            func(F2, args...);
            if(*(q.stop) == blockDim.x*gridDim.x/32*q.num_queues)
                break;
            __syncwarp(0xffffffff);
            iter++;
            if(iter == q.min_iter) // if just lane0 increment stop, get lower performance on road_ca, road_usa
		        if(LANE_ == 0)
                    atomicAdd((COUNTER_T *)(q.stop), 1);
            q.update_end();
            __syncwarp(0xffffffff);
	        //__nanosleep(600);
        }
//        __syncwarp(0xffffffff);
    }
    while(true);

    if(LANE_ == 0)
    {
        q.execute[WARPID] = index;
//    printf("warp %d, index %d\n", WARPID, index);
    }
}

template<typename T, typename COUNTER_T, int FETCH_SIZE, typename Functor, typename... Args >
__launch_bounds__(1024,1)
__global__ void _launchCTA_minIter(Queue<T, COUNTER_T> q, Functor F, Args... args)
{
    __shared__ COUNTER_T end;
    __shared__ T nodes[FETCH_SIZE];

    int fs = FETCH_SIZE;
    int iter = 0;
    COUNTER_T index = q.grab_cta(FETCH_SIZE);
    if(threadIdx.x == 0) {
        end = *(q.end);
        q.execute[blockIdx.x] = index;
//        printf("end %p, nodes %p\n", &end, nodes);
    }
    __syncthreads();

    do {
        while(index < end)
     	{
            fs = min(end-index, fs);
     	    if(threadIdx.x < fs)
     	    {
            //    printf("thread %d, block %d, end %d, index %d, fs %d\n", threadIdx.x, blockIdx.x, end, index, fs);
     		    nodes[threadIdx.x]= q.get_item(index+threadIdx.x);
     		    //assert(nodes[threadIdx.x]!= -1);
     	    }
     	    __syncthreads();

     	    func(F, nodes, fs, args...);
     	    __syncthreads();

     	    if(fs+index == align_up_yx(index+1, FETCH_SIZE))
     	    {
     		    index = q.grab_cta(FETCH_SIZE);
                 fs = FETCH_SIZE;
                 if(threadIdx.x == 0)
                    q.execute[blockIdx.x] = index;
            }
            else
            {
                index = index + fs;
                fs = align_up_yx(index, FETCH_SIZE) - index;
                if(threadIdx.x == 0)
                {
                    end = *(q.end);
                     q.update_end();
                     q.execute[blockIdx.x] = index;
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
            if(*(q.stop) == gridDim.x*q.num_queues)
                break;
            __syncthreads();
            if(threadIdx.x == 0)
            {
                iter++;
                if(iter == q.min_iter)
                    atomicAdd((COUNTER_T *)q.stop, 1);
            }
            q.update_end_execute();
//            __nanosleep(60);
        }
//     	__syncthreads();
    }
    while(true);
    if(threadIdx.x == 0)
    {
        q.execute[blockIdx.x] = index;
    }
}

template<typename T, typename COUNTER_T, int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args >
__launch_bounds__(1024,1)
__global__ void _launchCTA_minIter_2func(Queue<T, COUNTER_T> q, Functor1 F1, Functor2 F2, Args... args)
{
    __shared__ COUNTER_T end;
    __shared__ T nodes[FETCH_SIZE];

    int fs = FETCH_SIZE;
    int iter = 0;
    COUNTER_T index = q.grab_cta(FETCH_SIZE);
    if(threadIdx.x == 0) {
	    end = *(q.end);
    }
    __syncthreads();

    do {
        while(index < end)
     	{
            fs = min(end-index, fs);
     	    if(threadIdx.x < fs)
     	    {
     		    nodes[threadIdx.x]= q.get_item(index+threadIdx.x);
     		    assert(nodes[threadIdx.x]!= -1);
     	    }
     	    __syncthreads();

     	    func(F1, nodes, fs, args...);
     	    __syncthreads();

     	    if(fs+index == align_up_yx(index+1, FETCH_SIZE))
     	    {
     		    index = q.grab_cta(FETCH_SIZE);
     		    fs = FETCH_SIZE;
            }
            else
            {
                index = index + fs;
                fs = align_up_yx(index, FETCH_SIZE) - index;
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
            func(F2, args...);
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
//            __nanosleep(60);
        }
//     	__syncthreads();
    }
    while(true);
    if(threadIdx.x == 0)
    {
        q.execute[blockIdx.x] = index;
    }
}

template<typename T>
__global__ void ReductionMin(T *array, uint32_t size, T *out)
{
    T res = reductionMin_warp(array, size);
    if(LANE_ == 0)
        *out = res;
        //printf("size %d, min %d\n", size, int(res));
}

template<typename T, typename COUNTER_T>
template<typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchThreadPerItem(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchThreadPerItem<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchThreadPerItem_minIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg)
{
    std::cout << "LaunchThreadPerItem_minIter numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchThreadPerItem_minIter<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchWarpPerItem(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchWarpPerItem<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchWarpPer32Items_maxIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchWarpPer32Items_maxIter<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchWarpPer32Items_minIter(int numBlock, int numThread, cudaStream_t stream, Functor f, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchWarpPer32Items_minIter<<<numBlock, numThread, 0, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<typename Functor1, typename Functor2, typename... Args>
void Queue<T, COUNTER_T>::launchWarpPer32Items_minIter_2func(int numBlock, int numThread, cudaStream_t stream, Functor1 f1, Functor2 f2, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread<<std::endl;
    _launchWarpPer32Items_minIter_2func<<<numBlock, numThread, 0, stream>>>(*this, f1, f2, arg...);
}

template<typename T, typename COUNTER_T>
template<int FETCH_SIZE, typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchCTA_minIter(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor f, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " share memory size(bytes) "<< shareMem_size << std::endl;
    _launchCTA_minIter<T, COUNTER_T, FETCH_SIZE><<<numBlock, numThread, shareMem_size, stream>>>(*this, f, arg...);
}

template<typename T, typename COUNTER_T>
template<int FETCH_SIZE, typename Functor1, typename Functor2, typename... Args>
void Queue<T, COUNTER_T>::launchCTA_minIter_2func(int numBlock, int numThread, cudaStream_t stream, int shareMem_size, Functor1 f1, Functor2 f2, Args... arg)
{
    std::cout << "numBlock "<< numBlock << " numThread "<< numThread << " fetch size "<< FETCH_SIZE << " dynamic share memory size(bytes) "<< shareMem_size << std::endl;
    _launchCTA_minIter_2func<T, COUNTER_T, FETCH_SIZE><<<numBlock, numThread, shareMem_size, stream>>>(*this, f1, f2, arg...);
}

template<typename T, typename COUNTER_T>
void Queue<T, COUNTER_T>::launchDiscrete_prepare()
{
    discrete_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*33);
    for(int i=0; i<33; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(discrete_streams+i, cudaStreamNonBlocking));  
    }
}

template<typename T, typename COUNTER_T>
template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
void Queue<T, COUNTER_T>::launchDiscrete_minIter(uint32_t sharedMem, Functor F, Args... arg)
{
    uint32_t end_host, start_host=0;
    int iter=0;
    int log_iter=0;

    while(iter < min_iter)
    {
        CUDA_CHECK(cudaMemcpyAsync(&end_host, (uint32_t *)end, sizeof(uint32_t), cudaMemcpyDeviceToHost, discrete_streams[32]));
        CUDA_CHECK(cudaStreamSynchronize(discrete_streams[32]));
        int size = end_host-start_host;
        int stream_id = log_iter%32;
        if(size > 0)
        {
            func(F, start_host, size, sharedMem, discrete_streams[stream_id], arg...);
            log_iter++;
            start_host = end_host;
            iter = 0;
        }
        else {
            if(iter > min_iter/2)
                checkEnd<<<1,32,0, discrete_streams[32]>>>(*this);
            iter++;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy((uint32_t *)(start), &start_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
}

template<typename T, typename COUNTER_T=uint32_t>
struct Queues
{
    T * queue;
    COUNTER_T capacity;
    uint32_t num_queues;
    volatile COUNTER_T *counters; // 10 counters, last one for reservation
    int num_counters = 7;
    volatile COUNTER_T *start, *end, *start_alloc, *end_alloc, *end_max, *end_count;
    volatile COUNTER_T *stop;
    uint32_t min_iter;
    COUNTER_T *execute;
    MaxCountQueue::Queue<T, COUNTER_T> *worklist;
    cudaStream_t *streams;
    int warpSize_wl_pop=-1;

    __host__ void init(COUNTER_T _capacity, uint32_t _num_q=8, uint32_t _min_iter=800)
    {
        capacity = _capacity; 
        num_queues = _num_q;

        CUDA_CHECK(cudaMalloc(&queue, sizeof(T)*capacity*num_queues)); 
        CUDA_CHECK(cudaMemset((void *)queue, -1, sizeof(T)*capacity*num_queues));

        CUDA_CHECK(cudaMallocManaged(&counters, sizeof(COUNTER_T)*num_counters*num_queues*PADDING_SIZE));
        CUDA_CHECK(cudaMemset((void *)counters, 0, sizeof(COUNTER_T)*num_counters*num_queues*PADDING_SIZE));
        start = (counters);
        start_alloc = (counters+1*num_queues*PADDING_SIZE);
        end_alloc = (counters+2*num_queues*PADDING_SIZE);
        end = (counters+3*num_queues*PADDING_SIZE);
        end_max = (counters+4*num_queues*PADDING_SIZE);
        end_count = (counters+5*num_queues*PADDING_SIZE);
        stop = (counters+6*num_queues*PADDING_SIZE);
	min_iter = _min_iter;

        worklist = (MaxCountQueue::Queue<T, COUNTER_T> *)malloc(sizeof(MaxCountQueue::Queue<T, COUNTER_T>)*num_queues);
        streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*num_queues);

        for(uint64_t i=0; i<num_queues; i++)
        {
//            worklist[i].init(capacity, queue+i*capacity, start+i*64, end+i*64, start_alloc+i*64, end_alloc+i*64, end_max+i*64, end_count+i*64, stop+i*64, num_queues, min_iter, max_iter);
            worklist[i].init(capacity, queue+i*capacity, start+i*PADDING_SIZE, end+i*PADDING_SIZE, start_alloc+i*PADDING_SIZE, end_alloc+i*PADDING_SIZE, end_max+i*PADDING_SIZE, end_count+i*PADDING_SIZE, stop, num_queues, i, min_iter);
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

    __host__ void reset()
    {
        for(int i=0; i<num_queues; i++)
        {
            worklist[i].reset();
            if(warpSize_wl_pop!=-1)
	            CUDA_CHECK(cudaMemset(worklist[i].execute, 0, sizeof(COUNTER_T)*warpSize_wl_pop));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    __host__ COUNTER_T get_capacity() {return capacity;}

    __host__ void launchWarpPer32Items_minIter_preLaunch(int numBlock, int numThread)
    {
        if(warpSize_wl_pop == -1)
        {
            warpSize_wl_pop = numBlock/num_queues*numThread/32;
            for(int i=0; i<num_queues; i++)
            {
                CUDA_CHECK(cudaMalloc(&(worklist[i].execute), sizeof(COUNTER_T)*warpSize_wl_pop));
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
    
    __host__ void launchDiscrete_prepare()
    {
        worklist[0].launchDiscrete_prepare();
    }

    template<typename Functor, typename... Args>
    __host__ void launchThreadPerItem(int numBlock, int numThread, Functor f, Args... arg)
    {
        for(int i=0; i<num_queues; i++)
        {
            worklist[i].launchThreadPerItem_minIter(numBlock/num_queues, numThread, streams[i], f, arg...);
        }
    }

    template<typename Functor, typename... Args>
    __host__ void launchWarpPer32Items_minIter(int numBlock, int numThread, Functor f, Args... arg)
    {
        for(int i=0; i<num_queues; i++)
        {
            worklist[i].launchWarpPer32Items_minIter(numBlock/num_queues, numThread, streams[i], f, arg...);
        }
    }

    template<typename Functor1, typename Functor2, typename... Args>
    __host__ void launchWarpPer32Items_minIter_2func(int numBlock, int numThread, Functor1 f1, Functor2 f2, Args... arg)
    {
        for(int i=0; i<num_queues; i++)
        {
            worklist[i].launchWarpPer32Items_minIter_2func(numBlock/num_queues, numThread, streams[i], f1, f2, arg...);
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
    __host__ void launchCTA_minIter_2func(int numBlock, int numThread, int shared_mem, Functor1 f1, Functor2 f2, Args... arg)
    {
        for(int i=0; i<num_queues; i++)
        {
            worklist[i].template launchCTA_minIter_2func<FETCH_SIZE>(numBlock/num_queues, numThread, streams[i], shared_mem, f1, f2, arg...);
        }
    }

    template<int FETCH_SIZE, int BLOCK_SIZE, typename Functor, typename... Args>
    __host__ void launchDiscrete_minIter(uint32_t shared_mem, Functor f, Args... arg)
    {
        worklist[0].template launchDiscrete_minIter<FETCH_SIZE, BLOCK_SIZE>(shared_mem, f, arg...);
    }

    __host__ void sync_all_wl()
    {
        for(int i=0; i<num_queues; i++)
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    __host__ void print()
    {
        for(int i=0; i<num_queues; i++)
        {
            //assert(warpSize_wl_pop>0);
            if(warpSize_wl_pop > 0) {
            ReductionMin<<<1, 32>>>(worklist[i].execute, warpSize_wl_pop, (COUNTER_T *)(worklist[i].start));
            CUDA_CHECK(cudaDeviceSynchronize());
            }
            worklist[i].print();
        }
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
//            __threadfence();
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
     //       printf("res %p, total_cta %p\n", &res, &total_cta);
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

    __forceinline__ __device__ T get_item(COUNTER_T index, uint64_t q_id) const
    {
        return *((volatile T *)(queue+q_id*capacity+index));
    }
}; //Queues

template<typename T>
__global__ void checkEnd(MaxCountQueue::Queues<T> q)
{
    if(TID < q.num_queues)
        if(*(q.end+TID*PADDING_SIZE) != *(q.end_alloc+TID*PADDING_SIZE))
        {
            if(*(q.end_max+TID*PADDING_SIZE) == *(q.end_count+TID*PADDING_SIZE) && *(q.end_count+TID*PADDING_SIZE) == *(q.end_alloc+TID*PADDING_SIZE))
                *(q.end+TID*PADDING_SIZE) = *(q.end_alloc+TID*PADDING_SIZE);
            else 
                printf("queue end update error: end[%d] %d, end_alloc[%d] %d, end_count[%d] %d, end_max[%d] %d\n", TID, *(q.end+TID*PADDING_SIZE), TID, *(q.end_alloc+TID*PADDING_SIZE), TID, *(q.end_count+TID*PADDING_SIZE), TID, *(q.end_max+TID*PADDING_SIZE));
        }
}

template<typename T, typename C>
__global__ void checkEnd(MaxCountQueue::Queue<T, C> q)
{
        if(*(q.end) != *(q.end_alloc))
        {
            if(*(q.end_max) == *(q.end_count) && *(q.end_count) == *(q.end_alloc))
                *(q.end) = *(q.end_alloc);
            //else 
            //    printf("queue end update error: end[%d] %d, end_alloc[%d] %d, end_count[%d] %d, end_max[%d] %d\n", TID, *(q.end), TID, *(q.end_alloc), TID, *(q.end_count), TID, *(q.end_max));
        }
}
} //MaxCountQueue
#endif
