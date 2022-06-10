#include "../util/util.cuh"
#include "../util/nvshmem_util.cuh"
//#include "distr_queue_inter_nodes_agent_maxcount.cuh"
#include "distr_queue_base.cuh"

namespace Atos{
	namespace MAXCOUNT{
	namespace dev {
		template<typename RECV_T, typename COUNTER_T, int INTER_BATCH_SIZE, int PADDING_SIZE=1>
		class Agent
		{
		public:
			int my_pe;
			int n_pes;
			int nodes_size;
			int node_id;
			int group_size;
			int group_id;

			int num_aggregate_queues;
			COUNTER_T capacity;
			volatile COUNTER_T *aggregate_start;
			volatile COUNTER_T *aggregate_start_alloc;
			volatile COUNTER_T *aggregate_end_alloc;
			volatile COUNTER_T *aggregate_end;
			volatile COUNTER_T *aggregate_end_max;
			volatile COUNTER_T *aggregate_end_count;

			COUNTER_T *recv_start;
			COUNTER_T *recv_start_alloc;
			COUNTER_T *recv_end;
			
			RECV_T * recv_queues;
			RECV_T * aggregate_queues;
			volatile char * aggregate_maps;
					
			volatile int *stop;
			
			Agent(int _my_pe, int _n_pes, int _nodes_size, int _node_id, int _group_size, int _group_id,
				  int num_agg_queues, COUNTER_T cap, COUNTER_T *agg_start, COUNTER_T *agg_start_alloc, COUNTER_T *agg_end_alloc,
				  COUNTER_T *agg_end, COUNTER_T *agg_end_max, COUNTER_T *agg_end_count,
				  COUNTER_T * r_start, COUNTER_T *r_start_alloc, COUNTER_T *r_end, RECV_T *r_queues, RECV_T *agg_queues,
				  char * agg_maps, int *_stop): my_pe(_my_pe), n_pes(_n_pes), nodes_size(_nodes_size), node_id(_node_id),
				  group_size(_group_size), group_id(_group_id), num_aggregate_queues(num_agg_queues), capacity(cap), 
				  aggregate_start((volatile COUNTER_T *)agg_start), aggregate_start_alloc((volatile COUNTER_T *)agg_start_alloc), 
				  aggregate_end_alloc((volatile COUNTER_T *)agg_end_alloc), aggregate_end((volatile COUNTER_T *)agg_end),
				  aggregate_end_max((volatile COUNTER_T *)agg_end_max), aggregate_end_count((volatile COUNTER_T *)agg_end_count), recv_start(r_start), 
				  recv_start_alloc(r_start_alloc), recv_end(r_end), recv_queues(r_queues), aggregate_queues(agg_queues),
			      aggregate_maps((volatile char *)agg_maps), stop((volatile int *)_stop) {}		
			~Agent() {}
		};
	} //name space

	template<typename RECV_T, typename COUNTER_T, int INTER_BATCH_SIZE, int PADDING_SIZE=1>
	class Agent
	{
	public:
		int my_pe;
		int n_pes;
		int nodes_size;
		int node_id;
		int group_size;
		int group_id;

		int num_aggregate_queues;
		COUNTER_T capacity;
		COUNTER_T *aggregate_start;
		COUNTER_T *aggregate_start_alloc;
		COUNTER_T *aggregate_end_alloc;
		COUNTER_T *aggregate_end;
		COUNTER_T *aggregate_end_max;
		COUNTER_T *aggregate_end_count;

		COUNTER_T *recv_start;
		COUNTER_T *recv_start_alloc;
		COUNTER_T *recv_end;
		
		RECV_T * recv_queues;
		RECV_T * aggregate_queues;
		char * aggregate_maps;
				
		int *stop;

		template<typename LOCAL_T>
		Agent(DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE> &queue)
		{
			my_pe = queue.my_pe;
			n_pes = queue.n_pes;
			nodes_size = queue.nodes_size;
			node_id = queue.node_id;
			group_size = queue.group_size;
			group_id = queue.group_id;

			num_aggregate_queues = queue.num_aggregate_queues;
			capacity = queue.recv_capacity;
			aggregate_start = queue.start+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;
			aggregate_start_alloc = queue.start_alloc+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;
			aggregate_end_alloc = queue.end_alloc+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;
			aggregate_end = queue.end+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;
			aggregate_end_max = queue.end_max+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;
			aggregate_end_count = queue.end_count+(queue.num_local_queues+n_pes-1)*PADDING_SIZE;

			recv_start = queue.start+queue.num_local_queues*PADDING_SIZE;
			recv_start_alloc = queue.start_alloc+queue.num_local_queues*PADDING_SIZE;
			recv_end = queue.end+queue.num_local_queues*PADDING_SIZE;

			recv_queues = queue.recv_queues;
			aggregate_queues = queue.aggregate_queues;
			aggregate_maps = queue.aggregate_maps;

			CUDA_CHECK(cudaMalloc(&stop, sizeof(int)));
			CUDA_CHECK(cudaMemset(stop, 0, sizeof(int)));
		}

		~Agent() { CUDA_CHECK(cudaFree(stop)); }

		dev::Agent<RECV_T, COUNTER_T, INTER_BATCH_SIZE, PADDING_SIZE> deviceObject() {
			return dev::Agent<RECV_T, COUNTER_T, INTER_BATCH_SIZE, PADDING_SIZE>(
				  my_pe, n_pes, nodes_size, node_id, group_size, group_id, num_aggregate_queues,
				  capacity,aggregate_start, aggregate_start_alloc, aggregate_end_alloc,
				  aggregate_end, aggregate_end_max, aggregate_end_count, recv_start, 
				  recv_start_alloc, recv_end, recv_queues, aggregate_queues, aggregate_maps, stop);
		}
		
		void stopAgent(cudaStream_t stream=0) {
			int h_stop = 1;
			CUDA_CHECK(cudaMemcpyAsync(stop, &h_stop, sizeof(int), cudaMemcpyHostToDevice, stream));	
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}

		void resetAgent(cudaStream_t stream=0) {
			CUDA_CHECK(cudaMemsetAsync(stop, 0, sizeof(int), stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}

		template<int WAIT_TIMES>
		void launchAgent(cudaStream_t stream, RECV_T *temp_d);
	}; //class Agent


	template<int PADDING_SIZE, typename RECV_T, typename COUNTER_T>
	__forceinline__	__device__ void send_data(int aggregate_offset, COUNTER_T *aggregate_end, COUNTER_T *aggregate_start, 
							int my_pe, int group_size, RECV_T * recv_queues, RECV_T *aggregate_queues, COUNTER_T capacity, volatile COUNTER_T * remote_start, int flag)
	{
		size_t remote_offset = (aggregate_offset/group_size)<(my_pe/group_size)? my_pe-1:my_pe;
		int pe = aggregate_offset/group_size<my_pe/group_size? aggregate_offset:aggregate_offset+group_size;
		//printf("pe %d, tid %d, flag %d, agg_offset %d, agg/group_size %d, my_pe/grou_size %d, form < later? %d, remote_offset %d\n", 
		//		my_pe, threadIdx.x, flag, aggregate_offset, aggregate_offset/group_size, my_pe/group_size, int((aggregate_offset/group_size)<(my_pe/group_size)), remote_offset);
		//printf("pe %d,agg_offset %d, remote_offset %d, pe %d, end %d, start %d, size %d, flag %d\n", my_pe, aggregate_offset, remote_offset,  pe, 
		//aggregate_end[aggregate_offset], aggregate_start[aggregate_offset], aggregate_end[aggregate_offset]-aggregate_start[aggregate_offset], flag);
		if(aggregate_end[aggregate_offset] >= capacity) {
			nvshmem_quiet();
			COUNTER_T remote_s = *(remote_start+aggregate_offset*PADDING_SIZE);
			if(aggregate_end[aggregate_offset] > capacity+remote_s)
				printf("pe %d, recv local offset %d, value %d, aggregate_end %d, aggregate_start %d\n", my_pe, aggregate_offset, remote_s, aggregate_end[aggregate_offset], aggregate_start[aggregate_offset]);
			assert(aggregate_end[aggregate_offset] <= capacity+remote_s);
			if(aggregate_end[aggregate_offset]/capacity == aggregate_start[aggregate_offset]/capacity)
				nvshmem_putmem(recv_queues+remote_offset*capacity+(aggregate_start[aggregate_offset]&(capacity-1)),
                        aggregate_queues+aggregate_offset*capacity+(aggregate_start[aggregate_offset]&(capacity-1)),
                        (aggregate_end[aggregate_offset]-aggregate_start[aggregate_offset])*sizeof(RECV_T), pe);
			else {
				nvshmem_putmem(recv_queues+remote_offset*capacity+(aggregate_start[aggregate_offset]&(capacity-1)),
                        aggregate_queues+aggregate_offset*capacity+(aggregate_start[aggregate_offset]&(capacity-1)),
						(capacity - (aggregate_start[aggregate_offset]&(capacity-1)))*sizeof(RECV_T), pe);
				nvshmem_putmem(recv_queues+remote_offset*capacity, aggregate_queues+aggregate_offset*capacity,
                        (aggregate_end[aggregate_offset]&(capacity-1))*sizeof(RECV_T), pe);
			}
		}
		else
		nvshmem_putmem(recv_queues+remote_offset*capacity+aggregate_start[aggregate_offset],
						aggregate_queues+aggregate_offset*capacity+aggregate_start[aggregate_offset], 
						(aggregate_end[aggregate_offset]-aggregate_start[aggregate_offset])*sizeof(RECV_T), pe);
	}

	template<int PADDING_SIZE, typename COUNTER_T>
	__forceinline__	__device__ void update_remote_end(int aggregate_offset, COUNTER_T *recv_end, COUNTER_T * aggregate_start, COUNTER_T *aggregate_end,
					 int my_pe, int group_size, COUNTER_T *aggregate_start_alloc)
	{
		int remote_offset = aggregate_offset/group_size<my_pe/group_size? my_pe-1:my_pe;
		int pe = aggregate_offset/group_size<my_pe/group_size? aggregate_offset:aggregate_offset+group_size;
		nvshmem_fence();
		nvshmem_p_wraper(recv_end+remote_offset*PADDING_SIZE, aggregate_end[aggregate_offset], pe);
		aggregate_start[aggregate_offset] = aggregate_end[aggregate_offset];
		*(aggregate_start_alloc+aggregate_offset*PADDING_SIZE) = aggregate_end[aggregate_offset];
	}

	template<int PADDING_SIZE, typename RECV_T, typename COUNTER_T>
	__forceinline__	__device__ void send_update_remote_end(int aggregate_offset, COUNTER_T *recv_end, COUNTER_T aggregate_start, COUNTER_T aggregate_end,
					 int my_pe, int group_size, COUNTER_T *aggregate_start_alloc, RECV_T *recv_queues, RECV_T *aggregate_queues, COUNTER_T capacity, volatile COUNTER_T *remote_start)
	{
		size_t remote_offset = aggregate_offset/group_size<my_pe/group_size? my_pe-1:my_pe;
		int pe = aggregate_offset/group_size<my_pe/group_size? aggregate_offset:aggregate_offset+group_size;

		if(aggregate_end >= capacity) {
			COUNTER_T remote_s = *(remote_start+aggregate_offset*PADDING_SIZE);
			assert(aggregate_end < capacity+remote_s);
			if(aggregate_start/capacity == aggregate_end/capacity)
				nvshmem_putmem(recv_queues+remote_offset*capacity+(aggregate_start&(capacity-1)),
						aggregate_queues+aggregate_offset*capacity+(aggregate_start&(capacity-1)),
						(aggregate_end-aggregate_start)*sizeof(RECV_T), pe);
			else {
				nvshmem_putmem(recv_queues+remote_offset*capacity+(aggregate_start&(capacity-1)),
							aggregate_queues+aggregate_offset*capacity+(aggregate_start&(capacity-1)),
							(capacity-(aggregate_start&(capacity-1)))*sizeof(RECV_T), pe);
				nvshmem_putmem(recv_queues+remote_offset*capacity, aggregate_queues+aggregate_offset*capacity,
							(aggregate_end&(capacity-1))*sizeof(RECV_T), pe);
			}
		}
		else
		nvshmem_putmem(recv_queues+remote_offset*capacity+aggregate_start,
						aggregate_queues+aggregate_offset*capacity+aggregate_start,
						(aggregate_end-aggregate_start)*sizeof(RECV_T), pe);
		nvshmem_fence();
		nvshmem_p_wraper(recv_end+remote_offset*PADDING_SIZE, aggregate_end, pe);
		*(aggregate_start_alloc+aggregate_offset*PADDING_SIZE) = aggregate_end;
	}

	enum SEND_FLAG { ENOUGH, WAIT_ENOUGH, INSUFFICIENT };
	template<int WAIT_TIMES, typename RECV_T, typename COUNTER_T, int INTER_BATCH_SIZE, int PADDING_SIZE>
	__global__ void agent_kernel(dev::Agent<RECV_T, COUNTER_T, INTER_BATCH_SIZE, PADDING_SIZE> agent, RECV_T *temp_d)
	{
		if(threadIdx.x==0) printf("start Agent at PE %d\n", agent.my_pe);
		__shared__ int stop;
		extern __shared__ COUNTER_T shared_space[];
		COUNTER_T *aggregate_start = shared_space;
		COUNTER_T *aggregate_end = aggregate_start + agent.num_aggregate_queues;
		COUNTER_T *aggregate_attempt_end = aggregate_end + agent.num_aggregate_queues;
		COUNTER_T *aggregate_wait_time = aggregate_attempt_end + agent.num_aggregate_queues;
		COUNTER_T *aggregate_check_start = aggregate_wait_time + agent.num_aggregate_queues;
		COUNTER_T *aggregate_end_alloc = aggregate_check_start + agent.num_aggregate_queues;

		if(threadIdx.x < agent.num_aggregate_queues) {
			aggregate_start[threadIdx.x] = *(agent.aggregate_start_alloc+threadIdx.x*PADDING_SIZE);
			aggregate_end[threadIdx.x] = *(agent.aggregate_end+threadIdx.x*PADDING_SIZE);
			aggregate_end_alloc[threadIdx.x] = *(agent.aggregate_end_alloc+threadIdx.x*PADDING_SIZE);
			aggregate_check_start[threadIdx.x] = aggregate_end[threadIdx.x];
			aggregate_wait_time[threadIdx.x] = 0;
		}
		else if(threadIdx.x == agent.num_aggregate_queues) stop = *(agent.stop);
		__syncthreads();	

		while(stop == 0) {
			if(LANE_ == 0) stop = *(agent.stop);
			SEND_FLAG flag = INSUFFICIENT;
			COUNTER_T agg_end = 0;
			//COUNTER_T agg_end_max = 0;
			//COUNTER_T agg_end_count = 0;
			uint64_t agg_end_max_count = 0;
			COUNTER_T agg_end_alloc = 0;
			if(threadIdx.x < agent.num_aggregate_queues) {
				agg_end = *(agent.aggregate_end+threadIdx.x*PADDING_SIZE);
				//agg_end_max = *(agent.aggregate_end_max+threadIdx.x*PADDING_SIZE);
				//agg_end_count = *(agent.aggregate_end_count+threadIdx.x*PADDING_SIZE);
				agg_end_max_count = *((uint64_t *)(agent.aggregate_end_max+threadIdx.x*PADDING_SIZE));
				agg_end_alloc = *(agent.aggregate_end_alloc+threadIdx.x*PADDING_SIZE);
				if(aggregate_end_alloc[threadIdx.x] > agent.capacity) {
				int remote_pe = agent.my_pe/agent.group_size > threadIdx.x/agent.group_size? threadIdx.x:threadIdx.x+agent.group_size;
				int remote_offset = agent.my_pe/agent.group_size > threadIdx.x/agent.group_size? agent.my_pe-1:agent.my_pe;
				nvshmem_get32_nbi((COUNTER_T *)(agent.aggregate_start+threadIdx.x*PADDING_SIZE), agent.recv_start+remote_offset*PADDING_SIZE, 1, remote_pe);
				}
				// load agg_end, compute end_attempt or wait++
				aggregate_attempt_end[threadIdx.x] = align_down_yc(aggregate_end[threadIdx.x], INTER_BATCH_SIZE);
				if((long long)(aggregate_attempt_end[threadIdx.x])-INTER_BATCH_SIZE >= (long long)(align_down_yc(aggregate_start[threadIdx.x], INTER_BATCH_SIZE))) {
					flag = ENOUGH;
					aggregate_wait_time[threadIdx.x] = 0;
				}
				else if(aggregate_end_alloc[threadIdx.x] > aggregate_start[threadIdx.x]) {
					aggregate_attempt_end[threadIdx.x] = min(align_down_yc(aggregate_end_alloc[threadIdx.x], INTER_BATCH_SIZE), align_down_yc(aggregate_start[threadIdx.x]+131072, INTER_BATCH_SIZE));
					if(aggregate_attempt_end[threadIdx.x] > aggregate_start[threadIdx.x]) {
						flag = WAIT_ENOUGH;
						aggregate_wait_time[threadIdx.x] = 0;
					}
					else {
						aggregate_wait_time[threadIdx.x]++;
						if(aggregate_wait_time[threadIdx.x] >= WAIT_TIMES) {
							flag = WAIT_ENOUGH;
							aggregate_attempt_end[threadIdx.x] = aggregate_end_alloc[threadIdx.x];
						}
					}
					__threadfence_block();
				}
				if(flag == ENOUGH) {
					send_data<PADDING_SIZE>(threadIdx.x, aggregate_attempt_end, aggregate_start, agent.my_pe, agent.group_size, agent.recv_queues, agent.aggregate_queues,agent.capacity, (volatile COUNTER_T *)agent.aggregate_start, int(flag));
				}
			}
			//printf("pe %d, tid %d, remote_pe %d, end %d, start %d, attemp_end %d, enough %d\n", agent.my_pe, threadIdx.x, 0, aggregate_end[0], aggregate_start[0], aggregate_attempt_end[0], flag);
			unsigned enough_mask = __ballot_sync(0xffffffff, flag==WAIT_ENOUGH);
			while(enough_mask > 0) {
				unsigned temp_mask = enough_mask;
				int end_iter = __popc(enough_mask);
				for(int iter=0; iter<end_iter; iter++) {
					int aggregate_offset = WARPID*32+__ffs(temp_mask)-1;
					//printf("tid %d, enough_mask %x, temp_mask %x, iter %d, aggregate_offset %d\n", threadIdx.x, enough_mask, temp_mask, iter, aggregate_offset);
					COUNTER_T map_end=align_down_yc(aggregate_attempt_end[aggregate_offset]+31,32)/32;
					COUNTER_T map_start=align_down_yc(aggregate_check_start[aggregate_offset],32)/32;
					//printf("tid %d, agg_offset %d, attemp_end %d, check_start %d, map_start %d, map_end %d\n", threadIdx.x, aggregate_offset, aggregate_attempt_end[aggregate_offset], aggregate_check_start[aggregate_offset], 
							//map_start, map_end);
					for(int chunk=map_start; chunk<map_end; chunk+=32) {
						int arrive_size = 0;
						bool arrive = false;
						if(chunk+LANE_ < map_end) {
							longlong4 map = ((longlong4 *)(agent.aggregate_maps+aggregate_offset*(align_up_yc(agent.capacity, 32))*4) )[chunk+LANE_];
							unsigned map1 = ((map.x & 0x1) | ((map.x & 0x100) >> 7) | ((map.x & 0x10000) >> 14) | ((map.x & 0x1000000)>>21) | ((map.x & 0x100000000)>>28) | ((map.x & 0x10000000000)>>35) 
											| ((map.x & 0x1000000000000)>>42) | ((map.x & 0x100000000000000) >> 49));
							unsigned map2 = ((map.y & 0x1) | ((map.y & 0x100) >> 7) | ((map.y & 0x10000) >> 14) | ((map.y & 0x1000000)>>21) | ((map.y & 0x100000000)>>28) | ((map.y & 0x10000000000)>>35)
											| ((map.y & 0x1000000000000)>>42) | ((map.y & 0x100000000000000) >> 49));
							unsigned map3 = ((map.z & 0x1) | ((map.z & 0x100) >> 7) | ((map.z & 0x10000) >> 14) | ((map.z & 0x1000000)>>21) | ((map.z & 0x100000000)>>28) | ((map.z & 0x10000000000)>>35)
											| ((map.z & 0x1000000000000)>>42) | ((map.z & 0x100000000000000) >> 49));
							unsigned map4 = ((map.w & 0x1) | ((map.w & 0x100) >> 7) | ((map.w & 0x10000) >> 14) | ((map.w & 0x1000000)>>21) | ((map.w & 0x100000000)>>28) | ((map.w & 0x10000000000)>>35)
											| ((map.w & 0x1000000000000)>>42) | ((map.w & 0x100000000000000) >> 49));
							//printf("tid %d, id %d, map %lx %lx %lx %lx, map1 %d, map2 %d, map3 %d, map4 %d\n", threadIdx.x, chunk+LANE_, map.x, map.y, map.z, map.w, map1, map2, map3, map4);
							unsigned arrive_map = (map1 | (map2 << 8) | (map3 << 16) | (map4 << 24));
							arrive = (arrive_map == 0xffffffff);
							arrive_size = __ffs(~arrive_map)-1;
						}
						unsigned arrive_mask = __ballot_sync(0xffffffff, arrive);
						arrive_mask = ~(arrive_mask);
						if(arrive_mask == 0) {
							if(LANE_ == 0) {
								aggregate_check_start[aggregate_offset] = align_down_yc(aggregate_check_start[aggregate_offset],32)+1024;
								__threadfence_block();
							}
						}
						else {
							if(LANE_ == __ffs(arrive_mask)-1) {
								aggregate_check_start[aggregate_offset] = align_down_yc(aggregate_check_start[aggregate_offset],32)+(LANE_*32);
								if(chunk+LANE_ == map_end-1 && aggregate_attempt_end[aggregate_offset] != align_down_yc(aggregate_attempt_end[aggregate_offset], 32))
									aggregate_check_start[aggregate_offset] += arrive_size;
								__threadfence_block();
							}
							break;
						}
						__syncwarp();
					}
					__syncwarp();
					if(aggregate_check_start[aggregate_offset] >= aggregate_attempt_end[aggregate_offset])
					{
						enough_mask = (enough_mask & (~(1<<(__ffs(temp_mask)-1))));
					}
					//printf("tid %d, check_start[%d] %d, attemp_end[%d] %d\n", threadIdx.x, aggregate_offset, aggregate_check_start[aggregate_offset], aggregate_offset, aggregate_attempt_end[aggregate_offset]);
					temp_mask = temp_mask & (~(1<<(__ffs(temp_mask)-1)));
				} // for __popc(temp_mask)
			} // while(enough_mask > 0)
			if(flag == WAIT_ENOUGH)
				send_data<PADDING_SIZE>(threadIdx.x, aggregate_attempt_end, aggregate_start, agent.my_pe, agent.group_size, agent.recv_queues, agent.aggregate_queues, agent.capacity, (volatile COUNTER_T *)agent.aggregate_start, int(flag));
			if(flag != INSUFFICIENT)
				update_remote_end<PADDING_SIZE>(threadIdx.x, agent.recv_end, aggregate_start, aggregate_attempt_end, agent.my_pe, agent.group_size, (COUNTER_T *)agent.aggregate_start_alloc);
			__syncwarp();	
			if(threadIdx.x < agent.num_aggregate_queues) {
				COUNTER_T agg_end_max = COUNTER_T(agg_end_max_count);
				COUNTER_T agg_end_count = (COUNTER_T)(agg_end_max_count >> 32);
				if(agg_end_max == agg_end_count && agg_end_max > agg_end)
					agg_end = agg_end_max;
				aggregate_end[threadIdx.x] = max(aggregate_end[threadIdx.x], agg_end);
				aggregate_end_alloc[threadIdx.x] = agg_end_alloc;
				aggregate_check_start[threadIdx.x] = max(aggregate_check_start[threadIdx.x], aggregate_end[threadIdx.x]);
			}
			__threadfence_block();
			__syncwarp();
		} // while(stop)

		// unlikely pe_range need loop over since numThread allocated based on n_pes. However numThread has max value 1024
		if(threadIdx.x< agent.num_aggregate_queues) {
			SEND_FLAG flag = INSUFFICIENT;
			COUNTER_T agg_end_alloc = *(agent.aggregate_end_alloc+threadIdx.x*PADDING_SIZE);
			if(aggregate_start[threadIdx.x] < agg_end_alloc) flag = ENOUGH;
			if(flag == ENOUGH)
				send_update_remote_end<PADDING_SIZE>(threadIdx.x, agent.recv_end, aggregate_start[threadIdx.x], agg_end_alloc, agent.my_pe, agent.group_size, (COUNTER_T *)agent.aggregate_start_alloc,
													agent.recv_queues, agent.aggregate_queues, agent.capacity, (volatile COUNTER_T *)agent.aggregate_start);

		} // if(threadIdx < num_aggregate_queues)
	}
	
	template<typename RECV_T, typename COUNTER_T, int INTER_BATCH_SIZE, int PADDING_SIZE>
	template<int WAIT_TIMES>
	void Agent<RECV_T, COUNTER_T, INTER_BATCH_SIZE, PADDING_SIZE>::launchAgent(cudaStream_t stream, RECV_T *temp_d) {
		int shared_mem = sizeof(COUNTER_T)*(num_aggregate_queues*6);
		printf("launch with 1 block with %d threads\n", align_up_yc(num_aggregate_queues, 32));
		agent_kernel<WAIT_TIMES><<<1, align_up_yc(num_aggregate_queues, 32), shared_mem, stream>>>(this->deviceObject(), temp_d);
	}
	} //MAXCOUNT
} // Atos namespace
