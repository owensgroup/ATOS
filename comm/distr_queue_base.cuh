#ifndef DISTRIBUTED_QUEUE_BASE
#define DISTRIBUTED_QUEUE_BASE

namespace Atos{
	namespace MAXCOUNT{
		namespace dev{
			template<typename RECV_T, typename LOCAL_T, typename COUNTER_T=uint32_t, int PADDING_SIZE=32>
			class DistributedQueueBase
			{
			public:
				int n_pes;
                int my_pe;
                int group_size;

                COUNTER_T local_capacity;
                COUNTER_T recv_capacity;

                int min_iter;
                int num_local_queues;
                int num_aggregate_queues;
                int total_num_queues;

                LOCAL_T *local_queues;
                RECV_T *recv_queues;
                RECV_T *aggregate_queues;
                volatile char *aggregate_maps;

                volatile COUNTER_T *start, *start_alloc, *end_alloc, *end, *end_max, *end_count;
                volatile int *stop;

				static const int padding = PADDING_SIZE;

				DistributedQueueBase() {}

				DistributedQueueBase(int _n_pes, int _my_pe, int _group_size, LOCAL_T *l_q, RECV_T *r_q, RECV_T *a_q, char * a_m, COUNTER_T l_capacity, COUNTER_T r_capacity,
                COUNTER_T *_start, COUNTER_T *_start_alloc, COUNTER_T *_end_alloc, COUNTER_T *_end, COUNTER_T *_end_max, COUNTER_T *_end_count, int *_stop, int _num_l_q,
                int _num_a_q, int _num_iter): n_pes(_n_pes), my_pe(_my_pe), group_size(_group_size), local_queues(l_q), recv_queues(r_q), aggregate_queues(a_q), 
                num_local_queues(_num_l_q), num_aggregate_queues(_num_a_q), min_iter(_num_iter), local_capacity(l_capacity), recv_capacity(r_capacity) 
                {
					aggregate_maps = (volatile char *)a_m;
                    start = (volatile COUNTER_T *)_start;
                    start_alloc = (volatile COUNTER_T *)_start_alloc;
                    end_alloc = (volatile COUNTER_T *)_end_alloc;
                    end = (volatile COUNTER_T *)_end;
                    end_max = (volatile COUNTER_T *)_end_max;
                    end_count = (volatile COUNTER_T *)_end_count;
                    stop = (volatile int *)_stop;

                    total_num_queues=num_local_queues+n_pes-1+num_aggregate_queues;
                }
			};
		} //dev

		template<typename RECV_T, typename LOCAL_T, typename COUNTER_T=uint32_t, int PADDING_SIZE=32>
		class DistributedQueueBase 
		{
		public:
			int n_pes;
        	int my_pe;
        	int nodes_size;
        	int node_id;
        	int group_size;
        	int group_id;

        	LOCAL_T *local_queues;
        	RECV_T *recv_queues;
        	RECV_T *aggregate_queues;
        	char *aggregate_maps;

        	COUNTER_T local_capacity;
        	COUNTER_T recv_capacity;

        	int num_local_queues;
        	int num_aggregate_queues;
			int total_num_queues;

			int min_iter;
			COUNTER_T * counters;
        	int num_counters = 7;
        	COUNTER_T *start, *start_alloc, *end, *end_alloc, *end_max, *end_count;
        	int *stop;

			static const int padding = PADDING_SIZE;

			DistributedQueueBase() {}
			~DistributedQueueBase() { release(); }

			__host__ void baseInit(int _n_pes, int _my_pe, int _group_id, int _group_size, int local_id, int local_size, 
			COUNTER_T l_capacity, COUNTER_T r_capacity, int m_iter, int l_queues = 1, bool PRINT_INFO = false)
			{
				n_pes = _n_pes;
            	my_pe = _my_pe;
            	nodes_size= _group_size;
            	node_id = _group_id;
            	group_size = local_size;
            	group_id = local_id;
            	
            	local_capacity = l_capacity;
            	recv_capacity = r_capacity;
				assert((recv_capacity & (recv_capacity-1)) == 0);
           	
				min_iter = m_iter;
            	num_local_queues = l_queues;
            	num_aggregate_queues = (nodes_size-1)*group_size;
				total_num_queues = num_local_queues+n_pes-1+num_aggregate_queues;
            	printf("num_aggregate_queues %d, local_queues %d, total_queues %d\n", num_aggregate_queues, num_local_queues, total_num_queues);

            	alloc(PRINT_INFO); 
			}
			
			__host__ void reset(cudaStream_t stream=0) const
        	{
        	    CUDA_CHECK(cudaMemsetAsync((void *)local_queues, 0xffffffff, sizeof(LOCAL_T)*local_capacity*num_local_queues, stream));
        	    CUDA_CHECK(cudaMemsetAsync((void *)recv_queues, 0xffffffff, sizeof(RECV_T)*recv_capacity*(n_pes-1), stream));
        	    CUDA_CHECK(cudaMemsetAsync((void *)aggregate_queues, 0xffffffff, sizeof(RECV_T*)*recv_capacity*num_aggregate_queues, stream));
				CUDA_CHECK(cudaMemsetAsync((void *)aggregate_maps, 0, sizeof(char)*align_up_yc(recv_capacity,32)*4*num_aggregate_queues, stream));
        	    CUDA_CHECK(cudaMemsetAsync((void *)counters, 0, sizeof(COUNTER_T)*num_counters*total_num_queues*PADDING_SIZE, stream));
        	}

			__host__ void print() const
        	{
            	COUNTER_T check[6*total_num_queues*PADDING_SIZE];
            	CUDA_CHECK(cudaMemcpy(check, start, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));
            	CUDA_CHECK(cudaMemcpy(check+1*total_num_queues*PADDING_SIZE, start_alloc, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));
            	CUDA_CHECK(cudaMemcpy(check+2*total_num_queues*PADDING_SIZE, end_alloc, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));
            	CUDA_CHECK(cudaMemcpy(check+3*total_num_queues*PADDING_SIZE, end, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));
            	CUDA_CHECK(cudaMemcpy(check+4*total_num_queues*PADDING_SIZE, end_max, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));
            	CUDA_CHECK(cudaMemcpy(check+5*total_num_queues*PADDING_SIZE, end_count, sizeof(COUNTER_T)*total_num_queues*PADDING_SIZE, cudaMemcpyDeviceToHost));

            	for(int i=0; i<num_local_queues; i++) {
            	    if(check[i*PADDING_SIZE]== check[2*total_num_queues*PADDING_SIZE+i*PADDING_SIZE] &&
            	        check[2*total_num_queues*PADDING_SIZE+i*PADDING_SIZE] == check[3*total_num_queues*PADDING_SIZE+i*PADDING_SIZE])
            	        printf("PE %d, Local Queue %d, capacity: %zd, end = start: %zd, start_alloc: %zd\n", my_pe, i, size_t(local_capacity), size_t(check[i*PADDING_SIZE]),
            	                size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]));
            	    else printf("PE %d, Local Queue %d, capacity: %zd end_alloc: %zd, end: %zd, end_max: %zd, end_count: %zd, start: %zd, start_alloc: %zd\n", my_pe,
            	            i, size_t(local_capacity), size_t(check[2*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[3*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[4*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[5*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]));
            	}
				for(int i=num_local_queues; i<num_local_queues+n_pes-1; i++) {
            	    if(check[i*PADDING_SIZE]== check[3*total_num_queues*PADDING_SIZE+i*PADDING_SIZE])
            	        printf("PE %d, Recv Queue %d, capacity: %zd, end = start: %zd, start_alloc: %zd\n", my_pe, i-num_local_queues, size_t(recv_capacity), size_t(check[i*PADDING_SIZE]),
            	                size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]));
            	    else printf("PE %d, Recv Queue %d, capacity: %zd end_alloc: %zd, end: %zd, start: %zd, start_alloc: %zd\n", my_pe,
            	            i-num_local_queues, size_t(recv_capacity), size_t(check[2*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[3*total_num_queues*PADDING_SIZE+i*PADDING_SIZE]),
            	            size_t(check[i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]));
            	}
            	for(int i=num_local_queues+n_pes-1; i<num_local_queues+n_pes-1+num_aggregate_queues; i++) {
            	    if(check[i*PADDING_SIZE]!=0) printf("PE %d, aggregation queue not empty: protect %zd, start %zd, end %zd\n", my_pe, size_t(check[i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE]));
            	    else if(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE] == check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE] &&
            	            check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE] == check[total_num_queues*PADDING_SIZE*3+i*PADDING_SIZE])
            	        printf("PE %d, aggregation queue empty: start = end %zd\n", my_pe, size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]));
            	    else if(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE] == check[total_num_queues*PADDING_SIZE*3+i*PADDING_SIZE])
            	        printf("PE %d, aggregation queue %d: start %zd, end=end_alloc %zd\n", my_pe, i-(num_local_queues+n_pes-1),
            	        size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE]));
            	    else if(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE] == check[total_num_queues*PADDING_SIZE*4+i*PADDING_SIZE] &&
            	            check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE] == check[total_num_queues*PADDING_SIZE*5+i*PADDING_SIZE] )
            	        printf("PE %d, aggregation queue %d: start %zd, end %zd, end_alloc=end_max=end_count %zd\n", my_pe, i-(num_local_queues+n_pes-1),
            	            size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*PADDING_SIZE*3+i*PADDING_SIZE]),
            	            size_t(check[total_num_queues*PADDING_SIZE*2+i*PADDING_SIZE]));
            	    else printf("PE %d, aggregation bug: protect %zd, start %zd, end %zd, end_alloc %zd, end_max %zd, end_count %zd\n", my_pe, size_t(check[i*PADDING_SIZE]),  size_t(check[total_num_queues*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*3*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*2*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*4*PADDING_SIZE+i*PADDING_SIZE]), size_t(check[total_num_queues*5*PADDING_SIZE+i*PADDING_SIZE]));
            	}

        	}

			dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>
            deviceObject() const {
                return dev::DistributedQueueBase<RECV_T, LOCAL_T, COUNTER_T, PADDING_SIZE>
                (n_pes, my_pe, group_size, local_queues,  recv_queues, aggregate_queues, aggregate_maps, local_capacity, recv_capacity,
                start, start_alloc, end_alloc, end, end_max, end_count, stop, num_local_queues, num_aggregate_queues, min_iter);
            }

		private:
        	void alloc(bool PRINT_INFO = false)
        	{
        	    if(local_capacity+recv_capacity <= 0) return;
        	    if(PRINT_INFO)
        	            std::cout << "pe "<< my_pe << " called distributed queue base allocator\n";
        	    CUDA_CHECK(cudaMalloc(&local_queues, sizeof(LOCAL_T)*local_capacity*num_local_queues));
        	    CUDA_CHECK(cudaMemset(local_queues, 0xffffffff, sizeof(LOCAL_T)*local_capacity*num_local_queues));

        	    recv_queues = (RECV_T *)nvshmem_malloc(sizeof(RECV_T)*recv_capacity*(n_pes-1));
        	    CUDA_CHECK(cudaMemset(recv_queues, 0xffffffff, sizeof(RECV_T)*recv_capacity*(n_pes-1)));

        	    aggregate_queues = (RECV_T *)nvshmem_malloc(sizeof(RECV_T)*recv_capacity*num_aggregate_queues);
        	    CUDA_CHECK(cudaMemset(aggregate_queues, 0xffffffff, sizeof(RECV_T)*recv_capacity*num_aggregate_queues));

				CUDA_CHECK(cudaMalloc(&aggregate_maps, sizeof(char)*align_up_yc(recv_capacity,32)*4*num_aggregate_queues));
				CUDA_CHECK(cudaMemset(aggregate_maps, 0, sizeof(char)*align_up_yc(recv_capacity,32)*4*num_aggregate_queues));
				printf("pe %d, map size %d\n", my_pe, align_up_yc(recv_capacity,32)*4);

        	    counters = (COUNTER_T *)nvshmem_malloc(sizeof(COUNTER_T)*num_counters*PADDING_SIZE*total_num_queues);
        	    // local then remote counter then aggregate counter
        	    // local counter includes: start, start_alloc, end_alloc
        	    // recve counter includes: start, start_alloc, end_alloc
        	    // aggreaget counter include: protect(prevent overpop), start_alloc, end_alloc
        	    start = counters;
        	    start_alloc = (counters+1*PADDING_SIZE*total_num_queues);
        	    end_alloc = (counters+2*PADDING_SIZE*total_num_queues);
        	    end = (counters+3*PADDING_SIZE*total_num_queues);
        	    end_max = (counters+4*PADDING_SIZE*total_num_queues);
        	    end_count = end_max+1;

        	    stop = (int *)(counters+6*PADDING_SIZE*total_num_queues);
        	}

			void release(bool PRINT_INFO = false) {
            	if(local_capacity+recv_capacity <= 0) return;
            	if(PRINT_INFO)
            	    std::cout << "pe "<< my_pe << " call distributed queue base destructor\n";
				if(local_queues!=NULL)
            	CUDA_CHECK(cudaFree(local_queues));
				if(aggregate_maps!=NULL)
            	CUDA_CHECK(cudaFree(aggregate_maps));

            	//nvshmem_free(aggregate_queues);
            	//nvshmem_free(recv_queues);
            	//nvshmem_free(counters);
        	}
        };
	} //MAXCOUNT
} //Atos

#endif
