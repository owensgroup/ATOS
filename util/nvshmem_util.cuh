#ifndef NVSHMEM_UTIL
#define NVSHMEM_UTIL
__device__ int clockrate;
void nvshm_mpi_init(int &global_id, int &global_size, int &group_id, int &group_size,
					int &local_id, int &local_size, int *argc, char *** argv)
{
    int rank, nranks;
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    global_id = nvshmem_my_pe();
    global_size = nvshmem_n_pes();
    assert(global_id == rank);
    assert(global_size == nranks);

	MPI_Comm nodeComm, masterComm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, global_id,
						MPI_INFO_NULL, &nodeComm);
 	MPI_Comm_rank(nodeComm, &local_id);
	MPI_Comm_split(MPI_COMM_WORLD, local_id, global_id, &masterComm);
	MPI_Comm_rank(masterComm, &group_id);
	MPI_Comm_size(masterComm, &group_size);
	MPI_Comm_size(nodeComm, &local_size);

    cudaDeviceProp prop;
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    //assert(my_pe < dev_count);
    //CUDA_CHECK(cudaSetDevice(my_pe));
	CUDA_CHECK(cudaSetDevice(global_id%dev_count));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, global_id%dev_count));
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice));
    //printf("my_pe %d, n_pes %d, device name %s, bus id %d, clockrate %d\n", 
    //    my_pe, n_pes, prop.name, prop.pciBusID, prop.clockRate);
	printf("global [%d of %d] group [%d of %d], local[%d  of %d]\n", global_id, global_size, group_id, group_size, local_id, local_size);
  	printf("[%d of %d] has GPU %d, set on %d, device name %s, bus id %d, clockrate %d\n", global_id, global_size, dev_count, global_id%dev_count, prop.name, prop.pciBusID, prop.clockRate);
	MPI_Comm_free(&nodeComm);
	MPI_Comm_free(&masterComm);
}

void nvshm_mpi_finalize()
{
    nvshmem_barrier_all();
    nvshmem_finalize();
    MPI_Finalize();
}

template<typename T>
__forceinline__ __device__ void nvshmem_64bits_block_wraper(T *dest, T *source, size_t elem, int peer){
	if(sizeof(double)/sizeof(T) == 1)
		nvshmemx_double_put_block((double *)dest, (double *)source, elem, peer);
    else if(((unsigned long)dest & 7) == 0 && ((unsigned long)source & 7) == 0) {
        int chunk_elem = elem/(sizeof(double)/sizeof(T));
        int left_elem = elem - chunk_elem*(sizeof(double)/sizeof(T));
        nvshmemx_double_put_block((double *)dest, (double *)(source), chunk_elem, peer);
        if(left_elem)
            nvshmem_block_wraper(dest+chunk_elem*(sizeof(double)/sizeof(T)), source+chunk_elem*(sizeof(double)/sizeof(T)), left_elem, peer);
    }
    else if(((unsigned long)dest & 7) == ((unsigned long)source & 7)) {
        size_t size = ((unsigned long)(dest) & 7)/sizeof(T);
        if(threadIdx.x < size)
            nvshmem_p_wraper(dest+threadIdx.x, *(source+threadIdx.x), peer);
        int chunk_elem = (elem-size)/(sizeof(double)/sizeof(T));
        int left_elem = elem - size - chunk_elem*(sizeof(double)/sizeof(T));
        nvshmemx_double_put_block((double *)(dest+size), (double *)(source+size), chunk_elem, peer);
        if(left_elem)
            nvshmem_block_wraper(dest+size+chunk_elem*(sizeof(double)/sizeof(T)), source+size+chunk_elem*(sizeof(double)/sizeof(T)), left_elem, peer);
    }
    else assert(false);
}

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
#endif
