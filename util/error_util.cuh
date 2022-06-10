#ifndef ERROR_UTIL
#define ERROR_UTIL
 #define CUDA_CHECK(call) {                                    \
          cudaError_t err =                                                         call;                                                    \
          if( cudaSuccess != err) {                                                 \
                       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",         \
                                                __FILE__, __LINE__, cudaGetErrorString( err) );               \
                       exit(EXIT_FAILURE);                                                   \
                   } }

 #define MALLOC_CHECK(call) {     \
          if(call==NULL) \
          {              \
                       std::cout << "malloc fail in file: "<<__FILE__ << " in line "<<       __LINE__<<".\n";     \
                       exit(1);   \
                   }              \
      }
#define SERIALIZE_PRINT(my_pe, n_pes, call) {  \
    for(int i=0; i<n_pes; i++) {               \
        if(my_pe == i) {                       \
            std::cout << "[PE "<< my_pe << "]\n"; \
            call;                              \
        }                                      \
        nvshmem_barrier_all();                 \
    }                                          \
}

#define MPI_CHECK(stmt)                                 \
do {                                                    \
    int result = (stmt);                                \
    if (MPI_SUCCESS != result) {                        \
        fprintf(stderr, "[%s:%d] MPI failed with error %d \n",\
         __FILE__, __LINE__, result);                   \
        exit(-1);                                       \
    }                                                   \
} while (0)

#endif
