#ifndef CUDA_COMMON
#define CUDA_COMMON
#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)
#define LANE_ ((threadIdx.x+blockIdx.x*blockDim.x)&31)
#define MAX_U32 (~(uint32_t)0)
#define MAX_SZ (~(size_t)0)
#define FULLMASK_64BITS (0xffffffffffffffff)
#define FULLMASK_32BITS (0xffffffff)

__device__ unsigned int lanemask_lt() {
       int lane = threadIdx.x & 31;
          return (1<<lane) - 1;;
}

__device__ uint get_smid(void) {
    uint ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}

__device__ uint32_t ld_gbl_cg(const uint32_t *addr) {
    uint32_t return_value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
    return return_value;
}

__device__ uint32_t ld_gbl_cv(const uint32_t *addr) {
    uint32_t return_value;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
    return return_value;
}

__device__ int ld_gbl_cg(const int *addr) {
    int return_value;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(return_value) : "l"(addr));
    return return_value;
}

#if defined(_WIN64) || defined(__LP64__)
# define PTR_CONSTRAINT "l"
#else
# define PTR_CONSTRAINT "r"
#endif
__device__ int isShared(void *ptr)
{
    int res;
    asm("{"
        ".reg .pred p;\n\t"
        "isspacep.shared p, %1;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}" :
        "=r"(res): PTR_CONSTRAINT(ptr));
        return res;
}

template<typename T, typename Y>
__global__ void setKernel(T *data, Y size, T value) {
    for(Y i = TID; i<size; i+=gridDim.x*blockDim.x)
        data[i] = value;
}

template<typename T, typename Y>
void set(T *data, Y size, T value, cudaStream_t stream=0)
{
    setKernel<<<160, 512, 0 , stream>>>(data, size, value);
}
 
#endif

#ifndef ROUND_UTIL
#define ROUND_UTIL
template<class Size>
__device__ __host__ Size roundup_power2(Size num)
{
    if(num && !(num&(num-1)))
        return num;
    num--;
    for(int i=1; i<=sizeof(Size)*4; i=i*2)
        num |= num>>i;
    return ++num;
}

template<class Size>
__device__ __host__ Size rounddown_power2(Size num)
{
    if(num && !(num&(num-1)))
        return num;
    num--;
    for(int i=1; i<=sizeof(Size)*4; i=i*2)
        num |= num>>i;
    return (++num)>>1;
}
#endif

#ifndef align_up_yc
#define align_up_yc(num, align) \
        (((num) + ((align) - 1)) & ~((align) - 1))

#endif

#ifndef align_up_yx
#define align_up_yx(num, align) \
	(((num) + ((align) - 1)) & ~((align) - 1))

template<typename T, typename Y>
__device__ __host__ T Align_up(T num, Y align)
{
    return (num/align+(num%align !=0))*align;
}
#endif

#ifndef align_down_yc
#define align_down_yc(num, align) \
        ((num) & ~((align) - 1))
#endif


#ifndef IO_UTIL
#define IO_UTIL
#include <assert.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <sys/stat.h>

template<typename T, typename Y>
T *read_file(char *file_in, Y size) {
    T *output = (T *)malloc(sizeof(T)*size);
    std::ifstream infile(file_in);
    std::string line;
    while(!infile.eof()) {
        getline(infile, line);
        if(line.substr(0,1) != "%")
            break;
    }
    Y idx = 0;
    std::stringstream(line) >> output[idx];
    while(!infile.eof()) {
        getline(infile, line);
        idx++;
        std::stringstream(line) >> output[idx];
    }
    assert(idx+1 == size);
    return output;
}

template<typename T, typename Y>
T *read_binary(char *file_in, Y size) {
    T *output = (T *)malloc(sizeof(T)*size);
    std::ifstream fin(file_in);
    if(fin.is_open()) {
        fin.read((char *)output, sizeof(T)*size);
        fin.close();
    }
    return output;
}

template<typename T, typename Y>
void write_file(char *file_out, T *input, Y size) {
    std::ofstream outf(file_out);
    if(outf.is_open()) {
        std::copy(input, input+size, std::ostream_iterator<T>(outf, "\n"));
        outf.close();
    }
}

template<typename T, typename Y>
void write_binary(char *file_out, T *input, Y size) {
    std::ofstream outf(file_out);
    if(outf.is_open()) {
        outf.write((char *)input, sizeof(T)*size);
        outf.close();
    }
}

inline bool exists_file (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

inline bool exists_file (const char * name) {
    struct stat buffer;   
    return (stat (name, &buffer) == 0); 
}
  
#endif
