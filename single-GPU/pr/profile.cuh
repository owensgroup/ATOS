#ifndef PROFILE
#define PROFILE

#include "../../util/util.cuh"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

__device__ uint32_t wl_start_count[512];

void *profile_output;
int *profile_stop;
uint32_t profile_size = 3000000;
cudaStream_t profile_stream;
cudaEvent_t *event_start;
cudaEvent_t *event_stop;
cudaEvent_t event_begin[32];
int event_counter[32];
int event_log_iter = -1;

struct prof_entry {
    uint32_t start;
    uint32_t end;
    int stream_id;
    uint32_t size;
    prof_entry() {}
    prof_entry(uint32_t start_, uint32_t end_, int id, uint32_t size_){
	start =start_;
	end = end_;
	stream_id = id;
	size = size_;
    }
};

template<typename T>
__global__ void Sampling(T *pointer, T *output, int *stop, uint32_t size, int interval=100)
{
    long long int start_time = clock64();
    int i=0;
    while(*((volatile int *)stop) == 0 && i < size)
    {
        long long int end_time = clock64();
        if(end_time-start_time >= interval)
        {
            output[i] = *((volatile T *)pointer);
            i++;
            start_time = end_time;
        }
    }
}

template<typename T>
__global__ void Sampling(T *pointer1, T *pointer2, T *pointer3, int num_counters, T *output, int *stop, uint32_t size, int interval=100)
{
    __shared__ T sum1;
    __shared__ T sum2;
    __shared__ T sum3;
    if(threadIdx.x == 0)
    {
	    sum1 = 0;
	    sum2 = 0;
	    sum3 = 0;
    }
    __syncthreads();
    long long int start_time = clock64();
    int i=0;
    while(*((volatile int *)stop) == 0 && i < size)
    {
        long long int end_time = clock64();
        if(end_time-start_time >= interval) {
	        if(threadIdx.x < num_counters) {
                T temp_out = *((volatile T *)pointer1+64*threadIdx.x);
                atomicAdd(&sum1, temp_out);
                temp_out = *((volatile T *)pointer2+64*threadIdx.x);
                atomicAdd(&sum2, temp_out);
                temp_out = *((volatile T *)pointer3+64*threadIdx.x);
                atomicAdd(&sum3, temp_out);
            }
            __syncwarp();
            if(threadIdx.x == 0) {
                output[i] = sum1;
                output[i+1] = sum2;
                output[i+2] = sum3;
                sum1 = 0;
                sum2 = 0;
                sum3 = 0;
	        }
            __syncthreads();
            i=i+3;
            start_time = end_time;
        }
    }
}

template<typename T>
__global__ void Sampling2(T *pointer1, T *pointer2, int num_counters, T *output, int *stop, uint32_t size, int interval=100)
{
    __shared__ T sum1;
    __shared__ T sum2;
    __shared__ T sum3;
    if(threadIdx.x == 0)
    {
	    sum1 = 0;
	    sum2 = 0;
	    sum3 = 0;
    }
    __syncthreads();
    long long int start_time = clock64();
    int i=0;
    while(*((volatile int *)stop) == 0 && i < size)
    {
        long long int end_time = clock64();
        if(end_time-start_time >= interval) {
	        if(threadIdx.x < num_counters) {
                T temp_out = *((volatile T *)pointer1+64*threadIdx.x);
                atomicAdd(&sum1, temp_out);
                temp_out = *((volatile T *)pointer2+64*threadIdx.x);
                atomicAdd(&sum2, temp_out);
                temp_out = *((volatile T *)(wl_start_count)+64*threadIdx.x);
                atomicAdd(&sum3, temp_out);
            }
            __syncwarp();
            if(threadIdx.x == 0) {
                output[i] = sum1;
                output[i+1] = sum2;
                output[i+2] = sum3;
                sum1 = 0;
                sum2 = 0;
                sum3 = 0;
	        }
            __syncthreads();
            i=i+3;
            start_time = end_time;
        }
    }
}

template<typename T>
void writeToFile(string file_name, T *output, uint32_t size, uint32_t stride)
{
    ofstream myfile (file_name);
    if (myfile.is_open())
    {
        for(int count = 0; count < size; count = count+stride){
            myfile << output[count] << endl;
        }
        myfile.close();
    }
}

void outputFiles(string str_file, int option)
{
    int beginIdx = str_file.rfind('/');
    int endIdx = str_file.find(".csr");
    std::string filename = str_file.substr(beginIdx + 1, endIdx-beginIdx-1);   
    if(option == 2 || option == 3) {
        std::string file_name_end_count = filename + "_end_count";
        std::string file_name_end = filename + "_end";
        std::string file_name_start_count = filename + "_start_count";
        if(option == 2) {
            file_name_end = file_name_end + "_o2.txt";
            file_name_end_count = file_name_end_count + "_o2.txt";
            file_name_start_count = file_name_start_count + "_o2.txt";
        }
        else if(option == 3) {
            file_name_end = file_name_end + "_o3.txt";
            file_name_end_count = file_name_end_count + "_o3.txt";
            file_name_start_count = file_name_start_count + "_o3.txt";
        }
        std::cout << "writing profile to " << file_name_end_count << ", " << file_name_end << ", " << file_name_start_count << std::endl;
        writeToFile(file_name_end_count, ((uint32_t *)profile_output), profile_size, 30);
        writeToFile(file_name_end, ((uint32_t *)profile_output)+1, profile_size, 30);
        writeToFile(file_name_start_count, ((uint32_t *)profile_output)+2, profile_size, 30);
        CUDA_CHECK(cudaFree(profile_output));
    }
    else if(option == 4)
    {
        float * time_interval[32];
        float * time_accu[32];
        int max_len=0;
        for(int i=0; i<32; i++) {
            time_interval[i] = (float *)malloc(sizeof(float)*event_counter[i]);
            time_accu[i] = (float *)malloc(sizeof(float)*event_counter[i]);
            max_len = max(event_counter[i], max_len);
        }
            
        for(int stream_id=0; stream_id < 32; stream_id++) 
            for(int i=0; i<event_counter[stream_id]; i++) {
                CUDA_CHECK(cudaEventElapsedTime(time_interval[stream_id]+i, event_start[stream_id*1000+i], event_stop[stream_id*1000+i]))
                CUDA_CHECK(cudaEventElapsedTime(time_accu[stream_id]+i, event_begin[stream_id], event_stop[stream_id*1000+i]))
            }
        
            string file_name_interval = filename +"_interval_time.txt";
            string file_name_accu = filename +"_accu_time.txt";
            string file_name_size = filename +"_kernel_info.txt";
            string file_name_stream_kernel_size = filename +"_stream_size.txt";
            std::cout << "Writing to files: "<< file_name_interval << "\t" <<  file_name_accu << "\t" << file_name_size  << "\t"<< file_name_stream_kernel_size << std::endl;
         
            int * kernel_size = (int *)malloc(sizeof(int)*max_len);
            int local_size = 0;
            std::ofstream myfile4 (file_name_stream_kernel_size);
            if(myfile4.is_open()) {
                for(int stream_id =0; stream_id < 32; stream_id++)
                {
                    for(int i=0; i<event_log_iter; i++)
                    if(((prof_entry *)profile_output)[i].stream_id==stream_id) {
                      kernel_size[local_size] = ((prof_entry *)profile_output)[i].size;
                      local_size++;
                    }
                    //std::cout << "stream "<< stream_id <<  "\t" << local_size << std::endl;
                    for(int i=0; i<local_size; i++)
                        myfile4 << kernel_size[i] << "\t";
                    myfile4 << "\n\n";
                    local_size = 0;
                }
                myfile4.close();
            }
         
            std::ofstream myfile (file_name_interval);
            if (myfile.is_open())
            {
                for(int stream_id=0; stream_id<32; stream_id++) {
                    for(int count = 0; count < event_counter[stream_id]; count++){
                       myfile << time_interval[stream_id][count] << "\t";
                    }
                    myfile << "\n\n";
                }
                myfile.close();
            }
         
            std::ofstream myfile2 (file_name_accu);
            if (myfile2.is_open())
            {
                for(int stream_id=0; stream_id<32; stream_id++) {
                    for(int count = 0; count < event_counter[stream_id]; count++){
                       myfile2 << time_accu[stream_id][count] << "\t";
                    }
                    myfile2 << "\n\n";
                }
                myfile2.close();
            }
         
            std::ofstream myfile3 (file_name_size);
            if(myfile3.is_open())
            {
                for(int count = 0; count < event_log_iter; count++)
                   myfile3 << ((prof_entry *)profile_output)[count].start << "\t" << ((prof_entry *)profile_output)[count].end << "\t"<< ((prof_entry *)profile_output)[count].size << "\t" << ((prof_entry *)profile_output)[count].stream_id << "\n";
                myfile3.close();
            }
            free(profile_output);
            free(event_start);
            free(event_stop);
    }
}


#endif
