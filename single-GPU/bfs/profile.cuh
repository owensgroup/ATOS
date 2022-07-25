#ifndef PROFILE
#define PROFILE

#include "../../util/util.cuh"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

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
__global__ void Sampling(T *pointer1, T *pointer2, T *pointer3, T *output, int *stop, uint32_t size, int interval=100)
{
    long long int start_time = clock64();
    int i=0;
    while(*((volatile int *)stop) == 0 && i < size)
    {
        long long int end_time = clock64();
        if(end_time-start_time >= interval)
        {
            output[i] = *((volatile T *)pointer1);
            output[i+1] = *((volatile T *)pointer2);
            output[i+2] = *((volatile T *)pointer3);
            i=i+3;
            start_time = end_time;
        }
    }
}

template<typename T>
__global__ void Sampling(T *pointer1_a, T *pointer2_a, T *pointer3_a, T *pointer4_a, T *pointer1_b, T *pointer2_b, T *pointer3_b, T *pointer4_b, T *pointer1_c, T *pointer2_c, 
        T *pointer3_c,  T *pointer4_c, T *output, int *stop, uint32_t size, int interval=100)
{
    long long int start_time = clock64();
    int i=0;
    while(*((volatile int *)stop) == 0 && i < size)
    {
        long long int end_time = clock64();
        if(end_time-start_time >= interval)
        {
            output[i] = *((volatile T *)pointer1_a) + *((volatile T *)pointer2_a) + *((volatile T *)pointer3_a) + *((volatile T *)pointer4_a);
            output[i+1] = *((volatile T *)pointer1_b) + *((volatile T *)pointer2_b) + *((volatile T *)pointer3_b) + *((volatile T *)pointer4_b);
            output[i+2] = *((volatile T *)pointer1_c) + *((volatile T *)pointer2_c) + *((volatile T *)pointer3_c) + *((volatile T *)pointer4_c);
            i=i+3;
            start_time = end_time;
        }
    }
}


template<typename T>
__global__ void signalStop(T *stop, T value)
{
    *stop = value;
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


#endif
