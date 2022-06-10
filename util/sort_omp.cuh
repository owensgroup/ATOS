// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sort_omp.cuh
 *
 * @brief Fast sorting using OpenMP
 */

#pragma once

#include <omp.h>

namespace util {

template <typename T, typename SizeT, class Compare>
SizeT bsearch(
    T*      elements,
    SizeT   left_index,
    SizeT   right_index,
    T       element_to_find,
    Compare comp)
{
    SizeT center_index = ((long long)left_index + (long long) right_index)/2;
    if (right_index - left_index <=1)
    {
        if (comp( elements[right_index], element_to_find)) return right_index+1;
        if (comp( element_to_find, elements[left_index])) return left_index;
        return right_index;
    } else {
        if (comp( elements[center_index], element_to_find)) return bsearch(elements, center_index+1, right_index, element_to_find, comp);
        if (comp( element_to_find, elements[center_index])) return bsearch(elements, left_index, center_index-1, element_to_find, comp);
        return center_index;
    }
}

template <typename T, typename SizeT, class Compare>
void omp_sort(
    T*      elements,
    SizeT   _num_elements,
    Compare comp)
{
    long long num_elements = _num_elements;

    if (num_elements < 10000) {
        std::stable_sort(elements, elements + num_elements, comp);
        return;
    }

    T* table2 = NULL;
    T* pivots = NULL;
    T* table3 = (T*) malloc( sizeof(T) * num_elements);
    SizeT* pivot_pos = NULL;
    const int pivot_multi = 4;

    #pragma omp parallel
    {
        int   thread_num  = omp_get_thread_num();
        int   num_threads = omp_get_num_threads();
        SizeT start_pos   = num_elements * thread_num / num_threads;
        SizeT end_pos     = num_elements * (thread_num+1) / num_threads;
        #pragma omp single
        {
            table2 = (T*) malloc( sizeof(T) * num_threads * num_threads * pivot_multi);
            pivots = (T*) malloc( sizeof(T) * num_threads);
            pivot_pos = new SizeT[num_threads * num_threads];
        }

        std::stable_sort(elements + start_pos, elements + end_pos, comp);

        SizeT step = (end_pos - start_pos) / (num_threads * pivot_multi);
        for (int i=0; i< num_threads * pivot_multi; i++)
            table2[i*num_threads + thread_num] = elements[start_pos + int((i+0.1)*step)];

        #pragma omp barrier
        #pragma omp single
        {
            std::stable_sort(table2, table2 + num_threads * num_threads * pivot_multi, comp);
            for (int i=0; i<num_threads-1; i++)
                pivots[i] = table2[(i+1) * num_threads * pivot_multi];
        }

        for (int i=0; i<num_threads-1; i++)
        {
            pivot_pos[thread_num * num_threads + i] = bsearch(elements, start_pos, end_pos-1, pivots[i], comp);
        }
        #pragma omp barrier
        //#pragma omp single
        //    util::cpu_mt::PrintCPUArray("pivot_pos", pivot_pos, num_threads * num_threads);

        SizeT offset = 0, counter = 0;
        for (int i=0; i<thread_num; i++)
        for (int t=0; t<num_threads; t++)
        {
            SizeT end_p = (i == num_threads-1) ? (num_elements * (t+1) / num_threads) : (pivot_pos[t*num_threads + i]);
            SizeT start_p = i == 0 ? (num_elements * t / num_threads) : pivot_pos[t*num_threads+i-1];
            offset += end_p - start_p;
            //printf("thread %d: i=%d, t=%d, offset+=(%d - %d)\n", thread_num, i, t, end_p, start_p);
        }
        for (int t=0; t<num_threads; t++)
        {
            SizeT start_p = thread_num==0 ? num_elements * t / num_threads : pivot_pos[t * num_threads + thread_num -1];
            SizeT end_p = thread_num==num_threads-1 ? num_elements * (t+1) / num_threads : pivot_pos[t*num_threads + thread_num];
            if (end_p > start_p)
            {
                //printf("thread %d: moving elements[ %d ~] @ %p to table3[ %d ~] @ %p, size = %d\n",
                //    thread_num, start_p, elements + start_p, offset + counter, table3 + offset + counter, end_p - start_p);
                memcpy(table3 + offset + counter, elements + start_p, sizeof(T) * (end_p - start_p));
            }
            counter += end_p - start_p;
        }
        std::stable_sort(table3 + offset, table3 + offset + counter, comp);
        #pragma omp barrier

        memcpy(elements + start_pos, table3 + start_pos, sizeof(T) * (end_pos - start_pos));
    }

    free(table2); table2 = NULL;
    free(pivots); pivots = NULL;
    free(table3); table3 = NULL;
    delete[] pivot_pos; pivot_pos = NULL;
}

} //namespace util

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
