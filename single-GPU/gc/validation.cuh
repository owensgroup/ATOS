#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 

template<typename VertexId, typename SizeT, typename GCT>
__global__ void validation(GCT gc, bool *fail)
{
    for(VertexId i=TID; i<gc.nodes; i+=gridDim.x*blockDim.x)
    {
        VertexId color = gc.colors[i];
        SizeT node_offset = gc.csr_offset[i];
        SizeT neighborlen = gc.csr_offset[i+1]-node_offset;
        for(int item=0; item < neighborlen; item++)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(neighbor_color == color && neighbor!=i)
                *(fail) = 1;
        }
    }
}

template <typename VertexId, typename SizeT, typename GCT>
void GCValid(GCT &gc)
{
    bool * fail;
    CUDA_CHECK(cudaMallocManaged(&fail, sizeof(bool)));
    CUDA_CHECK(cudaMemset(fail, 0, sizeof(bool)));
    CUDA_CHECK(cudaDeviceSynchronize());
    validation<VertexId, SizeT><<<320, 512>>>(gc, fail);
    CUDA_CHECK(cudaDeviceSynchronize());
    VertexId colormap[gc.maxDegree+1] = {0};
    VertexId * colors = (VertexId *)malloc(sizeof(VertexId)*gc.nodes);
    CUDA_CHECK(cudaMemcpy(colors, gc.colors, sizeof(VertexId)*gc.nodes, cudaMemcpyDeviceToHost));
    for(int i=0; i<gc.nodes; i++) {
        VertexId colorid = colors[i];
        assert(colorid < gc.maxDegree+1);
        assert(colorid!=0xffffffff);
        colormap[colorid] = 1;
    }
    int total_colors=0;
    for(int i=0; i<gc.maxDegree; i++)
        total_colors+=colormap[i];
    printf("use color %d\n", total_colors);
    if(fail[0] == 1)
        printf("Fail Validation\n");
    else if(fail[0] == 0)
        printf("Pass Validation\n");
}//GCValid

