#ifndef GraphColoring_Baseline
#define GraphColoring_Baseline
#include "../../util/error_util.cuh"
#include "../../util/util.cuh"

#include <cub/cub.cuh>

template<typename VertexId, typename SizeT>
struct GC_BSP
{
    VertexId nodes;
    SizeT edges;

    SizeT *csr_offset = NULL;
    VertexId *csr_indices = NULL;

    VertexId maxDegree = 0xffffffff;
    VertexId fixSize = 256;  //if fixSize=512, forbiddenColors size will larger than max(int)

    VertexId *colors = NULL;
    bool *forbiddenColors = NULL;

    VertexId *workload = NULL;
    VertexId *workload_small = NULL;
    VertexId *workload_large = NULL;
    VertexId *frontier = NULL;
    VertexId *frontier_small = NULL;
    VertexId *frontier_large = NULL;
    VertexId *addNext = NULL;
    VertexId *addNext_small = NULL;
    VertexId *addNext_large = NULL;
    VertexId *addNext_scan = NULL;
    VertexId *addNext_scan_small = NULL;
    VertexId *addNext_scan_large = NULL;
    void     *d_temp_storage = NULL;
    void     *d_temp_storage_small = NULL;
    void     *d_temp_storage_large = NULL;
    size_t   temp_storage_bytes = 0;
   
    VertexId *init_queue = NULL;

    cudaStream_t streams[3];

    GC_BSP(Csr<VertexId, SizeT> &csr)
    {
        nodes = csr.nodes;
        edges = csr.edges;
        
        csr_offset = csr.row_offset;
        csr_indices = csr.column_indices;

        maxDegree = 0;
        for(int i=0; i<nodes; i++)
        {
            maxDegree = max(maxDegree, csr.row_offset[i+1]-csr.row_offset[i]);
        }
        printf("max degree %d\n", maxDegree);

        CUDA_CHECK(cudaMalloc(&colors, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&forbiddenColors, sizeof(bool)*fixSize*nodes));

        CUDA_CHECK(cudaMalloc(&workload, sizeof(VertexId)));
        CUDA_CHECK(cudaMalloc(&workload_small, sizeof(VertexId)));
        CUDA_CHECK(cudaMalloc(&workload_large, sizeof(VertexId)));

        CUDA_CHECK(cudaMalloc(&frontier, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&frontier_small, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&frontier_large, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&addNext, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&addNext_small, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&addNext_large, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMalloc(&addNext_scan, sizeof(VertexId)*(nodes+1)));
        CUDA_CHECK(cudaMalloc(&addNext_scan_small, sizeof(VertexId)*(nodes+1)));
        CUDA_CHECK(cudaMalloc(&addNext_scan_large, sizeof(VertexId)*(nodes+1)));

        CUDA_CHECK(cudaMemset(workload, 0xffffffff, sizeof(VertexId)))
        CUDA_CHECK(cudaMemset(workload_small, 0xffffffff, sizeof(VertexId)))
        CUDA_CHECK(cudaMemset(workload_large, 0xffffffff, sizeof(VertexId)))
        CUDA_CHECK(cudaMemset(colors, 0xffffffff, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(forbiddenColors, false, sizeof(bool)*fixSize*nodes));
        CUDA_CHECK(cudaMemset(addNext, 0, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(addNext_small, 0, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(addNext_large, 0, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(addNext_scan, 0, sizeof(VertexId)*(nodes+1)));
        CUDA_CHECK(cudaMemset(addNext_scan_small, 0, sizeof(VertexId)*(nodes+1)));
        CUDA_CHECK(cudaMemset(addNext_scan_large, 0, sizeof(VertexId)*(nodes+1)));

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes);
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CUDA_CHECK(cudaMalloc(&d_temp_storage_small, temp_storage_bytes));
        CUDA_CHECK(cudaMalloc(&d_temp_storage_large, temp_storage_bytes));

        CUDA_CHECK(cudaStreamCreateWithFlags(streams, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+1, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(streams+2, cudaStreamNonBlocking));
    }

    void release()
    {
        if(colors!=NULL)
            CUDA_CHECK(cudaFree(colors));
        if(forbiddenColors!=NULL)
            CUDA_CHECK(cudaFree(forbiddenColors));
        if(frontier!=NULL)
            CUDA_CHECK(cudaFree(frontier));
        if(addNext!=NULL)
            CUDA_CHECK(cudaFree(addNext));
        if(addNext_scan!=NULL)
            CUDA_CHECK(cudaFree(addNext_scan));
        if(d_temp_storage!=NULL)
            CUDA_CHECK(cudaFree(d_temp_storage));
        if(init_queue!=NULL)
            free(init_queue);
        if(frontier_small!=NULL)
            CUDA_CHECK(cudaFree(frontier_small));
        if(frontier_large!=NULL)
            CUDA_CHECK(cudaFree(frontier_large));
        if(addNext_small!=NULL)
            CUDA_CHECK(cudaFree(addNext_small));
        if(addNext_large!=NULL)
            CUDA_CHECK(cudaFree(addNext_large));
        if(addNext_scan_small!=NULL)
            CUDA_CHECK(cudaFree(addNext_scan_small));
        if(addNext_scan_large!=NULL)
            CUDA_CHECK(cudaFree(addNext_scan_large));
        if(d_temp_storage_small!=NULL)
            CUDA_CHECK(cudaFree(d_temp_storage_small));
        if(d_temp_storage_large!=NULL)
            CUDA_CHECK(cudaFree(d_temp_storage_large));
    }

    void reset()
    {
        CUDA_CHECK(cudaMemset(workload, 0xffffffff, sizeof(VertexId)));
        CUDA_CHECK(cudaMemset(workload_small, 0xffffffff, sizeof(VertexId)));
        CUDA_CHECK(cudaMemset(workload_large, 0xffffffff, sizeof(VertexId)));
        CUDA_CHECK(cudaMemset(colors, 0xffffffff, sizeof(VertexId)*nodes));
        CUDA_CHECK(cudaMemset(forbiddenColors, 0, sizeof(bool)*fixSize*nodes));
        CUDA_CHECK(cudaMemset(addNext, 0, sizeof(VertexId)*nodes));    
        CUDA_CHECK(cudaMemset(addNext_small, 0, sizeof(VertexId)*nodes));    
        CUDA_CHECK(cudaMemset(addNext_large, 0, sizeof(VertexId)*nodes));    
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void RandomInit(bool ifCreate);
    void GCInit(bool random, bool ifcreate);
    VertexId GCStart(cudaStream_t stream);
    VertexId GCStart_op(cudaStream_t stream);
    VertexId GCStart_warp_thread_cta();
};

template<typename VertexId, typename SizeT>
__global__ void assignColor_op_thread(GC_BSP<VertexId, SizeT> gc)
{
    //neighboer lis length < 32 will be processed here
    for(VertexId fidx = TID; fidx < *(gc.workload_small); fidx+=gridDim.x*blockDim.x)
    {
        VertexId node = gc.frontier_small[fidx];
        SizeT node_offset = gc.csr_offset[node];
        SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
        
        uint32_t forbiddenColors = 0;
        for(int item = 0; item < neighborlen; item++)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId color = gc.colors[neighbor];
            if(color!=0xffffffff && color < 32 && color >=0)
                forbiddenColors = forbiddenColors | (1<<(color)) ; 
        }
        forbiddenColors = (~forbiddenColors);
        assert(forbiddenColors!=0);
        gc.colors[node] = __ffs(forbiddenColors)-1;
    } 
}

template<typename VertexId, typename SizeT>
__global__ void assignColor_op_warp(GC_BSP<VertexId, SizeT> gc)
{
    //__shared__ uint32_t forbiddenColors[1024];  // 16 warps, each get 64 uint32_t: fixSize = 32*64 = 2048;
    extern __shared__ uint32_t forbiddenColors[];  // size number of warps in the block times 64, fixSize for each warp: 32*64=2048
    for(VertexId fidx = WARPID; fidx < *(gc.workload); fidx+=(gridDim.x*blockDim.x/32))
    {
        VertexId node = gc.frontier[fidx];
        SizeT node_offset = gc.csr_offset[node];
        SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
        unsigned found_mask = 0;
        
        for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=2048)
        {
            forbiddenColors[LANE_+(threadIdx.x>>5)*64] = 0;
            forbiddenColors[LANE_+32+(threadIdx.x>>5)*64] = 0;
            //__threadfence_block(); // ??
            __syncwarp(0xffffffff);

            //assert(forbiddenColors[(threadIdx.x>>5)*64] == 0);
            //__syncwarp();
            for(int item = LANE_; item<neighborlen; item=item+32)
            {
                VertexId neighbor = gc.csr_indices[node_offset+item];
                VertexId color = gc.colors[neighbor];
                VertexId color_forbid = color-colorRange;
                if(color!=0xffffffff && color_forbid < 2048 && color_forbid >= 0) {
                    atomicOr(forbiddenColors+(threadIdx.x>>5)*64+color_forbid/32, (1<<(color_forbid%32)));
                }
            }
            //__threadfence_block(); // ?
            __syncwarp();
            //VertexId chosenColor = 0xffffffff;
            for(int item = LANE_; item < 64; item=item+32)
            {
                unsigned mask = forbiddenColors[(threadIdx.x>>5)*64+item];
                mask = (~mask);
                found_mask = __ballot_sync(0xffffffff, (mask!=0));
                if(found_mask != 0) {
                    if(LANE_ == __ffs(found_mask)-1) {
                        //chosenColor = __ffs(mask)-1+colorRange+item*32;
                        //if( chosenColor >= 13)
                            //printf("LANE %d, node %d, item %d, mask %x, found_mas %x, chosencolor %d\n", LANE_, node, item, mask, found_mask, chosenColor);
                        gc.colors[node] = __ffs(mask)-1+colorRange+item*32;
                    }
                    //chosenColor = __shfl_sync(0xffffffff, chosenColor, __ffs(found_mask)-1);
                    break;
                }
            }
            //if(chosenColor >= 13 && LANE_ < neighborlen)
            //    printf("LANE %d, node %d, neighbor %d, neighborlen %d, color %d, colorrange %d, atomicor %d, mask %x, return or %x\n", LANE_, node, neighbor, neighborlen, color, colorRange, (1<<(color%32)), forbiddenColors[(threadIdx.x>>5)*64+LANE_], re_or);
            if(found_mask != 0)
                break;
            __syncwarp();
        }
        __syncwarp();
        assert(found_mask != 0);
    }
}

template<typename VertexId, typename SizeT>
__global__ void assignColor_op_CTA(GC_BSP<VertexId, SizeT> gc)
{
    __shared__ uint32_t forbiddenColors[1024];
    __shared__ VertexId node;
    __shared__ SizeT node_offset;
    __shared__ SizeT neighborlen;
    __shared__ VertexId newColor;

    for(VertexId fidx = blockIdx.x; fidx < *(gc.workload_large); fidx=fidx+gridDim.x)
    {
        if(threadIdx.x == 0)
        {
            node = gc.frontier_large[fidx];
            node_offset = gc.csr_offset[node];
            neighborlen = gc.csr_offset[node+1]-node_offset;
            newColor = 0x7fffffff;
        }
        __syncthreads();

        for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=32768)
        {
            for(int item = threadIdx.x; item < 1024; item=item+blockDim.x)
                forbiddenColors[item] = 0;
            __syncthreads();

            for(int item=threadIdx.x; item<neighborlen; item=item+blockDim.x)
            {
                VertexId neighbor = gc.csr_indices[node_offset+item];
                VertexId neighbor_color = gc.colors[neighbor];
                VertexId forbidd_color = neighbor_color-colorRange;
                if(neighbor_color!=0xffffffff && forbidd_color < 32768 && forbidd_color >=0)
                    atomicOr(forbiddenColors+forbidd_color/32, (1<<(forbidd_color%32)));
            }
            __syncthreads();
            for(int item = threadIdx.x; item-threadIdx.x < 1024; item=item+blockDim.x)
            {
                unsigned mask = 0;
                if(item < 1024) {
                    mask = forbiddenColors[item];
                    mask = (~mask);
                }

                unsigned found_mask = __ballot_sync(0xffffffff, (mask!=0));
                if(found_mask != 0) {
                    if(LANE_ == __ffs(found_mask)-1)
                        atomicMin(&newColor, item*32+colorRange+__ffs(mask)-1);
                }
                __syncthreads();
                if(newColor!=0x7fffffff)
                    break;
            }
            __syncthreads();
            if(newColor!=0x7fffffff) {
                if(threadIdx.x == 0)
                    gc.colors[node] = newColor;
                break;
            }
        }
        assert(newColor!=0x7fffffff);
        __syncthreads();
    }

}

template<typename VertexId, typename SizeT>
__global__ void assignColor(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId fidx = WARPID; fidx < *(gc.workload); fidx+=(gridDim.x*blockDim.x/32))
    {
        VertexId node = gc.frontier[fidx];
        SizeT node_offset = gc.csr_offset[node];
        SizeT neighborlen = gc.csr_offset[node+1]-node_offset;
        bool found = false;
        //for(int item = LANE_; item < gc.fixSize; item=item+32)
        //{

        //    uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
        //    assert(gc.forbiddenColors[forbid_idx] == false);
        //}
        //__syncwarp();
        for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=gc.fixSize)
        {
            for(int item = LANE_; item < gc.fixSize; item=item+32)
            {
                uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
                //gc.forbiddenColors[node*gc.fixSize+item] = false;
                gc.forbiddenColors[forbid_idx] = false;
            }
            //__threadfence();
            __syncwarp();

            //for(int item = LANE_-3; item < gc.fixSize; item=item+32)
            //{
            //    if(item >= 0) {
            //        uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
            //        assert(gc.forbiddenColors[forbid_idx] == false);
            //    }
            //}
            //__syncwarp();
            for(int item = LANE_; item<neighborlen; item=item+32)
            {
                VertexId neighbor = gc.csr_indices[node_offset+item];
                VertexId color = gc.colors[neighbor];
                //if(node == 4378452 || node == 4666829)
                //    printf("node %d, colorRange %d, neighbor %d, neighborcolor %d\n", node, colorRange, item, color);
                uint64_t forbid_idx = uint64_t(node)*gc.fixSize+color-colorRange;
                if(color!=0xffffffff && color < colorRange+gc.fixSize && color >= colorRange)
                    //gc.forbiddenColors[node*gc.fixSize+color-colorRange] = true;
                    gc.forbiddenColors[forbid_idx] = true;
            }
            //__threadfence();
            __syncwarp();
            bool ifForbid = false;
            for(int item = LANE_; item-LANE_ < gc.fixSize; item=item+32)
            {
                if(item < gc.fixSize) {
                    uint64_t forbid_idx = uint64_t(node)*gc.fixSize+item;
                    if(forbid_idx >= uint64_t(gc.nodes)*gc.fixSize)
                        printf("TID %d out of bound: node %d (total %d), fixsize %d, item %d, %ld > bound %ld\n",
                         TID, node, gc.nodes, gc.fixSize, item, forbid_idx, uint64_t(gc.nodes)*gc.fixSize);
                    assert(forbid_idx < uint64_t(gc.nodes)*gc.fixSize);
                    ifForbid = gc.forbiddenColors[forbid_idx];
                    //if(node*gc.fixSize+item >= gc.nodes*gc.fixSize)
                    //    printf("TID %d out of bound: node %d (total %d), fixsize %d, item %d, %d > bound %d\n",
                    //     TID, node, gc.nodes, gc.fixSize, item,node*gc.fixSize+item, gc.nodes*gc.fixSize);
                    //assert(node*gc.fixSize+item < gc.nodes*gc.fixSize);
                    //ifForbid = gc.forbiddenColors[node*gc.fixSize+item];
                }
                unsigned mask = __ballot_sync(0xffffffff, !ifForbid);
                if(mask!=0)
                {
                    found = true;
                    if(LANE_ == 0) {
                        gc.colors[node] = colorRange+item+__ffs(mask)-1;
                        assert(colorRange+item+__ffs(mask)-1!=0xffffffff);
                        assert(colorRange+item+__ffs(mask)-1 >= 0);
                        assert(colorRange+item+__ffs(mask)-1 < colorRange+gc.fixSize);
                        assert(colorRange+item+__ffs(mask)-1 < gc.maxDegree);
                    }
                    break;
                }
            }
            if(found == true)
                break;
            __syncwarp();
        }
        if(found == false)
        {
            for(int item = LANE_; item < gc.fixSize; item=item+32)
            {
                gc.forbiddenColors[node*gc.fixSize+item] = false;
            }
            __threadfence();
            __syncwarp();
            for(int item = LANE_; item < gc.fixSize; item=item+32)
            {
                assert(gc.forbiddenColors[node*gc.fixSize+item] == false);
            }
            __threadfence();
            __syncwarp();
            if(node == 4378453 ) {
            printf("node %d failed to assign color, neighborlen %d\n", node, neighborlen);
            //for(int colorRange = 0; colorRange < gc.maxDegree+1; colorRange+=gc.fixSize)
            int colorRange = 0;
            {
                for(int item = LANE_; item<neighborlen; item=item+32)
                {
                    VertexId neighbor = gc.csr_indices[node_offset+item];
                    VertexId color = gc.colors[neighbor];
                    printf("node %d, neighbor %d, color %d, colorRange %d\n", node, neighbor, color, colorRange);
                    if(color!=0xffffffff && color < colorRange+gc.fixSize && color >= colorRange)
                        gc.forbiddenColors[node*gc.fixSize+color-colorRange] = true; 
                }
                __threadfence();
                __syncwarp();
                bool ifForbid = false;
                for(int item = LANE_; item-LANE_ < gc.fixSize; item=item+32)
                {
                    if(item < gc.fixSize) {
                        ifForbid = gc.forbiddenColors[node*gc.fixSize+item];
                        printf("node %d, item %d, ifForbid %d\n", node, item, ifForbid);
                    }
                    unsigned mask = __ballot_sync(0xffffffff, !ifForbid);
                    if(mask!=0)
                    {
                        found = true;
                    }
                    printf("node %d, colorRange %d, item %d, !ifForbid %d, mask %x, found %d\n", 
                    node, colorRange, item, !ifForbid, mask, found);
                } 
            }
            assert(0);
            }
        }
        __syncwarp();
    }
}

template<typename VertexId, typename SizeT>
__global__ void detectConflicts(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId vid = WARPID; vid < gc.nodes; vid+=(gridDim.x*blockDim.x/32))
    {
        VertexId node_color = gc.colors[vid];
        SizeT node_offset = gc.csr_offset[vid];
        SizeT neighborlen = gc.csr_offset[vid+1]-node_offset;
        for(int item=LANE_; item<neighborlen; item=item+32)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(neighbor_color == node_color && neighbor < vid)
                gc.addNext[vid] = true;
        }
    }
}

template<typename VertexId, typename SizeT>
__global__ void detectConflicts_op_warp(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId fidx = WARPID; fidx < *(gc.workload); fidx+=(gridDim.x*blockDim.x/32))
    {
        VertexId vid = gc.frontier[fidx];
        VertexId node_color = gc.colors[vid];
        SizeT node_offset = gc.csr_offset[vid];
        SizeT neighborlen = gc.csr_offset[vid+1]-node_offset;
        for(int item=LANE_; item<neighborlen; item=item+32)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(neighbor_color == node_color && neighbor < vid)
                gc.addNext[vid] = true;
        }
    }
}

template<typename VertexId, typename SizeT>
__global__ void detectConflicts_op_thread(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId fidx = TID; fidx < *(gc.workload_small); fidx+=(gridDim.x*blockDim.x))
    {
        VertexId vid = gc.frontier_small[fidx];
        VertexId node_color = gc.colors[vid];
        SizeT node_offset = gc.csr_offset[vid];
        SizeT neighborlen = gc.csr_offset[vid+1]-node_offset;
        for(int item = 0; item<neighborlen; item++)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(neighbor_color == node_color && neighbor < vid)
                gc.addNext_small[vid] = true;
        }
    }
}


template<typename VertexId, typename SizeT>
__global__ void detectConflicts_op_CTA(GC_BSP<VertexId, SizeT> gc)
{
    __shared__ VertexId node;
    __shared__ VertexId color;
    __shared__ SizeT node_offset;
    __shared__ SizeT neighborlen;

    for(VertexId fidx = blockIdx.x; fidx < *(gc.workload_large); fidx=fidx+gridDim.x)
    {
        if(threadIdx.x==0)
        {
            node = gc.frontier_large[fidx];
            color = gc.colors[node];
            node_offset = gc.csr_offset[node];
            neighborlen = gc.csr_offset[node+1]-node_offset;
        }
        __syncthreads();

        for(int item=threadIdx.x; item < neighborlen; item=item+blockDim.x)
        {
            VertexId neighbor = gc.csr_indices[node_offset+item];
            VertexId neighbor_color = gc.colors[neighbor];
            if(color == neighbor_color && neighbor < node)
                gc.addNext_large[node] = true;
        }
    }
}

template<typename VertexId, typename SizeT>
__global__ void compactKernel(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId i=TID; i<gc.nodes; i+=blockDim.x*gridDim.x)
    {
        VertexId idx = gc.addNext_scan[i];
        if(gc.addNext_scan[i+1] - idx == 1)
        {
            gc.frontier[idx] = i;
        }
        else if(gc.addNext_scan[i+1] - idx != 0)
        {
            printf("scan[%d]: %d scan[%d]: %d, addNext[%d] %d, addNext[%d] %d\n", 
            i+1, gc.addNext_scan[i+1], i, gc.addNext_scan[i], i+1, gc.addNext[i+1], i, gc.addNext[i]);
            printf("Scan has more than 1 different consecutively, ERROR\n");
            assert(0);
        }
        if(i == gc.nodes-1)
            *(gc.workload) = gc.addNext_scan[i+1];
    }
}

template<typename VertexId, typename SizeT>
__global__ void compactKernel_thread_warp_cta(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId i=TID; i<gc.nodes; i+=blockDim.x*gridDim.x)
    {
        VertexId idx = gc.addNext_scan[i];
        if(gc.addNext_scan[i+1] - idx == 1)
        {
            gc.frontier[idx] = i;
        }
        else if(gc.addNext_scan[i+1] - idx != 0)
        {
            printf("scan[%d]: %d scan[%d]: %d, addNext[%d] %d, addNext[%d] %d\n", 
            i+1, gc.addNext_scan[i+1], i, gc.addNext_scan[i], i+1, gc.addNext[i+1], i, gc.addNext[i]);
            printf("Scan has more than 1 different consecutively, ERROR\n");
            assert(0);
        }
        if(i == gc.nodes-1)
            *(gc.workload) = gc.addNext_scan[i+1];


        // small
        idx = gc.addNext_scan_small[i];
        if(gc.addNext_scan_small[i+1]-idx == 1)
            gc.frontier_small[idx] = i;
        else if(gc.addNext_scan_small[i+1]-idx != 0)
            assert(0);
        
        if(i == gc.nodes-1)
            *(gc.workload_small) = gc.addNext_scan_small[i+1];
        
        // large
        idx = gc.addNext_scan_large[i];
        if(gc.addNext_scan_large[i+1]-idx == 1)
            gc.frontier_large[idx] = i;
        else if(gc.addNext_scan_large[i+1]-idx != 0)
            assert(0);

        if(i == gc.nodes-1)
            *(gc.workload_large) = gc.addNext_scan_large[i+1];
    }
}

template<typename VertexId, typename SizeT>
__global__ void sortNodes(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId vid = TID; vid < gc.nodes; vid+=gridDim.x*blockDim.x)
    {
        SizeT len = gc.csr_offset[vid+1]-gc.csr_offset[vid];
        if(len < 32)
            gc.addNext_small[vid] = true;
        else if(len >= 32 && len < 1024) 
            gc.addNext[vid] = true;
        else 
            gc.addNext_large[vid] = true;
    }
}

template<typename VertexId, typename SizeT>
__global__ void init(GC_BSP<VertexId, SizeT> gc)
{
    for(VertexId id = TID; id < gc.nodes; id+=gridDim.x*blockDim.x)
        gc.frontier[id] = id;
    if(TID == 0)
        *(gc.workload) = gc.nodes;
}

template<typename VertexId, typename SizeT>
void GC_BSP<VertexId, SizeT>::GCInit(bool random, bool ifcreate)
{
    if(random) {
        RandomInit(ifcreate);
    }
    else {
        init<<<320, 512>>>(*this);   
    }
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename VertexId, typename SizeT>
void GC_BSP<VertexId, SizeT>::RandomInit(bool ifCreate)
{
    if(ifCreate) {
        init_queue = (VertexId *)malloc(sizeof(VertexId)*nodes);
        VertexId *array = (VertexId *)malloc(sizeof(VertexId)*nodes);
        for(int i=0; i<nodes; i++) {
            array[i] = i;
            init_queue[i] = 0xffffffff;
        }
    
        srand(12321);
        VertexId nodeAdded = 0;
        while(nodeAdded < nodes)
        {
            VertexId nodeId = rand()%nodes;
            if(array[nodeId]!=0xffffffff) {
                assert(array[nodeId] == nodeId);
                init_queue[nodeAdded] = nodeId;
                array[nodeId] = 0xffffffff;
                nodeAdded++;
            }
        }
    
        for(int i=0; i<nodes; i++)
            assert(init_queue[i]!=0xffffffff);
    }
    
    CUDA_CHECK(cudaMemcpy((void *)(frontier), init_queue, sizeof(VertexId)*nodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void *)(workload), &nodes, sizeof(VertexId), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

}

template<typename VertexId, typename SizeT>
VertexId GC_BSP<VertexId, SizeT>::GCStart(cudaStream_t stream)
{
    int numBlock = 320;
    int numThread = 512;
    VertexId h_workload;
    VertexId total_workload = nodes;
    int iter = 0;
    do 
    {   
        //assignColor<<<320, 512, 0, stream>>>(*this);
        assignColor_op_warp<<<numBlock, numThread, sizeof(uint32_t)*64*(numThread/32), stream>>>(*this);
        detectConflicts<<<320, 512, 0, stream>>>(*this);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes, stream);
        compactKernel<<<320, 512, 0, stream>>>(*this);
        CUDA_CHECK(cudaMemcpyAsync(&h_workload, workload, sizeof(VertexId), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemsetAsync(addNext, 0, sizeof(VertexId)*nodes, stream));
        //CUDA_CHECK(cudaMemsetAsync(forbiddenColors, 0, sizeof(bool)*fixSize*nodes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); 
        #ifdef ITER_WORKLOAD
        printf("iter %d, workload %d\n", iter, h_workload);
        #endif
        total_workload += h_workload;
        iter++;
    }
    while(h_workload>0);
    return total_workload;
}

template<typename VertexId, typename SizeT>
VertexId GC_BSP<VertexId, SizeT>::GCStart_warp_thread_cta()
{
    int numBlock = 320;
    int numThread = 512;
    VertexId h_workload, h_workload_small, h_workload_large;
    VertexId total_workload = nodes;
    int iter = 0;
    
    sortNodes<<<320, 512, 0, streams[0]>>>(*this);
    CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes, streams[0]);
    cub::DeviceScan::InclusiveSum(d_temp_storage_small, temp_storage_bytes, addNext_small, addNext_scan_small+1, nodes, streams[1]);
    cub::DeviceScan::InclusiveSum(d_temp_storage_large, temp_storage_bytes, addNext_large, addNext_scan_large+1, nodes, streams[2]);
    CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    CUDA_CHECK(cudaStreamSynchronize(streams[1])); 
    CUDA_CHECK(cudaStreamSynchronize(streams[2])); 
    compactKernel_thread_warp_cta<<<320, 512, 0, streams[0]>>>(*this);
    CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    CUDA_CHECK(cudaMemsetAsync(addNext, 0, sizeof(VertexId)*nodes, streams[0]));
    CUDA_CHECK(cudaMemsetAsync(addNext_small, 0, sizeof(VertexId)*nodes, streams[0]));
    CUDA_CHECK(cudaMemsetAsync(addNext_large, 0, sizeof(VertexId)*nodes, streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    //CUDA_CHECK(cudaMemcpyAsync(&h_workload, workload, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[0]));
    //CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    //CUDA_CHECK(cudaMemcpyAsync(&h_workload_small, workload_small, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[0]));
    //CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    //CUDA_CHECK(cudaMemcpyAsync(&h_workload_large, workload_large, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[0]));
    //CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
    //printf(" workload %d %d %d, total %d\n", h_workload, h_workload_small, h_workload_large, h_workload+h_workload_small+h_workload_large);

    do 
    {   
        assignColor_op_warp<<<numBlock, numThread, sizeof(uint32_t)*64*(numThread/32), streams[0]>>>(*this);
        assignColor_op_thread<<<numBlock, numThread, 0, streams[1]>>>(*this);
        assignColor_op_CTA<<<numBlock, numThread, 0, streams[2]>>>(*this);
        CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[1])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[2])); 
        detectConflicts_op_warp<<<320, 512, 0, streams[0]>>>(*this);
        detectConflicts_op_thread<<<320, 512, 0, streams[1]>>>(*this);
        detectConflicts_op_CTA<<<320, 512, 0, streams[2]>>>(*this);
        CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));  // have to keep otherwise road graph will go wrong
        CUDA_CHECK(cudaStreamSynchronize(streams[2])); 
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes, streams[0]);
        cub::DeviceScan::InclusiveSum(d_temp_storage_small, temp_storage_bytes, addNext_small, addNext_scan_small+1, nodes, streams[1]);
        cub::DeviceScan::InclusiveSum(d_temp_storage_large, temp_storage_bytes, addNext_large, addNext_scan_large+1, nodes, streams[2]);
        CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[1])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[2])); 
        compactKernel_thread_warp_cta<<<320, 512, 0, streams[0]>>>(*this);
        CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
        CUDA_CHECK(cudaMemcpyAsync(&h_workload, workload, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CHECK(cudaMemcpyAsync(&h_workload_small, workload_small, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[1]));
        CUDA_CHECK(cudaMemcpyAsync(&h_workload_large, workload_large, sizeof(VertexId), cudaMemcpyDeviceToHost, streams[2]));
        CUDA_CHECK(cudaMemsetAsync(addNext, 0, sizeof(VertexId)*nodes, streams[0]));
        CUDA_CHECK(cudaMemsetAsync(addNext_small, 0, sizeof(VertexId)*nodes, streams[1]));
        CUDA_CHECK(cudaMemsetAsync(addNext_large, 0, sizeof(VertexId)*nodes, streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[1])); 
        CUDA_CHECK(cudaStreamSynchronize(streams[2])); 
        #ifdef ITER_WORKLOAD
        printf("iter %d, workload %d %d %d, total %d\n", iter, h_workload, h_workload_small, h_workload_large, h_workload+h_workload_small+h_workload_large);
        #endif
        total_workload += h_workload+h_workload_small+h_workload_large;
        iter++;
    }
    while(h_workload+h_workload_small+h_workload_large>0);
    return total_workload;
}

template<typename VertexId, typename SizeT>
VertexId GC_BSP<VertexId, SizeT>::GCStart_op(cudaStream_t stream)
{
    int numBlock = 320;
    int numThread = 512;
    VertexId h_workload;
    VertexId total_workload = nodes;
    int iter = 0;
    do 
    {   
        assignColor_op_warp<<<numBlock, numThread, sizeof(uint32_t)*64*(numThread/32), stream>>>(*this);
        detectConflicts_op_warp<<<320, 512, 0, stream>>>(*this);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, addNext, addNext_scan+1, nodes, stream);
        compactKernel<<<320, 512, 0, stream>>>(*this);
        CUDA_CHECK(cudaMemcpyAsync(&h_workload, workload, sizeof(VertexId), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemsetAsync(addNext, 0, sizeof(VertexId)*nodes, stream));
        //CUDA_CHECK(cudaMemsetAsync(forbiddenColors, 0, sizeof(bool)*fixSize*nodes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); 
        #ifdef ITER_WORKLOAD
        printf("iter %d, workload %d\n", iter, h_workload);
        #endif
        total_workload += h_workload;
        iter++;
    }
    while(h_workload>0);
    return total_workload;
}
#endif

