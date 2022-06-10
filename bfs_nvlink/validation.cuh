#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 


void find_mother(int vertex, int *depth, int *h_depth, Csr<int, int> &csr)
{
    for(int i=0; i<csr.edges; i++)
    {
        if(csr.column_indices[i] == vertex)
        {
            for(int j=0; j<csr.nodes+1; j++)
            {
                if(i>=csr.row_offset[j] && i<csr.row_offset[j+1])
                {
                    if(h_depth[j]!=depth[j]) {
                        cout << "vertex "<< vertex << " has mother "<< j << " indices " << i<<" csr.row_ffset "<< csr.row_offset[j]<<" , "<< csr.row_offset[j+1] << ", h_depth["<< j<<"]: "<< h_depth[j] <<" depth["<<j<<"]:"<< depth[j] <<endl;
                    //if(depth[j] == depth[vertex]-1) {
                    //    cout << "vertex "<< vertex << " has mother "<< j << " indices " << i<<" csr.row_ffset "<< csr.row_offset[j]<<" , "<< csr.row_offset[j+1] << ", h_depth["<< j<<"]: "<< h_depth[j] <<" depth["<<j<<"]:"<< depth[j] <<endl;
                    //    find_mother(j, depth, h_depth, csr);
                    }
                }
            }
        }
    }
}

template<typename BFSTYPE>
__global__ void warmup_bfs(BFSTYPE bfs, uint32_t *out)
{
    uint32_t sum=0;
    for(int i=TID; i<bfs.nodes+1; i=i+gridDim.x*blockDim.x)
    {
        uint32_t node = bfs.csr_offset[i];
        sum = sum+node;
    }

    for(int i=TID; i<bfs.edges; i=i+gridDim.x*blockDim.x)
    {
        uint32_t node = bfs.csr_indices[i];
        sum = sum+node;
    }
    out[TID] = sum;
}

template<typename BFSTYPE>
void warmup_bfs(BFSTYPE &bfs) {
    uint32_t *out;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int numBlock = 160, numThread = 512;
    CUDA_CHECK(cudaMalloc(&out, sizeof(uint32_t)*numBlock*numThread));
    warmup_bfs<<<numBlock, numThread, 0, stream>>>(bfs.DeviceObject(), out); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(out));
}

template<typename BFSTYPE, typename RUNTIMETYPE >
__global__ void warmup_bfs(BFSTYPE bfs, RUNTIMETYPE runtime, uint32_t *out)
{
    uint32_t sum=0;
    for(int i=TID; i<bfs.nodes+1; i=i+gridDim.x*blockDim.x)
    {
        uint32_t node = bfs.csr_offset[i];
        sum = sum+node;
    }

    for(int i=TID; i<bfs.edges; i=i+gridDim.x*blockDim.x)
    {
        uint32_t node = bfs.csr_indices[i];
        sum = sum+node;
    }
    out[TID] = sum;

    for(int i=TID; i<runtime.qp_capacity*runtime.num_qp; i=i+gridDim.x*blockDim.x)
        runtime.send_queue[i] = runtime.send_queue_init_value;
}

template<typename BFSTYPE, typename RUNTIMETYPE>
void warmup_bfs(BFSTYPE &bfs, RUNTIMETYPE &runtime) {
    uint32_t *out;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int numBlock = 160, numThread = 512;
    CUDA_CHECK(cudaMalloc(&out, sizeof(uint32_t)*numBlock*numThread));
    warmup_bfs<<<numBlock, numThread, 0, stream>>>(bfs.DeviceObject(), runtime.ContextDeviceObject(), out); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(out));
}

namespace host {
    template <typename VertexId, typename SizeT, typename BFSTYPE>
    void BFSValid(Csr<VertexId, SizeT> &csr,  BFSTYPE &bfs, VertexId source, int partition_idx, VertexId *new_labels_old)
    {
        VertexId nodes = csr.nodes;
    
        VertexId *h_depth= (VertexId*)malloc(sizeof(VertexId)*nodes);
        MALLOC_CHECK(h_depth);
        for(int i=0; i<nodes; i++)
            h_depth[i] = nodes+1;
    
        queue<VertexId> wl;
        wl.push(source);
        h_depth[source] = 0;
        uint32_t total_workload = 0;

        //finish init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            VertexId depth = h_depth[node_item];
            wl.pop();
            total_workload++;

            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                if(depth+1 < h_depth[dest_item]) 
                {
                    h_depth[dest_item] = depth+1;
                    wl.push(dest_item);
                }
            }
        }//while worklist

        int error=0.0;
        int intserre = 0;
        VertexId *d_depth = (VertexId *)malloc(sizeof(VertexId)*csr.nodes);
        CUDA_CHECK(cudaMemcpy(d_depth, bfs.depth, sizeof(VertexId)*csr.nodes, cudaMemcpyDeviceToHost));
        for(int i=0; i<bfs.nodes; i++)
        {
            if(partition_idx == 2 || partition_idx == 3) {
                VertexId old_vertex = new_labels_old[i+bfs.startNode];
                error = error + abs(h_depth[old_vertex] - d_depth[i+bfs.startNode]);
                if(h_depth[old_vertex]!=d_depth[i+bfs.startNode] && intserre < 10) 
                {
                    cout << "h_depth["<<old_vertex<<"]: "<< h_depth[old_vertex] << "  d_depth["<<i+bfs.startNode<<"]: "<< d_depth[i+bfs.startNode]<<endl;
		            intserre++;
        //            find_mother(i, h_depth, d_depth, csr);
                }

            }
            else {
                if(d_depth[i+bfs.startNode] < 0) cout << "vertex " << i+bfs.startNode << " depth "<< d_depth[i+bfs.startNode] << " h_depth "<< h_depth[i+bfs.startNode] << endl;
                assert(d_depth[i+bfs.startNode] >= 0);
                error = error + abs(h_depth[i+bfs.startNode] - d_depth[i+bfs.startNode]);
                if(h_depth[i+bfs.startNode]!=d_depth[i+bfs.startNode] && intserre < 2) 
                {
                    cout << "h_depth["<<i+bfs.startNode<<"]: "<< h_depth[i+bfs.startNode] << "  d_depth["<<i+bfs.startNode<<"]: "<< d_depth[i+bfs.startNode]<<endl;
		            intserre++;
                    find_mother(i+bfs.startNode, d_depth, h_depth, csr);
                }
            }
        }

        cout << "enqueued nodes: "<< wl.size() << endl;
        cout << "ERROR between CPU and GPU implimentation: "<< error <<  endl;
        cout << "CPU total workload "<< total_workload << endl;
    
        cout << "Print the first 20 depth: \n";
        cout << "host:\n";
        for(int i=bfs.startNode; i<20+bfs.startNode; i++)
        {
            if(partition_idx == 2 || partition_idx == 3)
                cout << h_depth[new_labels_old[i]] << " ";
            else 
                cout <<  h_depth[i] << " ";
        }
        cout << endl;
        cout << "device:\n";
        for(int i=bfs.startNode; i<20+bfs.startNode; i++)
            cout << d_depth[i] << " ";
        cout << endl;

        //std::string file_name; 

        //if(bfs.my_pe == 0)
        //    file_name = "gpu1.txt";
        //else if(bfs.my_pe == 1)
        //    file_name = "gpu2.txt";

        //ofstream outfile;
        //outfile.open(file_name);

        //for(int i=0; i<bfs.worklists.end[0]; i++)
        //{
        //    VertexId item = bfs.worklists.queue[i];
        //    if(d_depth[item]!=h_depth[item])
        //        outfile << "queue["<<i << "] " << item << " depth "<< d_depth[item] << " true depth "<< h_depth[item] << endl;
        //}
        //int *add_times;
        //add_times = (int *)malloc(sizeof(int)*bfs.totalNodes);
        //memset(add_times, 0, sizeof(int)*bfs.totalNodes);
        //for(int i=0; i<bfs.worklists.end[0]; i++)
        //   add_times[bfs.worklists.queue[i]+bfs.startNode]++; 
        
        //string file_name;
        //if(bfs.my_pe == 0)
        //    file_name = "freq_gpu1.txt";
        //else if(bfs.my_pe == 1)
        //    file_name = "freq_gpu2.txt";

        //ofstream outfile;
        //outfile.open(file_name);
        //for(int i=0; i<bfs.totalNodes; i++)
        //{
        //    if(add_times[i] > 100)
        //        outfile << "vertex "<< i << " add " << add_times[i] << " d_depth "<< d_depth[i] << " h_depth "<< h_depth[i] << endl;
        //}
        //outfile.close();

        //string file_name[10];
        //if(bfs.my_pe == 0)  {file_name[0] = "queue_gpu1_1.txt"; file_name[1] = "queue_gpu1_2.txt"; file_name[2] = "queue_gpu1_3.txt"; file_name[3] = "queue_gpu1_4.txt";
        //                     file_name[4] = "queue_gpu1_5.txt"; file_name[5] = "queue_gpu1_6.txt"; file_name[6] = "queue_gpu1_7.txt"; file_name[7] = "queue_gpu1_8.txt";
        //                     file_name[8] = "queue_gpu1_9.txt"; file_name[9] = "queue_gpu1_10.txt";}
        //else if(bfs.my_pe == 1) {file_name[0] = "queue_gpu2_1.txt"; file_name[1] = "queue_gpu2_2.txt"; file_name[2] = "queue_gpu2_3.txt"; file_name[3] = "queue_gpu2_4.txt";
        //                     file_name[4] = "queue_gpu2_5.txt"; file_name[5] = "queue_gpu2_6.txt"; file_name[6] = "queue_gpu2_7.txt"; file_name[7] = "queue_gpu2_8.txt";
        //                     file_name[8] = "queue_gpu2_9.txt"; file_name[9] = "queue_gpu2_10.txt";}
        
        //int chunck_size = (bfs.worklists.end[0]+9)/10;
        //for(int file_id = 0; file_id < 10; file_id++)
        //{
        //    ofstream outfile;
        //    outfile.open(file_name[file_id]);
        //    for(int i= file_id*chunck_size; i<min((file_id+1)*chunck_size ,bfs.worklists.end[0]); i++)
        //    {
        //        if(bfs.checksum[bfs.worklists.capacity+i] != bfs.checksum[i] || bfs.checksum[bfs.worklists.capacity+i] < 1 || bfs.checksum[i] < 1)
        //        {
        //            if(bfs.checksum[bfs.worklists.capacity+i] < bfs.checksum[i] ) 
        //                outfile << "slot "<< i << " vertex " << bfs.worklists.queue[i] << " push depth "<< bfs.checksum[bfs.worklists.capacity+i] << " pop depth "<< bfs.checksum[i] << " ERROR" << endl;
        //            else
        //                outfile << "slot "<< i << " vertex " << bfs.worklists.queue[i] << " push depth "<< bfs.checksum[bfs.worklists.capacity+i] << " pop depth "<< bfs.checksum[i] << " d_depth " << bfs.depth[bfs.worklists.queue[i]] << " h_depth "<< h_depth[bfs.worklists.queue[i]] << endl;
        //        }
        //    }
        //    outfile.close();
        //}
        free(h_depth);
    }//BFSValid

    template <typename VertexId, typename SizeT, typename BFSTYPE>
    void BFSValid2(Csr<VertexId, SizeT> &csr,  BFSTYPE &bfs, VertexId source)
    {
        VertexId nodes = csr.nodes;
    
        VertexId *h_depth= (VertexId*)malloc(sizeof(VertexId)*nodes);
        MALLOC_CHECK(h_depth);
        for(int i=0; i<nodes; i++)
            h_depth[i] = nodes+1;
    
        queue<VertexId> wl;
        wl.push(source);
        h_depth[source] = 0;
        uint32_t total_workload=0;

        //finish init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            VertexId depth = h_depth[node_item];
            wl.pop();
            total_workload++;

            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                if(depth+1 < h_depth[dest_item]) 
                {
                    h_depth[dest_item] = depth+1;
                    wl.push(dest_item);
                }
            }
        }//while worklist

        int error=0.0;
	    int error_print = 0;
        for(int i=0; i<bfs.nodes; i++)
        {
            error = error+ abs(h_depth[i]-bfs.depth[i]);
            if(h_depth[i]!=bfs.depth[i] && error_print < 10) 
            {
                cout << "h_depth["<<i<<"]: "<< h_depth[i] << "  bfs.depth["<<i<<"]: "<< bfs.depth[i]<<endl;
        //        find_mother(i, h_depth, bfs.depth, csr);
		        error_print++;
            }
        }

        cout << "enqueued nodes: "<< wl.size() << endl;
        cout << "ERROR between CPU and GPU implimentation: "<< error <<  endl;
        cout << "CPU total workload "<< total_workload << endl;
    
        cout << "Print the first 20 depth: \n";
        cout << "host:\n";
        for(int i=0; i<20; i++)
            cout << h_depth[i] << " ";
        cout << endl;
        cout << "device:\n";
        for(int i=0; i<20; i++)
            cout << bfs.depth[i] << " ";
        cout << endl;
        free(h_depth);
    }//BFSValid

    template <typename VertexId, typename SizeT, typename BFSTYPE, typename RT>
    void BFSValid_level(Csr<VertexId, SizeT> &csr, BFSTYPE &bfs, VertexId source, int partition_idx, VertexId *new_labels_old, VertexId level, RT &runtime)
    {
        VertexId nodes = csr.nodes;
    
        VertexId *h_depth= (VertexId*)malloc(sizeof(VertexId)*nodes);
        MALLOC_CHECK(h_depth);
        for(int i=0; i<nodes; i++)
            h_depth[i] = nodes+1;
    
        queue<VertexId> wl;
        wl.push(source);
        h_depth[source] = 0;
        uint32_t total_workload = 0;

        //finish init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            VertexId depth = h_depth[node_item];
            wl.pop();
            total_workload++;

            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                if(depth+1 < h_depth[dest_item]) 
                {
                    h_depth[dest_item] = depth+1;
                    if(depth+1 < level)
                        wl.push(dest_item);
                }
            }
        }//while worklist

        int error=0.0;
        int intserre = 0;
        VertexId *d_depth = (VertexId *)malloc(sizeof(VertexId)*csr.nodes);
        CUDA_CHECK(cudaMemcpy(d_depth, bfs.depth, sizeof(VertexId)*csr.nodes, cudaMemcpyDeviceToHost));
        uint32_t wl_end =0;
        CUDA_CHECK(cudaMemcpy(&wl_end, (uint32_t *)(bfs.worklists.end), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "wl_end " << wl_end <<std::endl;
        VertexId *d_wl_queue = (VertexId *)malloc(sizeof(VertexId)*wl_end);
        CUDA_CHECK(cudaMemcpy(d_wl_queue, bfs.worklists.queues, sizeof(VertexId)*wl_end, cudaMemcpyDeviceToHost));
        uint32_t recv_end =0;
        CUDA_CHECK(cudaMemcpy(&recv_end, runtime.recv_read_local_rq_end, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "recv end: "<< recv_end << std::endl;
        VertexId * d_recv_queue = (VertexId *)malloc(sizeof(VertexId)*recv_end);
        CUDA_CHECK(cudaMemcpy(d_recv_queue, runtime.recv_queue, sizeof(VertexId)*recv_end, cudaMemcpyDeviceToHost));

        int dev_level_count = 0;
        for(int i=0; i<bfs.nodes; i++)
        {
            if(partition_idx == 2 || partition_idx == 3) {
                VertexId old_vertex = new_labels_old[i+bfs.startNode];
                error = error + abs(h_depth[old_vertex] - d_depth[i+bfs.startNode]);
                if(h_depth[old_vertex]!=d_depth[i+bfs.startNode] && intserre < 10) 
                {
                    cout << "h_depth["<<old_vertex<<"]: "<< h_depth[old_vertex] << "  d_depth["<<i+bfs.startNode<<"]: "<< d_depth[i+bfs.startNode]<<endl;
		            intserre++;
        //            find_mother(i, h_depth, d_depth, csr);
                }

            }
            else {
                if(d_depth[i+bfs.startNode] < 0) cout << "vertex " << i+bfs.startNode << " depth "<< d_depth[i+bfs.startNode] << " h_depth "<< h_depth[i+bfs.startNode] << endl;
                assert(d_depth[i+bfs.startNode] >= 0);
                if(d_depth[i+bfs.startNode] < level) dev_level_count++;
                if(h_depth[i+bfs.startNode] < level) {
                    error = error + abs(h_depth[i+bfs.startNode] - d_depth[i+bfs.startNode]);
                    if(h_depth[i+bfs.startNode]!=d_depth[i+bfs.startNode] && intserre < 2) 
                    {
                        cout << "h_depth["<<i+bfs.startNode<<"]: "<< h_depth[i+bfs.startNode] << "  d_depth["<<i+bfs.startNode<<"]: "<< d_depth[i+bfs.startNode]<<endl;
		                intserre++;
                        //find_mother(i+bfs.startNode, d_depth, h_depth, csr);
                    }
                }
            }
        }

        //for(int l=0; l<level; l++) {
        //    for(int i=0; i<bfs.nodes; i++)
        //    {
        //        if(d_depth[i+bfs.startNode]!=h_depth[i+bfs.startNode] && h_depth[i+bfs.startNode] == l) {
        //            printf("d_depth[%6d] %6d, h_depth[%6d] %6d\n", i+bfs.startNode, d_depth[i+bfs.startNode], i+bfs.startNode, h_depth[i+bfs.startNode]);
        //            for(int j=0; j<wl_end; j++)
        //            {
        //                if(d_wl_queue[j] == i+bfs.startNode)
        //                    printf("node %6d in local worklist\n", i+bfs.startNode);
        //            }

        //            for(int j=0; j<recv_end; j++)
        //            {
        //                if(d_recv_queue[j] == i+bfs.startNode)
        //                    printf("node %6d in recve queue\n", i+bfs.startNode);
        //            }


        //            for(int j=0; j<csr.edges; j++)
        //            {
        //                if(csr.column_indices[j] == i+bfs.startNode)
        //                {
        //                    for(int k=0; k<csr.nodes+1; k++)
        //                    {
        //                        if(j>=csr.row_offset[k] && j<csr.row_offset[k+1])
        //                        {
        //                            //if(h_depth[k]!=d_depth[k] && h_depth[k] == l-1) {
        //                            if(h_depth[k] == l-1) {
        //                               cout << "vertex "<< i+bfs.startNode << " has mother "<< k << ", h_depth["<< k<<"]: "<< h_depth[k] <<" depth["<<k<<"]:"<< d_depth[k] <<endl;                                           
        //                               for(int w=0; w<wl_end; w++){
        //                                   if(d_wl_queue[w] == k)
        //                                        printf("mother %6d in local worklist %d\n", k, w);
        //                               }

        //                               for(int w=0; w<recv_end; w++){
        //                                   if(d_recv_queue[w] == k)
        //                                        printf("mother %6d in recv queue %d\n", k, w);
        //                               }

        //                            }
        //                            if(i+bfs.startNode == 22830 && h_depth[k] == l-1 && k>=bfs.startNode && k<bfs.endNode)
        //                                printf("22830 has local mother %d\n", k);
        //                            else if(i+bfs.startNode == 22830 && h_depth[k] == l-1)
        //                                printf("22830 has remote moterh %d, h_depth %d, d_depth %d\n", k, h_depth[k], d_depth[k]);
        //                        }
        //                    }
        //                }
        //            }

        //        }
        //    }
        //}

        cout << "enqueued nodes: "<< wl.size() << endl;
        cout << "dev less than level: "<< dev_level_count << endl;
        cout << "ERROR between CPU and GPU implimentation: "<< error <<  endl;
        cout << "CPU total workload "<< total_workload << endl;
    
        cout << "Print the first 20 depth: \n";
        cout << "host:\n";
        for(int i=bfs.startNode; i<20+bfs.startNode; i++)
        {
            if(partition_idx == 2 || partition_idx == 3)
                cout << h_depth[new_labels_old[i]] << " ";
            else 
                cout <<  h_depth[i] << " ";
        }
        cout << endl;
        cout << "device:\n";
        for(int i=bfs.startNode; i<20+bfs.startNode; i++)
            cout << d_depth[i] << " ";
        cout << endl;

        //std::string file_name; 

        //if(bfs.my_pe == 0)
        //    file_name = "gpu1.txt";
        //else if(bfs.my_pe == 1)
        //    file_name = "gpu2.txt";

        //ofstream outfile;
        //outfile.open(file_name);

        //for(int i=0; i<bfs.worklists.end[0]; i++)
        //{
        //    VertexId item = bfs.worklists.queue[i];
        //    if(d_depth[item]!=h_depth[item])
        //        outfile << "queue["<<i << "] " << item << " depth "<< d_depth[item] << " true depth "<< h_depth[item] << endl;
        //}
        //int *add_times;
        //add_times = (int *)malloc(sizeof(int)*bfs.totalNodes);
        //memset(add_times, 0, sizeof(int)*bfs.totalNodes);
        //for(int i=0; i<bfs.worklists.end[0]; i++)
        //   add_times[bfs.worklists.queue[i]+bfs.startNode]++; 
        
        //string file_name;
        //if(bfs.my_pe == 0)
        //    file_name = "freq_gpu1.txt";
        //else if(bfs.my_pe == 1)
        //    file_name = "freq_gpu2.txt";

        //ofstream outfile;
        //outfile.open(file_name);
        //for(int i=0; i<bfs.totalNodes; i++)
        //{
        //    if(add_times[i] > 100)
        //        outfile << "vertex "<< i << " add " << add_times[i] << " d_depth "<< d_depth[i] << " h_depth "<< h_depth[i] << endl;
        //}
        //outfile.close();

        //string file_name[10];
        //if(bfs.my_pe == 0)  {file_name[0] = "queue_gpu1_1.txt"; file_name[1] = "queue_gpu1_2.txt"; file_name[2] = "queue_gpu1_3.txt"; file_name[3] = "queue_gpu1_4.txt";
        //                     file_name[4] = "queue_gpu1_5.txt"; file_name[5] = "queue_gpu1_6.txt"; file_name[6] = "queue_gpu1_7.txt"; file_name[7] = "queue_gpu1_8.txt";
        //                     file_name[8] = "queue_gpu1_9.txt"; file_name[9] = "queue_gpu1_10.txt";}
        //else if(bfs.my_pe == 1) {file_name[0] = "queue_gpu2_1.txt"; file_name[1] = "queue_gpu2_2.txt"; file_name[2] = "queue_gpu2_3.txt"; file_name[3] = "queue_gpu2_4.txt";
        //                     file_name[4] = "queue_gpu2_5.txt"; file_name[5] = "queue_gpu2_6.txt"; file_name[6] = "queue_gpu2_7.txt"; file_name[7] = "queue_gpu2_8.txt";
        //                     file_name[8] = "queue_gpu2_9.txt"; file_name[9] = "queue_gpu2_10.txt";}
        
        //int chunck_size = (bfs.worklists.end[0]+9)/10;
        //for(int file_id = 0; file_id < 10; file_id++)
        //{
        //    ofstream outfile;
        //    outfile.open(file_name[file_id]);
        //    for(int i= file_id*chunck_size; i<min((file_id+1)*chunck_size ,bfs.worklists.end[0]); i++)
        //    {
        //        if(bfs.checksum[bfs.worklists.capacity+i] != bfs.checksum[i] || bfs.checksum[bfs.worklists.capacity+i] < 1 || bfs.checksum[i] < 1)
        //        {
        //            if(bfs.checksum[bfs.worklists.capacity+i] < bfs.checksum[i] ) 
        //                outfile << "slot "<< i << " vertex " << bfs.worklists.queue[i] << " push depth "<< bfs.checksum[bfs.worklists.capacity+i] << " pop depth "<< bfs.checksum[i] << " ERROR" << endl;
        //            else
        //                outfile << "slot "<< i << " vertex " << bfs.worklists.queue[i] << " push depth "<< bfs.checksum[bfs.worklists.capacity+i] << " pop depth "<< bfs.checksum[i] << " d_depth " << bfs.depth[bfs.worklists.queue[i]] << " h_depth "<< h_depth[bfs.worklists.queue[i]] << endl;
        //        }
        //    }
        //    outfile.close();
        //}
        free(h_depth);   
    }
} //namespace host
