#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 

using namespace std;

namespace host {

    template <typename VertexId, typename SizeT, typename PRTYPE>
    void PrValid(Csr<VertexId, SizeT> &csr,  PRTYPE &pr)
    {
        float lambda = pr.lambda;
        float epsilon = pr.epsilon;
    
        VertexId nodes = csr.nodes;
    
        float *h_rank = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_rank);
        float *h_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(float)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId neigh_len = csr.row_offset[i+1]-csr.row_offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.column_indices[csr.row_offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            wl.pop();
            h_rank[node_item] = h_rank[node_item]+h_res[node_item];
            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            float res_owner = h_res[node_item];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                float res_old = h_res[dest_item];
                h_res[dest_item] = h_res[dest_item] + res_owner*lambda/(float)(destEnd-destStart); 
                if(res_old < epsilon && h_res[dest_item] > epsilon)
                    wl.push(dest_item);
            }
            h_res[node_item] = 0.0;
    
        }//while worklist
    
        float totalRank=0.0;
        float totalRes = 0.0;
        for(VertexId i=0; i<nodes; i++)
        {
            totalRank = totalRank + h_rank[i];
            totalRes= totalRes+ h_res[i];
        }
        
     //   for(int i=0; i<nodes; i++)
     //       cout << h_rank[i] << " ";
     //   cout << endl;

        cout << "CPU total mass: " << totalRank + totalRes/(1.0-lambda) << " CPU total res: " << totalRes << " CPU total rank: " << totalRank << endl;
    
        float error=0.0;
        float sum_rank=0.0;
        float sum_res = 0.0;
        uint32_t large = 0;
        float *d_rank;
        d_rank = (float *)malloc(sizeof(float)*pr.nodes);
        CUDA_CHECK(cudaMemcpy(d_rank, pr.rank, sizeof(float)*pr.nodes, cudaMemcpyDeviceToHost));
        for(VertexId i=0; i<pr.nodes; i++)
        {
     //       error = error + abs(check_rank[i]-h_rank[i]/totalRank);
            error = error + abs(d_rank[i]-h_rank[i]);
            sum_rank = sum_rank + d_rank[i];
        }
        cout << "GPU rank: sum of rank "<< sum_rank << " error from CPU "<< error << "\n";
        error = 0.0;
        float *d_res;
        d_res = (float *)malloc(sizeof(float)*pr.nodes);
        CUDA_CHECK(cudaMemcpy(d_res, pr.res, sizeof(float)*pr.nodes, cudaMemcpyDeviceToHost));
        for(VertexId i=0; i<pr.nodes; i++)
        {
            error = error + abs(d_res[i]-h_res[i]);
            sum_res = sum_res + d_res[i];
            if(d_res[i] > epsilon)
                large++;

        }
        cout << "GPU res: sum of res "<< sum_res << " error from CPU "<< error << " "<<large << " number of res larger than "<< epsilon << "\n";
        cout << endl;

        cout<<"GPU sum_rank: "<< sum_rank << " GPU sum_res: "<< sum_res << " GPU total mass: "<< sum_rank+sum_res/(1.0-lambda) << endl;
        //if(error > 0.01) cout << "FAILE\n";

        cout << "Print the first 20 res: \n";
        cout << "host:\n";
        for(int i=0; i<20; i++)
            cout << h_rank[i] << " ";
        cout << endl;
        cout << "device:\n";
        for(int i=0; i<20; i++)
            cout << d_rank[i] << " ";
        cout << endl;
    }//PrValid

    template <typename VertexId, typename SizeT, typename PRTYPE>
    void PrInitValid(Csr<VertexId, SizeT> &csr,  PRTYPE &pr)
    {
        float lambda = pr.lambda;
    
        VertexId nodes = csr.nodes;
    
        float *h_rank = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_rank);
        float *h_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(float)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId neigh_len = csr.offset[i+1]-csr.offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.indices[csr.offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        float *check_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(check_res);
        CUDA_CHECK(cudaMemcpy(check_res, pr.res, sizeof(float)*nodes, cudaMemcpyDeviceToHost));
        float error=0.0;
        float max_error = 0.0;
        int max_idx = -1;
        for(int i=0; i<pr.nodes; i++)
        {
            error = error + abs(check_res[i]-h_res[i]);
            max_error = max(max_error, abs(check_res[i]-h_res[i]));
            if(max_error == abs(check_res[i]-h_res[i]))
                max_idx = i;
        }
        cout <<"\nerror :" << error << endl;
        cout << "max error: "<< max_error << endl;
        cout << "max error, host: " << h_res[max_idx] << " device: "<< check_res[max_idx]<< endl;
        CUDA_CHECK(cudaMemcpy(check_res, pr.rank, sizeof(float)*pr.nodes, cudaMemcpyDeviceToHost));
        for(int i=0; i<pr.nodes; i++)
            if(check_res[i]!=1.0-lambda)
                cout << "Rank: " << check_res[i] << " not equal to " << 1.0-lambda << endl;
        free(check_res);
    }//PrInitValid

    template<typename VertexId, typename Rank>
	struct PrRes {
		Rank cpu_total_res;
		Rank cpu_total_rank;
		Rank gpu_total_res;
		Rank gpu_total_rank;

		Rank total_rank_d;
		Rank total_res_d;
		Rank max_rank_d;
		Rank max_rank_value_h;
		Rank max_rank_value_d;

		VertexId num_large_epsilon;
		VertexId cpu_process=0;

		vector<Rank> h_ranks;
		vector<Rank> d_ranks;

		PrRes() {};
		PrRes(Rank cpu_t_res, Rank cpu_t_rank, Rank gpu_t_res, Rank gpu_t_rank, Rank t_rank_d, Rank max_r_d, 
			  VertexId max_rank_id_h, VertexId max_rank_id_d, VertexId large): cpu_total_res(cpu_t_res), 
			cpu_total_rank(cpu_t_rank), gpu_total_res(gpu_t_res), gpu_total_rank(gpu_t_rank), total_rank_d(t_rank_d),
			max_rank_d(max_r_d), max_rank_value_h(max_rank_id_h), max_rank_value_d(max_rank_id_d), num_large_epsilon(large)
		{}

		template<typename PRTYPE>
		void print_detail(int permute_idx, PRTYPE &pr)
		{
			printf("cpu total res %.4f, cpu total rank %.4f, total mass %.4f\n", cpu_total_res, cpu_total_rank, cpu_total_rank + cpu_total_res/(1.0-pr.lambda));
			printf("gpu total res %.4f, gpu total rank %.4f, total mass %.4f\n", gpu_total_res, gpu_total_rank, gpu_total_rank + gpu_total_res/(1.0-pr.lambda));
			printf("res diff %.4f, rank diff %.4f, max rank %.4f (host %.4f, device %.4f)\n", total_res_d, total_rank_d, max_rank_d, max_rank_value_h, max_rank_value_d);
			printf("%d number of res larger than epsilon, cpu totalwork %d\n", num_large_epsilon, cpu_process);

			cout << "Print the first 20 rank: \n";
        	cout << "host:\n";
			for(auto rank: h_ranks)
				cout << rank << " ";
			cout << endl;
        	cout << "device:\n";
			for(auto rank: d_ranks)
        	    cout << rank << " ";
        	cout << endl;
		}

		template<typename PRTYPE>
		void print(int permute_idx, PRTYPE &pr)
		{
			//printf("%d number of res larger than epsilon\n", num_large_epsilon);	
			if(num_large_epsilon < 5) printf("PASS VALIDATION\n");
			else printf("VALIDATION FAIL\n");
		}
	};

    template <typename VertexId, typename SizeT, typename PRTYPE>
    PrRes<VertexId, float> PrValid(Csr<VertexId, SizeT> &csr,  PRTYPE &pr, int permute_idx, VertexId *new_labels_old)
    {
        float lambda = pr.lambda;
        float epsilon = pr.epsilon;
    
        VertexId nodes = csr.nodes;
    
        float *h_rank = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_rank);
        float *h_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(float)*nodes);
        uint32_t total_proc = 0;
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
            total_proc++;
    
            VertexId neigh_len = csr.row_offset[i+1]-csr.row_offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.column_indices[csr.row_offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            wl.pop();
            h_rank[node_item] = h_rank[node_item]+h_res[node_item];
            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            float res_owner = h_res[node_item];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                float res_old = h_res[dest_item];
                h_res[dest_item] = h_res[dest_item] + res_owner*lambda/(float)(destEnd-destStart); 
                if(res_old < epsilon && h_res[dest_item] > epsilon) {
                    wl.push(dest_item);
                    total_proc++;
                }
            }
            h_res[node_item] = 0.0;
    
        }//while worklist
    
        PrRes<VertexId, float> pr_res;

        float totalRank=0.0;
        float totalRes = 0.0;
        for(VertexId i=0; i<nodes; i++)
        {
            totalRank = totalRank + h_rank[i];
            totalRes= totalRes+ h_res[i];
        }
        
        pr_res.cpu_total_res = totalRes;
		pr_res.cpu_total_rank = totalRank; 
		pr_res.cpu_process = total_proc;
        //for(int i=0; i<nodes; i++)
        //    cout << h_rank[i] << " ";
        //cout << endl;

        //cout << "CPU total processed node "<< total_proc << endl;
        //cout << "CPU total mass: " << totalRank + totalRes/(1.0-lambda) << " CPU total res: " << totalRes << " CPU total rank: " << totalRank << endl;
        
        float error=0.0;
        float sum_rank=0.0;
        float sum_res = 0.0;
        uint32_t large = 0;
        float max_rank_d =0; 
        int max_error_id_h = -1;
		int max_error_id_d = -1;
        float *dev_rank = (float *)malloc(sizeof(float)*pr.nodes);
        CUDA_CHECK(cudaMemcpy(dev_rank, pr.rank, sizeof(float)*pr.nodes, cudaMemcpyDeviceToHost));
        for(VertexId i=0; i<pr.nodes; i++)
        {
            if(permute_idx == 2 || permute_idx == 3)
            {
                VertexId old_vertex = new_labels_old[i+pr.startNode];
                max_rank_d = max(max_rank_d,abs(dev_rank[i]-h_rank[old_vertex]));
                //if(abs(pr.rank[i]-h_rank[old_vertex]) >= 0.5) cout << "rank has large error: old rank["<< old_vertex << "] "<< h_rank[old_vertex] << " new rank[ "<< i+pr.startNode << ", "<< i<<"] "<< pr.rank[i]<< endl;
                if(max_rank_d == abs(dev_rank[i]-h_rank[old_vertex]))
				{
                    max_error_id_d = i;
                    max_error_id_h = old_vertex;
				}
                error = error + abs(dev_rank[i]-h_rank[old_vertex]);
            }
            else
            {
                error = error + abs(dev_rank[i]-h_rank[i+pr.startNode]);
                max_rank_d = max(max_rank_d,abs(dev_rank[i]-h_rank[i+pr.startNode]));
                if(max_rank_d == abs(dev_rank[i]-h_rank[i+pr.startNode]))
				{
					max_error_id_d = i;
					max_error_id_h = i+pr.startNode;
				}
            }
            sum_rank = sum_rank + dev_rank[i];
        }

        pr_res.gpu_total_rank = sum_rank;
		pr_res.max_rank_d = max_rank_d;
		pr_res.total_rank_d = error;
		pr_res.max_rank_value_h = h_rank[max_error_id_h];
		pr_res.max_rank_value_d = dev_rank[max_error_id_d]; 
        //cout << "max rank differenc: "<< max_rank_d << endl;
        //cout << "GPU rank: sum of rank "<< sum_rank << " error from CPU "<< error << "\n";
        float *dev_res = (float *)malloc(sizeof(float)*pr.totalNodes);
        CUDA_CHECK(cudaMemcpy(dev_res, pr.res, sizeof(float)*pr.totalNodes, cudaMemcpyDeviceToHost));
        error = 0.0;
        for(VertexId i=0; i<pr.totalNodes; i++)
        {
            if(i>=pr.startNode && i<=pr.endNode)
            {
                if(permute_idx == 2 || permute_idx == 3) {
                    VertexId old_vertex = new_labels_old[i];
                    error = error + abs(dev_res[i]-h_res[old_vertex]);
                }
                else 
                    error = error + abs(dev_res[i]-h_res[i]);
            }
            sum_res = sum_res + dev_res[i];
            //cout << "res[" << i <<"]: "<< pr.res[i] << endl;
            if(dev_res[i] > epsilon)
            {
                //cout << "res["<<i<<"]: "<< pr.res[i] << endl;
                large++;
            }

        }

        pr_res.gpu_total_res = sum_res;
		pr_res.num_large_epsilon = large;
		pr_res.total_res_d = error;

        for(int i=0; i<min(20, int(nodes)); i++)
		{
			if(permute_idx == 2 || permute_idx == 3) {
				VertexId old_vertex = new_labels_old[pr.startNode+i];
				pr_res.h_ranks.push_back(h_rank[old_vertex]);
			}
			else pr_res.h_ranks.push_back(h_rank[i+pr.startNode]);

			pr_res.d_ranks.push_back(dev_rank[i]);
		}

        free(dev_rank);
		free(dev_res);
		return pr_res;
        //cout << "GPU res: sum of res "<< sum_res << " error from CPU "<< error << " "<<large << " number of res larger than "<< epsilon << "\n";
        //cout << endl;

        //cout<<"GPU sum_rank: "<< sum_rank << " GPU sum_res: "<< sum_res << " GPU total mass: "<< sum_rank+sum_res/(1.0-lambda) << endl;
        ////if(error > 0.01) cout << "FAILE\n";

        //cout << "Print the first 20 rank: \n";
        //cout << "host:\n";
        //for(int i=0; i<min(20, int(nodes)); i++)
        //{
        //    if(permute_idx == 2 || permute_idx == 3) {
        //        VertexId old_vertex = new_labels_old[pr.startNode+i];
        //        cout << h_rank[old_vertex] << " ";
        //    }
        //    else cout << h_rank[i+pr.startNode] << " ";
        //}
        //cout << endl;
        //cout << "device:\n";
        //for(int i=0; i<min(20, int(pr.nodes)); i++)
        //    cout << dev_rank[i] << " ";
        //cout << endl;
    }//PrValid

    template <typename VertexId, typename SizeT, typename PRTYPE>
    void PrInitValid(Csr<VertexId, SizeT> &csr,  PRTYPE &pr, int parition_idx, VertexId * new_labels_old)
    {
        float lambda = pr.lambda;
    
        VertexId nodes = csr.nodes;
    
        float *h_rank = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_rank);
        float *h_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(float)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId neigh_len = csr.row_offset[i+1]-csr.row_offset[i];
            for(int j=0; j<neigh_len; j++)
            {
                VertexId neighbor = csr.column_indices[csr.row_offset[i]+j];
                h_res[neighbor] = h_res[neighbor] + (1.0-lambda)*lambda/neigh_len;
            }
        } //for vertices
        //finish res and rank init

        float *check_res = (float *)malloc(sizeof(float)*nodes);
        MALLOC_CHECK(check_res);
        CUDA_CHECK(cudaMemcpy(check_res, pr.res, sizeof(float)*nodes, cudaMemcpyDeviceToHost));
        float error=0.0;
        float max_error = 0.0;
        int max_error_idx = -1;
        for(int i=pr.startNode; i<pr.endNode; i++)
        {
            if(parition_idx==2 || parition_idx == 3)
            {
                VertexId old_vertex = new_labels_old[i];
                error = error + abs(check_res[i]-h_res[old_vertex]);
                max_error = max(max_error, abs(check_res[i]-h_res[old_vertex]));
                if(max_error == abs(check_res[i]-h_res[i]))
                    max_error_idx = i;
            }
            else {
                error = error + abs(check_res[i]-h_res[i]);
                max_error = max(max_error, abs(check_res[i]-h_res[i]));
                if(max_error == abs(check_res[i]-h_res[i]))
                    max_error_idx = i;
            }
        }
        cout <<"\n["<< pr.my_pe << "] error :" << error << ", max error: "<< max_error << " with index "<< 
        max_error_idx << " dev res " << check_res[max_error_idx] << " host res " << h_res[max_error_idx] << endl;
        CUDA_CHECK(cudaMemcpy(check_res, pr.rank, sizeof(float)*pr.nodes, cudaMemcpyDeviceToHost));
        for(int i=0; i<pr.nodes; i++)
            if(check_res[i]!=1.0-lambda)
                cout << "["<<pr.my_pe << "] Rank: " << check_res[i] << " not equal to " << 1.0-lambda << endl;
        free(check_res);
    }//PrInitValid

    template<typename PRTYPE>
    __global__ void warmup_pr(PRTYPE pr, uint32_t *out)
    {
        uint32_t sum=0;
        for(int i=TID; i<pr.nodes+1; i=i+gridDim.x*blockDim.x)
        {
            uint32_t node = pr.csr_offset[i];
            sum = sum + node;
        }

        for(int i=TID; i<pr.edges; i=i+gridDim.x*blockDim.x)
        {
            uint32_t node = pr.csr_indices[i];
            sum = sum + node;
        }
        out[TID] = sum;
    }

    template<typename PRTYPE>
    void warmup_pr(PRTYPE &pr) {
        uint32_t *out;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int numBlock = 160, numThread = 512;
        CUDA_CHECK(cudaMalloc(&out, sizeof(uint32_t)*numBlock*numThread));
        warmup_pr<<<numBlock, numThread, 0, stream>>>(pr.DeviceObject(), out); 
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(out));
    }
} //namespace