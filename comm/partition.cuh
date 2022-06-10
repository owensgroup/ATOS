#include <stdint.h>
#include <cmath>
#include <random>
#include <list>
#include <metis.h>

#include "../util/error_util.cuh"
#include "../util/sort_omp.cuh"
#include "../util/util.cuh"

using namespace std;

namespace partitioner {
    typedef std::mt19937 Engine;
    typedef std::uniform_int_distribution<uint32_t> Distribution;

    template<typename SizeT, typename ValueT>
    struct SortNode {
        SizeT posit;
        ValueT value;
    };

    template <typename SizeT, typename ValueT>
    bool Compare_SortNode(SortNode<SizeT, ValueT> A, SortNode<SizeT, ValueT> B) {
        return (A.value < B.value);
    }

    template<typename VertexId, typename SizeT>
    __global__ void relable(VertexId *old_labels_new, Csr<VertexId, SizeT> my_csr)
    {
        for(SizeT i = TID; i<my_csr.edges; i=i+blockDim.x*gridDim.x)
        {
            VertexId orig_vertex = (my_csr.column_indices)[i];
            (my_csr.column_indices)[i] = old_labels_new[orig_vertex];
        }
    }

    template<typename VertexId, typename SizeT>
    VertexId metis(int n_pes, int my_pe, Csr<VertexId, SizeT> &csr, Csr<VertexId, SizeT> &my_csr, VertexId *new_labels_old, VertexId *scheme, VertexId source, 
        bool ifwrite=false, char *file=NULL)
    {
        idx_t totalNodes = csr.nodes;
        idx_t ncons = 1;
        idx_t nsubgraphs = n_pes;
        idx_t objval;
        idx_t *part;
        part = (idx_t *)malloc(sizeof(idx_t)*totalNodes);
        if(part==nullptr) cout << "fail to allocate part with size "<< sizeof(idx_t)*totalNodes << " bytes\n";
        memset(part, 0, sizeof(idx_t)*totalNodes);
        idx_t *offset, *indices;
        offset = (idx_t*)malloc(sizeof(idx_t)*(totalNodes+1));
        indices = (idx_t *)malloc(sizeof(idx_t)*csr.edges);
        if(offset==nullptr) cout << "fail to allocate offset  with size "<< sizeof(idx_t)*(totalNodes+1) << " bytes\n";
        if(indices==nullptr) cout << "fail to allocate indices with size "<< sizeof(idx_t)*(csr.edges) << " bytes\n";
        for(int i=0; i<totalNodes+1; i++)
            offset[i] = static_cast<idx_t>(csr.row_offset[i]);
        for(int i=0; i<csr.edges; i++)
            indices[i] = static_cast<idx_t>(csr.column_indices[i]);

        std::cout << "Start Metis partition\n";
        int ret = METIS_PartGraphKway(
            &totalNodes, &ncons, offset, indices,
            NULL, NULL, NULL, &nsubgraphs,
            NULL, NULL, NULL, &objval, part);
        assert(ret == 1 && "Metis partition failed\n");
        std::cout << "Metis partition done\n";
        if(ifwrite) {
            std::cout << "Metis use " <<  sizeof(idx_t)*8 << "bits, export partition data to file "<< file << std::endl;
            write_binary(file, part, totalNodes);
        }
            
        VertexId *old_labels_new;
        CUDA_CHECK(cudaMallocManaged(&old_labels_new, sizeof(VertexId)*totalNodes));
        for(int i=0; i<n_pes+1;i++)
            scheme[i] = 0;
        for(int i=0; i<totalNodes; i++) {
            old_labels_new[i] = scheme[part[i]+1];
            scheme[part[i]+1]++;
        }
        for(int i=1; i<n_pes+1; i++)
            scheme[i] = scheme[i] + scheme[i-1];
        for(int i=0; i<totalNodes; i++) {
            old_labels_new[i] = scheme[part[i]]+old_labels_new[i];
            new_labels_old[old_labels_new[i]] = i;
        }

//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<n_pes+1; i++) std::cout << "scheme["<< i<<"] "<< scheme[i] << "\n";);
//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<totalNodes; i++) std::cout << "old "<< i<<" new "<< old_labels_new[i] << "\n";);
//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<totalNodes; i++) std::cout << "new "<< i<<" old "<< new_labels_old[i] << "\n";);

        SizeT partition_edges[n_pes];
        partition_edges[my_pe] = 0;
        VertexId vertex_start = scheme[my_pe];
        VertexId vertex_end = scheme[my_pe+1];
        for(int v=vertex_start; v<vertex_end; v++) {
            VertexId old_vertex = new_labels_old[v];
            partition_edges[my_pe] = partition_edges[my_pe]+((csr.row_offset)[old_vertex+1]-(csr.row_offset)[old_vertex]);
        }

//        SERIALIZE_PRINT(my_pe, n_pes, std::cout << "Partition edges "<< partition_edges[my_pe] << std::endl);
        my_csr.nodes = vertex_end-vertex_start;
        my_csr.edges = partition_edges[my_pe];
        my_csr.AllocCsr();
        my_csr.row_offset[0] = 0;

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            VertexId num_nodes = scheme[my_pe+1]-scheme[my_pe];
            SizeT node_end = (long long)(num_nodes)*(thread_num+1) / num_threads + scheme[my_pe];
            SizeT node_start = (long long)(num_nodes)*thread_num / num_threads + scheme[my_pe];

            for(int v = node_start; v< node_end; v++)
            {
                VertexId orig_vertex = new_labels_old[v];
                (my_csr.row_offset)[v-scheme[my_pe]+1]= (csr.row_offset)[orig_vertex+1]-(csr.row_offset)[orig_vertex];
            }
        }
        for(int v = 0; v<(scheme[my_pe+1]-scheme[my_pe]); v++)
            (my_csr.row_offset)[v+1] = (my_csr.row_offset)[v]+(my_csr.row_offset)[v+1];
        
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            for(int v = thread_num; v<(scheme[my_pe+1]-scheme[my_pe]); v=v+num_threads)
            {
                VertexId orig_vertex = new_labels_old[scheme[my_pe]+v];
                VertexId orig_vertex_offset = (csr.row_offset)[orig_vertex];
                SizeT orig_vertex_length = (csr.row_offset)[orig_vertex+1]- orig_vertex_offset;
                SizeT new_vertex_length = (my_csr.row_offset)[v+1]-(my_csr.row_offset)[v];
                if(orig_vertex_length != new_vertex_length) cout <<"ERROR:: new " << v << " old "<< orig_vertex<< " new_length "<< new_vertex_length << " old_length "<< orig_vertex_length << std::endl;
                assert(orig_vertex_length == new_vertex_length);
                SizeT i = 0;
                for(int c = (my_csr.row_offset)[v]; c< (my_csr.row_offset)[v+1]; c++)
                {
                    (my_csr.column_indices)[c] = (csr.column_indices)[orig_vertex_offset+i];
                    i++;
                }
            }
        }


        relable<<<1, 1>>>(old_labels_new, my_csr);
        CUDA_CHECK(cudaDeviceSynchronize());

        //check partition correctness
//        VertexId accum=0;
//        for(VertexId i=0; i<totalNodes; i++) {
//            if(part[i] == my_pe) {
//                assert(csr.row_offset[i+1]-csr.row_offset[i] == my_csr.row_offset[accum+1]-my_csr.row_offset[accum]);
//                for(int n=0; n<csr.row_offset[i+1]-csr.row_offset[i]; n++) {
//                    VertexId new_vertex = my_csr.column_indices[my_csr.row_offset[accum]+n];
//                    assert(csr.column_indices[csr.row_offset[i]+n] == new_labels_old[new_vertex]);
//                }
//                accum++;
//            }
//        }
//
//        SERIALIZE_PRINT(my_pe, n_pes, std::cout << "PASS Partition Validation\n");

        VertexId new_source = old_labels_new[source];
        CUDA_CHECK(cudaFree(old_labels_new));
        free(offset);
        free(indices);
        free(part);
		return new_source;
    }

    template<typename VertexId, typename SizeT>
    VertexId metis(int n_pes, int my_pe, Csr<VertexId, SizeT> &csr, Csr<VertexId, SizeT> &my_csr, VertexId *new_labels_old, VertexId *scheme, VertexId source, 
        char * p_file, bool ifbinary=true)
    {
        // this is ugly, should not like this. I did this because I generate the metis mega data for soc, osm-eur, road_usa and indochina, hollywood in 32 bits and twitter in 64 bits, BAD
        idx_t totalNodes = csr.nodes;
        VertexId *old_labels_new;
        CUDA_CHECK(cudaMallocManaged(&old_labels_new, sizeof(VertexId)*totalNodes));
        
        assert(sizeof(idx_t)*8 == 64);
        cout << "reading METIS parititon data in 64 bits\n";
        idx_t *part=NULL;
        if(ifbinary)
            part = read_binary<idx_t>(p_file, totalNodes);
        else 
            part = read_file<idx_t>(p_file, totalNodes);
        for(int i=0; i<n_pes+1;i++)
            scheme[i] = 0;
        for(int i=0; i<totalNodes; i++) {
            old_labels_new[i] = scheme[part[i]+1];
            scheme[part[i]+1]++;
        }
        for(int i=1; i<n_pes+1; i++)
            scheme[i] = scheme[i] + scheme[i-1];
        for(int i=0; i<totalNodes; i++) {
            old_labels_new[i] = scheme[part[i]]+old_labels_new[i];
            new_labels_old[old_labels_new[i]] = i;
        }
        free(part);

//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<n_pes+1; i++) std::cout << "scheme["<< i<<"] "<< scheme[i] << "\n";);
//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<totalNodes; i++) std::cout << "old "<< i<<" new "<< old_labels_new[i] << "\n";);
//        SERIALIZE_PRINT(my_pe, n_pes, for(int i=0; i<totalNodes; i++) std::cout << "new "<< i<<" old "<< new_labels_old[i] << "\n";);

        SizeT partition_edges[n_pes];
        partition_edges[my_pe] = 0;
        VertexId vertex_start = scheme[my_pe];
        VertexId vertex_end = scheme[my_pe+1];
        for(int v=vertex_start; v<vertex_end; v++) {
            VertexId old_vertex = new_labels_old[v];
            partition_edges[my_pe] = partition_edges[my_pe]+((csr.row_offset)[old_vertex+1]-(csr.row_offset)[old_vertex]);
        }

//        SERIALIZE_PRINT(my_pe, n_pes, std::cout << "Partition edges "<< partition_edges[my_pe] << std::endl);
        my_csr.nodes = vertex_end-vertex_start;
        my_csr.edges = partition_edges[my_pe];
        my_csr.AllocCsr();
        my_csr.row_offset[0] = 0;

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            VertexId num_nodes = scheme[my_pe+1]-scheme[my_pe];
            SizeT node_end = (long long)(num_nodes)*(thread_num+1) / num_threads + scheme[my_pe];
            SizeT node_start = (long long)(num_nodes)*thread_num / num_threads + scheme[my_pe];

            for(int v = node_start; v< node_end; v++)
            {
                VertexId orig_vertex = new_labels_old[v];
                (my_csr.row_offset)[v-scheme[my_pe]+1]= (csr.row_offset)[orig_vertex+1]-(csr.row_offset)[orig_vertex];
            }
        }
        for(int v = 0; v<(scheme[my_pe+1]-scheme[my_pe]); v++)
            (my_csr.row_offset)[v+1] = (my_csr.row_offset)[v]+(my_csr.row_offset)[v+1];
        
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            for(int v = thread_num; v<(scheme[my_pe+1]-scheme[my_pe]); v=v+num_threads)
            {
                VertexId orig_vertex = new_labels_old[scheme[my_pe]+v];
                VertexId orig_vertex_offset = (csr.row_offset)[orig_vertex];
                SizeT orig_vertex_length = (csr.row_offset)[orig_vertex+1]- orig_vertex_offset;
                SizeT new_vertex_length = (my_csr.row_offset)[v+1]-(my_csr.row_offset)[v];
                if(orig_vertex_length != new_vertex_length) cout <<"ERROR:: new " << v << " old "<< orig_vertex<< " new_length "<< new_vertex_length << " old_length "<< orig_vertex_length << std::endl;
                assert(orig_vertex_length == new_vertex_length);
                SizeT i = 0;
                for(int c = (my_csr.row_offset)[v]; c< (my_csr.row_offset)[v+1]; c++)
                {
                    (my_csr.column_indices)[c] = (csr.column_indices)[orig_vertex_offset+i];
                    i++;
                }
            }
        }


        relable<<<1, 1>>>(old_labels_new, my_csr);
        CUDA_CHECK(cudaDeviceSynchronize());

        //check partition correctness
//        VertexId accum=0;
//        for(VertexId i=0; i<totalNodes; i++) {
//            if(part[i] == my_pe) {
//                assert(csr.row_offset[i+1]-csr.row_offset[i] == my_csr.row_offset[accum+1]-my_csr.row_offset[accum]);
//                for(int n=0; n<csr.row_offset[i+1]-csr.row_offset[i]; n++) {
//                    VertexId new_vertex = my_csr.column_indices[my_csr.row_offset[accum]+n];
//                    assert(csr.column_indices[csr.row_offset[i]+n] == new_labels_old[new_vertex]);
//                }
//                accum++;
//            }
//        }
//
//        SERIALIZE_PRINT(my_pe, n_pes, std::cout << "PASS Partition Validation\n");

        VertexId new_source = old_labels_new[source];
        CUDA_CHECK(cudaFree(old_labels_new));
        return new_source; 
    }

    template<typename VertexId, typename SizeT>
    VertexId random(int n_pes, int my_pe, Csr<VertexId, SizeT> &csr, Csr<VertexId, SizeT> &my_csr, VertexId *new_labels_old, VertexId *scheme, VertexId source, int partition_seed = 12321)
    {
        VertexId totalNodes = csr.nodes;
        if(n_pes == 0 || totalNodes == 0)
        {
            cout << "Error: didn't provide num_device info or nodes info\n";
            return source;
        }

        VertexId partition_size = (totalNodes+n_pes-1)/n_pes;
        scheme[0] = 0;
        scheme[n_pes] = totalNodes;
        for(int i=1; i<n_pes; i++)
            scheme[i] = partition_size*i;
    
        for(int i=0; i<n_pes+1; i++)
            cout << scheme[i] << " ";
        cout << endl;

       
        SortNode<VertexId, uint32_t> * sort_list;
        sort_list = (SortNode<VertexId, uint32_t> *)malloc(sizeof(SortNode<VertexId, uint32_t>)*totalNodes);

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            SizeT node_start = (long long)(totalNodes)*thread_num / num_threads;
            SizeT node_end = (long long)(totalNodes)*(thread_num + 1) / num_threads;
            unsigned int thread_seed = partition_seed + 754 * thread_num;
            Engine engine(thread_seed);
            Distribution distribution(0, UINT32_MAX);
            for (SizeT v = node_start; v < node_end; v++) {
                uint32_t x = distribution(engine);
                sort_list[v].value = x;
                sort_list[v].posit = v;
            }
        }

        util::omp_sort(sort_list + 0, totalNodes, Compare_SortNode<VertexId, uint32_t>);
        
        SizeT partition_edges[n_pes];
        partition_edges[my_pe] = 0;
        VertexId vertex_start = scheme[my_pe];
        VertexId vertex_end = scheme[my_pe+1];
        for(int v = vertex_start; v<vertex_end; v++) {
            VertexId vertex = sort_list[v].posit;
            partition_edges[my_pe] = partition_edges[my_pe]+((csr.row_offset)[vertex+1]-(csr.row_offset)[vertex]);
        }

        my_csr.nodes=vertex_end - vertex_start;
        my_csr.edges = partition_edges[my_pe];
        my_csr.AllocCsr();
        my_csr.row_offset[0]=0;

        VertexId * old_labels_new;
        CUDA_CHECK(cudaMallocManaged(&old_labels_new, sizeof(VertexId)*csr.nodes));

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            SizeT node_start = (long long)(totalNodes)*thread_num / num_threads;
            SizeT node_end = (long long)(totalNodes)*(thread_num + 1) / num_threads;
            for(int v = node_start; v< node_end; v++)
            {
                old_labels_new[sort_list[v].posit] = v;
                new_labels_old[v] = sort_list[v].posit;
            }
        }

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            VertexId num_nodes = scheme[my_pe+1]-scheme[my_pe];
            SizeT node_end = (long long)(num_nodes)*(thread_num+1) / num_threads + scheme[my_pe];
            SizeT node_start = (long long)(num_nodes)*thread_num / num_threads + scheme[my_pe];

            for(int v = node_start; v< node_end; v++)
            {
                VertexId orig_vertex = sort_list[v].posit;
                (my_csr.row_offset)[v-scheme[my_pe]+1]= (csr.row_offset)[orig_vertex+1]-(csr.row_offset)[orig_vertex];
            }
        }
        for(int v = 0; v<(scheme[my_pe+1]-scheme[my_pe]); v++)
            (my_csr.row_offset)[v+1] = (my_csr.row_offset)[v]+(my_csr.row_offset)[v+1]; 

        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            for(int v = thread_num; v<(scheme[my_pe+1]-scheme[my_pe]); v=v+num_threads)
            {
                VertexId orig_vertex_offset = (csr.row_offset)[sort_list[scheme[my_pe]+v].posit];
                SizeT orig_vertex_length = (csr.row_offset)[sort_list[scheme[my_pe]+v].posit+1]-(csr.row_offset)[sort_list[scheme[my_pe]+v].posit];
                SizeT new_vertex_length = (my_csr.row_offset)[v+1]-(my_csr.row_offset)[v];
                if(orig_vertex_length != new_vertex_length) cout <<"ERROR:: vertex length not equal\n";
                SizeT i = 0;
                for(int c = (my_csr.row_offset)[v]; c< (my_csr.row_offset)[v+1]; c++)
                {
                    (my_csr.column_indices)[c] = (csr.column_indices)[orig_vertex_offset+i];
                    i++;
                }
            }
        }

        relable<<<160, 512>>>(old_labels_new, my_csr);
        CUDA_CHECK(cudaDeviceSynchronize());

  //      for(int p=0; p<n_pes; p++)
  //      {
  //          if(p == my_pe)
  //          {
  //              cout << "new_lables\n";
  //              for(int v=0; v<totalNodes; v++)
  //                  cout << new_lables[v] << " ";
  //              cout<<endl;

  //              cout << "row_offset\n";
  //              for(int v=0; v<my_csr.nodes+1; v++)
  //                  cout << my_csr.row_offset[v] << " ";
  //              cout << endl;

  //              cout << "colum_indices\n";
  //              for(int v=0; v<my_csr.edges; v++)
  //                  cout << my_csr.column_indices[v] << " ";
  //              cout << endl;
  //          }
  //          nvshmem_barrier_all();
  //      }
        VertexId new_source = old_labels_new[source];
        free(sort_list);
        CUDA_CHECK(cudaFree(old_labels_new));
        return new_source;
    }

    template<typename VertexId, typename Size>
    void vertices(int n_pes, int my_pe, Csr<VertexId, Size> &csr, Csr<VertexId, Size> &my_csr, VertexId *partition_scheme)
    {
        //compute scheme;
        VertexId totalNodes = csr.nodes;
        if(n_pes == 0 || totalNodes == 0)
        {
            cout << "Error: didn't provide num_device info or nodes info\n";
            return;
        }

        VertexId partition_size = (totalNodes+n_pes-1)/n_pes;
        partition_scheme[0] = 0;
        partition_scheme[n_pes] = totalNodes;
        for(int i=1; i<n_pes; i++)
            partition_scheme[i] = partition_size*i;

        my_csr.nodes=partition_scheme[my_pe+1]-partition_scheme[my_pe];
        my_csr.edges = csr.row_offset[partition_scheme[my_pe+1]]-csr.row_offset[partition_scheme[my_pe]];
        my_csr.AllocCsr();

        for(int i=0; i<my_csr.nodes+1; i++)
            my_csr.row_offset[i] = csr.row_offset[partition_scheme[my_pe]+i]-csr.row_offset[partition_scheme[my_pe]];
        for(int i=0; i<my_csr.edges; i++)
        {
            my_csr.column_indices[i] = csr.column_indices[csr.row_offset[partition_scheme[my_pe]]+i];
        }
    }
    template<typename VertexId, typename Size>
    void vertices(int n_pes, int my_pe, Csr<VertexId, Size> &csr, Csr<VertexId, Size> &my_csr, VertexId *partition_scheme, float ratio)
    {
        //compute scheme;
        VertexId totalNodes = csr.nodes;
        if(n_pes == 0 || totalNodes == 0)
        {
            cout << "Error: didn't provide num_device info or nodes info\n";
            return;
        }

        partition_scheme[0] = 0;
        partition_scheme[n_pes] = totalNodes;
        for(int i=1; i<n_pes; i++) {
            partition_scheme[i] = partition_scheme[i-1]+totalNodes*ratio;
            assert(partition_scheme[i] < totalNodes);
        }

        my_csr.nodes=partition_scheme[my_pe+1]-partition_scheme[my_pe];
        my_csr.edges = csr.row_offset[partition_scheme[my_pe+1]]-csr.row_offset[partition_scheme[my_pe]];
        my_csr.AllocCsr();

        for(int i=0; i<my_csr.nodes+1; i++)
            my_csr.row_offset[i] = csr.row_offset[partition_scheme[my_pe]+i]-csr.row_offset[partition_scheme[my_pe]];
        for(int i=0; i<my_csr.edges; i++)
        {
            my_csr.column_indices[i] = csr.column_indices[csr.row_offset[partition_scheme[my_pe]]+i];
        }
    }
    template<typename VertexId, typename SizeT>
    void edges(int n_pes, int my_pe, Csr<VertexId, SizeT> &csr, Csr<VertexId, SizeT> &my_csr, VertexId *scheme)
    {
        SizeT totalEdges = csr.edges;
        if(n_pes == 0 || totalEdges == 0)
        {
            cout << "Error: didn't provide num devices info or edges info\n";
            return;
        }
       
        SizeT partition_size = (totalEdges+n_pes-1)/n_pes;
        scheme[0] = 0;
        scheme[n_pes] = csr.nodes;
       
        omp_set_num_threads(16);
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads(); 
            int chunk_start = (csr.nodes)*thread_num/num_threads;
            int chunk_end = (csr.nodes)*(thread_num+1)/num_threads;
        
            for(int p=1; p<=n_pes; p++) {
                if(csr.row_offset[chunk_start]>partition_size*p || csr.row_offset[chunk_end] <= partition_size*p)
                    continue;
                for(int i=chunk_start; i<chunk_end; i++) {
                    if(csr.row_offset[i] <= partition_size*p && csr.row_offset[i+1] > partition_size*p)
                    {
                        if(partition_size*p - csr.row_offset[i] <= csr.row_offset[i+1]-partition_size*p)
                            scheme[p] = i; 
                        else scheme[p] = i+1;
                        break;
                    }
                }
            }
        }
        my_csr.nodes = scheme[my_pe+1] - scheme[my_pe];
        my_csr.edges = csr.row_offset[scheme[my_pe+1]]-csr.row_offset[scheme[my_pe]];
        my_csr.AllocCsr();
       
        for(int i=0; i<my_csr.nodes+1; i++)
            my_csr.row_offset[i] = csr.row_offset[scheme[my_pe]+i]-csr.row_offset[scheme[my_pe]];
        for(int i=0; i<my_csr.edges; i++)
        {
            my_csr.column_indices[i] = csr.column_indices[csr.row_offset[scheme[my_pe]]+i];
        }
    }
} //partitioner
