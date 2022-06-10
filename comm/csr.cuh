#ifndef CSR
#define CSR
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <cuda.h>

#include "coo.cuh"

#include "../util/error_util.cuh"
#include "../util/sort_omp.cuh"

template<typename VertexId, typename SizeT>
struct Csr
{
    using edge_type = Edge<VertexId>;
    size_t nodes=0;              //Number of nodes in the graph
    size_t edges=0;                 //Number of nodes in the graph
    size_t ave_degree=0;            //average degree
     
    bool direct=true;               //If graph is directed

    VertexId *column_indices=NULL;  //Indices of column of each row, length:edges
    SizeT *row_offset=NULL;         //Offset to each row, started from 0, length:nodes+1

    void AllocCsr()
    {
   //     CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMallocManaged(&column_indices, sizeof(VertexId)*edges));
        CUDA_CHECK(cudaMallocManaged(&row_offset, sizeof(SizeT)*(nodes+1)));
    }

    void release() {
        CUDA_CHECK(cudaFree(column_indices));
        CUDA_CHECK(cudaFree(row_offset));
    }

    void PrintCsr()
    {
        std::cout << "Vertices: "<< nodes << " Edges: "<< edges<<"\n";
    //    std::cout << "row offset: \n";
    //    for(int i=0; i<nodes+1; i++)
    //        std::cout << row_offset[i] << " ";
    //    std::cout << "\n";
    //    
    //    std::cout << "column indices: \n";
    //    for(int i=0; i<edges; i++)
    //        std::cout << column_indices[i] << " ";
    //    std::cout<< "\n";
    }

    // Write to File, content readable
    void WriteToFile(char * file_in)
    {
        std::string file(file_in); 
        std::string file_name = file.substr(0, file.length()-4);
    //    std::cout <<"file_name " <<file_name << std::endl;
        char rows[256], cols[256];

        sprintf(rows, "%s.rows", file_name.c_str());
        sprintf(cols, "%s.cols", file_name.c_str());

        std::ofstream rows_output(rows);
        if (rows_output.is_open())
        {
            std::copy(row_offset, row_offset + nodes + 1,
                        std::ostream_iterator<SizeT>(rows_output, "\n"));
            rows_output.close();
        }

        std::ofstream cols_output(cols);
        if (cols_output.is_open())
        {
            std::copy(column_indices, column_indices + edges,
                        std::ostream_iterator<VertexId>(cols_output, "\n"));
            cols_output.close();
        }
    }

    void WriteToBinary(char * file_in)
    {
        std::string file(file_in);
        std::string file_name = file.substr(0, file.length()-4);
        char output[256];
        sprintf(output, "%s.csr", file_name.c_str());

        std::ofstream fout(output);
        if(fout.is_open())
        {
            fout.write((char *)&nodes, sizeof(VertexId));
            fout.write((char *)&edges, sizeof(SizeT));
            fout.write((char *)row_offset, sizeof(SizeT)*(nodes+1));
            fout.write((char *)column_indices, sizeof(VertexId)*(edges));
            fout.close();
        }
    }

    void ReadFromBinary(char * file_in)
    {
        std::ifstream fin(file_in);
        if(fin.is_open())
        {
            fin.read((char *)&nodes, sizeof(VertexId));
            fin.read((char *)&edges, sizeof(SizeT));
            AllocCsr();
            fin.read((char *)row_offset, sizeof(SizeT)*(nodes+1));
            fin.read((char *)column_indices, sizeof(VertexId)*edges);
            fin.close();
        }
    }

    void FromCooToCsr(Coo<VertexId, SizeT>& _coo)
    {
        nodes = _coo.nodes;
        edges = _coo.edges;
        AllocCsr();
        util::omp_sort(_coo.coos, edges, RowFirstTupleCompare<edge_type>);
        SizeT *edge_offsets = NULL;
        SizeT *edge_counts  = NULL;
        #pragma omp parallel
        {
            int num_threads  = omp_get_num_threads();
            int thread_num   = omp_get_thread_num();
            if (thread_num == 0)
            {
                edge_offsets = new SizeT[num_threads+1];
                edge_counts  = new SizeT[num_threads+1];
            }
            #pragma omp barrier
            SizeT edge_start = (long long)(edges) * thread_num / num_threads;
            SizeT edge_end   = (long long)(edges) * (thread_num + 1) / num_threads;
            SizeT node_start = (long long)(nodes) * thread_num / num_threads;
            SizeT node_end   = (long long)(nodes) * (thread_num + 1) / num_threads;
            edge_type *new_coo   = (edge_type*) malloc (sizeof(edge_type) * (edge_end - edge_start));
            SizeT edge       = edge_start;
            SizeT new_edge   = 0;
            for (edge = edge_start; edge < edge_end; edge++)
            {
                VertexId col = _coo.coos[edge].col;
                VertexId row = _coo.coos[edge].row;
                if ((col != row) && (edge == 0 || col != _coo.coos[edge - 1].col || row != _coo.coos[edge - 1].row))
                {
                    new_coo[new_edge].col = col;
                    new_coo[new_edge].row = row;
                    new_edge++;
                }
            }
            edge_counts[thread_num] = new_edge;
            for (VertexId node = node_start; node < node_end; node++)
                row_offset[node] = -1;
            #pragma omp barrier
            #pragma omp single
            {
                edge_offsets[0] = 0;
                for (int i = 0; i < num_threads; i++)
                    edge_offsets[i + 1] = edge_offsets[i] + edge_counts[i];
                row_offset[0] = 0;
            }
            SizeT edge_offset = edge_offsets[thread_num];
            VertexId first_row = new_edge > 0 ? new_coo[0].row : -1;
            SizeT pointer = -1;
            for (edge = 0; edge < new_edge; edge++)
            {
                SizeT edge_  = edge + edge_offset;
                VertexId row = new_coo[edge].row;
                row_offset[row + 1] = edge_ + 1;
                if (row == first_row) pointer = edge_ + 1;
                // Fill in rows up to and including the current row
                column_indices[edge + edge_offset] = new_coo[edge].col;
            }
            #pragma omp barrier
           if (edge_start > 0 && _coo.coos[edge_start].row == _coo.coos[edge_start - 1].row) 
               // same row as previous thread
               if (edge_end == edges || _coo.coos[edge_end].row != _coo.coos[edge_start].row) 
                   // first row ends at this thread
               {
                   row_offset[first_row + 1] = pointer;
               }
               #pragma omp barrier
               // Fill out any trailing edgeless nodes (and the end-of-list element)
               if (row_offset[node_start] == -1)
               {
                   VertexId i = node_start;
                   while (row_offset[i] == -1) i--;
                   row_offset[node_start] = row_offset[i];
               }
               for (VertexId node = node_start + 1; node < node_end; node++)
                   if (row_offset[node] == -1)
                   {
                       row_offset[node] = row_offset[node - 1];
                   }
               if (thread_num == 0) edges = edge_offsets[num_threads];
               free(new_coo); new_coo = NULL;
        }
        row_offset[nodes] = edges;
        delete[] edge_offsets; edge_offsets = NULL;
        delete[] edge_counts ; edge_counts  = NULL;
    }
};   
#endif
