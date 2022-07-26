#ifndef CSR_F
#define CSR_F
#include <iostream>
#include <list>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

#define MAX_SZ (0xffffffffffffffff)

template<typename VertexId, typename SizeT>
struct CSR
{
    VertexId nodes=0;
    SizeT edges=0;
    bool start_from_zero=false;
    bool directed=true;

    SizeT *offset=NULL;
    VertexId *indices=NULL;

    CSR() {}
    CSR(bool _start_0, bool _directed): start_from_zero(_start_0),directed(_directed) {}
    CSR(VertexId _nodes, SizeT _edges, bool _start_0=false, bool _directed=true):
        nodes(_nodes), edges(_edges), start_from_zero(_start_0), directed(_directed) {}

    void Alloc()
    {
	offset = (SizeT *)malloc(sizeof(SizeT)*(nodes+1));
	indices = (VertexId *)malloc(sizeof(VertexId)*edges);
	assert(offset!=NULL);
	assert(indices!=NULL);
    }

    void release()
    {
        if(offset!=NULL)
	    free(offset);
        if(indices!=NULL)
	    free(indices);
    }

    void Print()
    {
        std::cout << "nodes: " << nodes << "  edges: "<< edges <<  " vertexId start from zero? "<< start_from_zero << " directed? "<< directed << std::endl;
    }

    void PrintCSR()
    {

        SizeT lastoffset = offset[0];
        std::cout << "0";
        int same = 0;
        for(VertexId i=1; i<nodes+1; i++)
        {
            if(offset[i]!=lastoffset)
            {
                if(same==0)
                    std::cout<< ": "<<lastoffset <<", "<<i;
                else
                    std::cout<< "-"<<i-1<<": "<<lastoffset <<", "<<i;
                same = 0;
            }
            else same++;
            lastoffset = offset[i];
        }
        std::cout << ": "<< lastoffset <<std::endl;
        std::cout << std::endl;
        for(int i=0; i<edges; i++)
            std::cout << i << ": "<< indices[i] << " , ";
        std::cout << std::endl;
    }

    void BuildFromMtx(char *file_in)
    {
        std::ifstream infile(file_in);
        std::string line;
        while(!infile.eof())
        {
            getline(infile, line);
            if(line.substr(0,1) != "%")
                break;
        }
        VertexId vertices[3];
        std::stringstream(line) >> vertices[0] >> vertices[1] >> vertices[2];
        std::cout <<vertices[0] << " "<< vertices[1] << " "<< vertices[2] << std::endl;

        nodes = vertices[0];
        edges = vertices[2];

        std::list<VertexId> * coo_array = new std::list<VertexId>[nodes];

        SizeT i=0;
        vertices[0] = VertexId(MAX_SZ);
        vertices[1] = VertexId(MAX_SZ);

        while(!infile.eof()){
            getline(infile, line);
            i++;
            std::stringstream(line) >> vertices[0] >> vertices[1];
            assert(vertices[0]!=VertexId(MAX_SZ));
            assert(vertices[0]<=nodes);
            assert(vertices[1]!=VertexId(MAX_SZ));
            assert(vertices[1]<=nodes);
            if(!start_from_zero)
            {
                vertices[0]--;
                vertices[1]--;
            }
            if(i<10)
            std::cout << i << " ("<<vertices[0] << ", "<< vertices[1] << ")\n"; 
            if(vertices[0]!=vertices[1])
            {
                coo_array[vertices[0]].push_back(vertices[1]);
                if(!directed)
                    coo_array[vertices[1]].push_back(vertices[0]);
            }
            if(i == edges)
                break;
            vertices[0]=VertexId(MAX_SZ);
            vertices[1]=VertexId(MAX_SZ);
        } // while

        SizeT num_edges = 0;
        for(VertexId i=0; i<nodes; i++)
        {
            coo_array[i].sort();
            VertexId lastone = VertexId(MAX_SZ);
            for(auto item=coo_array[i].begin(); item!=coo_array[i].end(); item++)
            {
                if(lastone == *item)
                    item = coo_array[i].erase(item);
                lastone = *item;
            }
        }

        for(VertexId i=0; i<nodes; i++)
        {
            num_edges = num_edges + coo_array[i].size();
        }
        std::cout << "Collected "<< num_edges << " number of edges\n";
        edges = num_edges;
        Alloc();
        offset[0] = 0;
        for(VertexId i=0; i<nodes; i++)
        {
            offset[i+1] = offset[i]+coo_array[i].size();
            if(!coo_array[i].empty())
            {
                VertexId k=0;
                for( VertexId neighbor: coo_array[i])
                {
                    indices[offset[i]+k] = neighbor; 
                    k++;
                }
            }
        }
        assert(offset[nodes]==edges);
    } //BuildFromMtx

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
            std::copy(offset, offset + nodes + 1,
                        std::ostream_iterator<SizeT>(rows_output, "\n"));
            rows_output.close();
        }

        std::ofstream cols_output(cols);
        if (cols_output.is_open())
        {
            std::copy(indices, indices + edges,
                        std::ostream_iterator<VertexId>(cols_output, "\n"));
            cols_output.close();
        }
    }

    void WriteToBinary(char * file_in)
    {
        std::string file(file_in);
        std::string file_name = file.substr(0, file.length()-4);
        char output[256];
	if(directed)
        sprintf(output, "%s_di.csr", file_name.c_str());
	else
        sprintf(output, "%s_ud.csr", file_name.c_str());

        std::ofstream fout(output);
        if(fout.is_open())
        {
            fout.write((char *)&nodes, sizeof(VertexId));
            fout.write((char *)&edges, sizeof(SizeT));
            fout.write((char *)offset, sizeof(SizeT)*(nodes+1));
            fout.write((char *)indices, sizeof(VertexId)*(edges));
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
            Alloc();
            fin.read((char *)offset, sizeof(SizeT)*(nodes+1));
            fin.read((char *)indices, sizeof(VertexId)*edges);
            fin.close();
        }
    }

    bool ValidWithMtx(char *file_in)
    {
        std::ifstream infile(file_in);
        std::string line;
        while(!infile.eof())
        {
            getline(infile, line);
            if(line.substr(0,1) != "%")
                break;
        }
        VertexId vertices[3];
        std::stringstream(line) >> vertices[0] >> vertices[1] >> vertices[2];
        assert(vertices[0]==nodes);

        std::list<VertexId> * coo_array = new std::list<VertexId>[vertices[0]];
        SizeT i=0;
        vertices[0] = VertexId(MAX_SZ);
        vertices[1] = VertexId(MAX_SZ);


        while(!infile.eof()){
            getline(infile, line);
            i++;
            std::stringstream(line) >> vertices[0] >> vertices[1];
            assert(vertices[0]!=VertexId(MAX_SZ));
            assert(vertices[0]<=nodes);
            assert(vertices[1]!=VertexId(MAX_SZ));
            assert(vertices[1]<=nodes);
            if(!start_from_zero)
            {
                vertices[0]--;
                vertices[1]--;
            }
            coo_array[vertices[0]].push_back(vertices[1]);
            if(vertices[0]!=vertices[1])
            {
                SizeT offset_start = offset[vertices[0]]; 
                SizeT offset_end = offset[vertices[0]+1]; 
                bool find = false;
                for(SizeT j=offset_start; j<offset_end; j++)
                    if(indices[j]==vertices[1])
                    {
                        find = true;
                        break;
                    }
                if(!find) {
                    std::cout << " directed part, didn't find edge: ("<< vertices[0]<<", "<< vertices[1]<<") "<<std::endl;
                    return false;    
                }
                if(!directed)
                {
                    offset_start = offset[vertices[1]];
                    offset_end = offset[vertices[1]+1];
                    find = false;
                    for(SizeT j=offset_start; j<offset_end; j++)
                        if(indices[j]==vertices[0])
                        {
                            find = true;
                            break;
                        }
                    if(!find) return false;
                }
            }
            if(i == vertices[2])
                break;
            vertices[0]=VertexId(MAX_SZ);
            vertices[1]=VertexId(MAX_SZ);
        } // while

        for(VertexId j=0; j<nodes; j++)
        {
            SizeT offset_start = offset[j];
            SizeT offset_end = offset[j+1];
            for(SizeT k = offset_start; k<offset_end; k++)
            {
                VertexId item = indices[k];
                bool find=false;
                for(auto neighbor:coo_array[j])
                    if(neighbor == item)
                    {
                        find = true;
                        break;
                    }
                if(!find) {
                    if(!directed)
                    {
                        for(auto neighbor:coo_array[item])
                            if(neighbor == j)
                            {
                                find = true;
                                break;
                            }
                    }
                }
                if(!find)
                {
                    std::cout << " didn't fine edge from CSR ("<< i<<", "<< item <<") in mtx file\n";
                    return false;
                }
            }
        }
        return true;
    }
};
#endif
