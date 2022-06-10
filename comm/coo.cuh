#ifndef COO
#define COO
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <type_traits>

#include "../util/error_util.cuh"

template<typename VertexId>
struct Edge 
{
    VertexId row;
    VertexId col;
    
    Edge(){}
    Edge(VertexId row, VertexId col) : row(row), col(col){}
};

template<typename Edge>
bool RowFirstTupleCompare (Edge elem1, Edge elem2) {
    if (elem1.row < elem2.row) {
       // Sort edges by source node
       return true;
    } else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
        // Sort edgelists as well for coherence
        return true;
    }
    return false;
}

template<typename Edge>
bool ColumnFirstTupleCompare (Edge elem1, Edge elem2) {
    if (elem1.col < elem2.col) {
    // Sort edges by source node
        return true;
    } else if ((elem1.col == elem2.col) && (elem1.row < elem2.row)) {
        // Sort edgelists as well for coherence
        return true;
    }
    return false;
}

template<typename VertexId, typename SizeT>
struct Coo
{
    using edge_type = Edge<VertexId>;

    VertexId nodes=0;
    SizeT edges=0;
    edge_type* coos = NULL;
    bool start_from_zero = false;

    Coo(bool _start_0=false): start_from_zero(_start_0){}
    Coo(VertexId _nodes, SizeT _edges, bool _start_0=false):nodes(_nodes), edges(_edges), start_from_zero(_start_0) 
    {}

    void AllocCoo()
    {
        coos = (edge_type *)malloc(sizeof(edge_type)*edges);
        std::cout << "alloc: "<< edges << std::endl;
        MALLOC_CHECK(coos);
    }

    void Print()
    {
        std::cout << "nodes: " << nodes << "  edges: "<< edges <<  " vertexId start from zero? "<< start_from_zero << std::endl;
        std::cout << "First edge:" << coos[0].row << "-->"<<coos[0].col<<std::endl;
        std::cout << "Last edge:" << coos[edges-1].row << "-->"<<coos[edges-1].col<<std::endl;
    }

    void BuildCooFromMtx(char *file_in)
    {
        std::ifstream infile(file_in);
        std::string line;
        while(!infile.eof())
        {
            getline(infile, line);
            if(line.substr(0,1) != "%")
                break;
        }
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" "));
        VertexId vertices[3];
        SizeT i=0; 
        for(auto& s:tokens)
        {
            std::stringstream geek(s);
            geek >> vertices[i];
            i++;
        }
        nodes = vertices[0];
        edges = vertices[2];
        std::cout << "nodes: " << nodes << "  edges: "<< edges <<  std::endl;

        AllocCoo();

        i=0;
    //    while(infile >> vertices[0] >> vertices[1]) {
        while(!infile.eof()){
            getline(infile, line);
            std::stringstream(line) >> vertices[0] >> vertices[1];
        //    std::cout << i << " ("<<vertices[0] << ", "<< vertices[1] << ")\n"; 
            if(!start_from_zero)
            {
                vertices[0]--;
                vertices[1]--;
            }
            coos[i].row = vertices[0];
            coos[i].col = vertices[1];
            i++;
            if(i == edges)
                break;
        }
        if(i != edges) 
            std::cout << "ERROR: coos is length "<<i<<" is not equal to number of edges given by the file:" << edges << " \n";
    } //BuildCooFromMtx
};

   
#endif
