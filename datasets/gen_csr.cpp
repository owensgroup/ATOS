#include <iostream>
#include <string>

#include "csr.cuh"

using namespace std;

int main(int argc, char *argv[])
{
     char *input_file = NULL;
     bool start_from_0 = false;
     bool directed = true;
     bool ifvalid = false;
     if(argc == 1)
     {
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -d <if graph is directed=directed> -valid <if validate the generated csr file>\n";
         exit(0);
     }
     if(argc > 1)
         for(int i=1; i<argc; i++) {
             if(string(argv[i]) == "-f")
                input_file = argv[i+1];
             else if(string(argv[i]) == "-s")
                 start_from_0 = stoi(argv[i+1]);
             else if(string(argv[i]) == "-d")
                 directed = stoi(argv[i+1]);
             else if(string(argv[i]) == "-valid")
		 ifvalid = stoi(argv[i+1]);
         }
     if(input_file == NULL)
     {
         cout << "input file is needed\n";
         cout<< "./test -f <file> -s <file vertex ID start from 0?=false> -d <if graph is directed=directed>\n";
         exit(0);
     }

    std::cout << "file: "<< input_file << " start from 0: " << start_from_0 << " directed? "<< directed << " validate generated file? "<< ifvalid << std::endl;
    std::string str_file(input_file);
    CSR<int, int> csr(start_from_0, directed);
    if(str_file.substr(str_file.length()-4) == ".mtx")
    {
        csr.BuildFromMtx(input_file);
        csr.Print();
         //       csr.PrintCSR();
        csr.WriteToBinary(input_file);
	cout << "Start to write to binary file\n";
	if(ifvalid) {
	bool res = csr.ValidWithMtx(input_file);
        if(!res) std::cout <<"Validation fail\n";
        if(!res) exit(1);
        cout << "PASS Validation\n";
	} 

    }
    return 0;
}
