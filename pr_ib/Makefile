# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information


NVSHMEM_HOME ?= /ccs/home/yuxinc/pkg/nvshmem_2.2.1_build
CUDA_HOME ?= /sw/summit/cuda/11.4.2
MPI_HOME ?= /sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-11.1.0/spectrum-mpi-10.4.0.3-20210112-6kg6anupjriji6pnvijebfn7ha5vsqp2 
METIS64_HOME ?=/ccs/home/yuxinc/pkg/metis-5.1.0/build-gcc-11.1.0
MPISUPPORT ?= 1
DEBUG ?= 0
VERBOSE ?= 1

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE ?= -gencode=arch=compute_70,code=sm_70
NVCC_CG_GENCODE ?= -gencode=arch=compute_70,code=sm_70

TESTCUFLAGS  := -ccbin $(CXX) 

# Use addprefix so that we can specify more than one path
TESTLDFLAGS :=

TESTINC := -I$(CUDA_INC) -I$(NVSHMEM_HOME)/include -I${METIS64_HOME}/include

ifeq ($(MPISUPPORT), 1)
MPI_LIB = -lmpi_ibm
TESTINC += -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT
endif

ifeq ($(DEBUG), 0)
TESTCUFLAGS  += -O3 
else
TESTCUFLAGS  += -O0 -g -G -lineinfo
endif

ifneq ($(VERBOSE), 0)
TESTCUFLAGS  += -lineinfo -Xptxas -v --expt-extended-lambda -Xcompiler -fopenmp
endif


TESTLDFLAGS += -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart -L${METIS64_HOME}/lib -lmetis -lgomp
ifeq ($(MPISUPPORT), 1)
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIB)
endif
 
.PHONY : default 
default : examples

EXAMPLECUSRCFILES := pr.cu 

INTERBATCHSIZE := 16
FETCHSIZE := 32
WAITTIMES := 32
CUPERFBIN   := $(patsubst %.cu, %, $(filter %.cu, $(EXAMPLECUSRCFILES)))
CXXPERFBIN  := $(patsubst %.cpp, %, $(filter %.cpp, $(PERFCXXSRCFILES)))
TESTBIN  := $(patsubst %.cu, %_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE}, $(filter %.cu, $(EXAMPLECUSRCFILES)))

$(info $(CUPERFBIN))
$(info $(CXXPERFBIN))
$(info $(TESTBIN))

examples : $(TESTBIN)

%_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE} : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -DFETCHSIZE=${FETCHSIZE} -DWAITTIMES=${WAITTIMES} -DINTERBATCHSIZE=${INTERBATCHSIZE} -rdc=true $< -o $@ $(TESTLDFLAGS)


FETCHSIZE := 256
TESTBIN  := $(patsubst %.cu, %_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE}, $(filter %.cu, $(EXAMPLECUSRCFILES)))

$(info $(TESTBIN))
examples : $(TESTBIN)

%_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE} : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -DFETCHSIZE=${FETCHSIZE} -DWAITTIMES=${WAITTIMES} -DINTERBATCHSIZE=${INTERBATCHSIZE} -rdc=true $< -o $@ $(TESTLDFLAGS)


FETCHSIZE := 512
TESTBIN  := $(patsubst %.cu, %_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE}, $(filter %.cu, $(EXAMPLECUSRCFILES)))

$(info $(TESTBIN))
examples : $(TESTBIN)

%_${FETCHSIZE}_$(WAITTIMES)_${INTERBATCHSIZE} : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -DFETCHSIZE=${FETCHSIZE} -DWAITTIMES=${WAITTIMES} -DINTERBATCHSIZE=${INTERBATCHSIZE} -rdc=true $< -o $@ $(TESTLDFLAGS)


clean : 
	rm -rf $(TESTBIN)
