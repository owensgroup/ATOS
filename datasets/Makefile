CC=g++

FLAGS=-O3 --std=c++11 -fopenmp 
LDFLAGS =-lgomp 

SOURCES = gen_csr.cu 
TARGETS := $(patsubst %.cu, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
		$(CC) -o $@ $^ $(FLAGS)

clean:
		rm -fv $(TARGETS)
