CXX = g++
MKLROOT = ~/miniconda3/envs/computation_playground/lib
DNNL_ROOT = ./dnnl_lnx_1.6.0_cpu_gomp
TARGET = $(patsubst %.cpp, build/%.out, $(wildcard *.cpp))

MKL_FLAGS = -Wl,--start-group $(MKLROOT)/libmkl_intel_lp64.so $(MKLROOT)/libmkl_core.so $(MKLROOT)/libmkl_gnu_thread.so -Wl,--end-group -ldl -lpthread -fopenmp -lm
MKL_FLAGS += -I ${MKLROOT}/../include
FLAGS += ${MKL_FLAGS}

DNNL_FLAGS = -I ${DNNL_ROOT}/include -I ${DNNL_ROOT}/examples
DNNL_FLAGS += -L ${DNNL_ROOT}/lib -ldnnl
FLAGS += ${DNNL_FLAGS}


debug : FLAGS += -g -O0
debug : all

release: FLAGS += -Ofast -DNDEBUG
release : all

all : $(TARGET)

${TARGET}:build/%.out:%.cpp
	        ${CXX} $< ${FLAGS} -o $@