CXX = g++
TARGET = $(patsubst %.cpp, build/%.out, $(wildcard *.cpp))
FLAGS = -std=c++11 -O3 -DNDEBUG
SSE_FLAGS = -msse -msse2 -msse3 -mfpmath=sse
FLAGS += ${SSE_FLAGS}
all:$(TARGET)

${TARGET}:build/%.out:%.cpp
	        ${CXX} $< ${FLAGS} -o $@