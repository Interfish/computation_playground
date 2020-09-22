CXX = g++
TARGET = $(patsubst %.cpp, build/%.out, $(wildcard *.cpp))
FLAGS = -std=c++11 #-DNDEBUG
SSE_FLAGS = -msse -msse2 -msse3 -mfpmath=sse
FLAGS += ${SSE_FLAGS}

debug : FLAGS += -g -O0
debug : all

release: FLAGS += -Ofast -DNDEBUG
release : all

all:$(TARGET)

${TARGET}:build/%.out:%.cpp
	        ${CXX} $< ${FLAGS} -o $@



