#include <assert.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "../x86/conv1d.hpp"

int main(int argc, char *argv[]) {
  int inChannels, outChannels, width, kernelSize, padding;
  width = atoi(argv[1]);
  inChannels = atoi(argv[2]);
  outChannels = atoi(argv[3]);
  kernelSize = atoi(argv[4]);
  // must be a odd number, otherwise padding is incorrect
  assert(kernelSize % 2 != 0);
  // same padding
  padding = (kernelSize - 1) / 2;

  float* kernelData = new float[inChannels * kernelSize * outChannels];
  float* input = new float[(width + 2 * padding) * inChannels];
  for(int i = 0; i < inChannels * kernelSize * outChannels; i++)
    kernelData[i] = 1.0;

  for(int i = 0; i < (width + 2 * padding) * inChannels; i++)
    input[i] = 1.0;

  auto c_start = std::chrono::high_resolution_clock::now();
  float* output = new float[outChannels * width];
  for(int i = 0; i < outChannels * width; i++) {
    output[i] = 0.0;
  }

  conv1d(input, kernelData, output, inChannels, outChannels, kernelSize, (width + 2 * padding));
  auto c_end = std::chrono::high_resolution_clock::now();
  std::cout << std::fixed << std::setprecision(2) << "conv1d time used: "
            << std::chrono::duration<double, std::milli>(c_end - c_start).count() << " ms" << std::endl;
  for(int i = 0; i < width * outChannels; i++) {
    assert(int(output[i]) == kernelSize * inChannels);
  }

}