#include <assert.h>

#include "conv1d.hpp"

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
  float* output = new float[outChannels * width];

  for(int i = 0; i < inChannels * kernelSize * outChannels; i++)
    kernelData[i] = 1.0f;

  for(int i = 0; i < (width + 2 * padding) * inChannels; i++)
    input[i] = 1.0f;

  conv1d(input, kernelData, output, inChannels, outChannels, kernelSize, (width + 2 * padding));
}