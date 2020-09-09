#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <ctime>
#include <cstring>
#include <iomanip>

#include "dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

int main(int argc, char *argv[]) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  int batch = 1;
  int in_channels = 512;
  int width = 22;
  int out_channels = 512;
  int kernel_size = 3;
  int padding = int((kernel_size - 1) / 2);

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  memory::dims conv_src_tz = {batch, in_channels, width};
  memory::dims conv_weights_tz = {out_channels, in_channels, kernel_size};
  memory::dims conv_dst_tz = {batch, out_channels, width};
  memory::dims conv_strides = {1};
  memory::dims conv_padding = {padding};

  auto zero_desc = memory::desc();

  float* user_src = new float[batch * in_channels * width];
  float* user_weights = new float[kernel_size * in_channels * out_channels];

  auto conv_src_memory = memory({conv_src_tz, dt::f32, tag::ncw}, eng);
  write_to_dnnl_memory(user_src, conv_src_memory);

  auto conv_weights_memory = memory({{conv_weights_tz }, dt::f32, tag::oiw}, eng);
  write_to_dnnl_memory(user_weights, conv_weights_memory);

  auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
  auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
  auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

  auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            zero_desc, conv_dst_md, conv_strides, conv_padding,
            conv_padding);

  auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

  auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

  std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, conv_src_memory},
                                         {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                         {DNNL_ARG_DST, conv_dst_memory}};

  auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
                       .count();

  auto conv = convolution_forward(conv_prim_desc);
  conv.execute(s, args);

  auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
                     .count();
  std::cout << "Use time: " << (end - begin) << " ms." << std::endl;

}