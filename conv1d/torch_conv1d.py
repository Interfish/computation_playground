import time

import torch

def main():
    in_channels = 512
    out_channels = 512
    kernel_size = 3
    width = 22
    stride = 1
    dilation = 1
    padding = int(dilation * (kernel_size - 1) / 2)
    bias = False

    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    ones = torch.ones([1, in_channels, width], dtype=torch.float32, device='cpu')

    import pdb;pdb.set_trace()
    conv = torch.nn.Conv1d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation,
                           bias=bias)
    conv.weight.data.fill_(1.0)
    conv.cpu()

    check = time.time()
    result = conv(ones)
    print("Conv1d cost: {}".format(time.time() - check))

if __name__ == '__main__':
    main()