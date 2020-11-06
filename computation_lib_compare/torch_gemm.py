import time

import torch

def main():
    m = 4096
    n = 4096
    k = 4096

    a = torch.ones([m, k], dtype=torch.float32, device='cuda')
    a.requires_grad = False
    #b = torch.ones([k, n], dtype=torch.float32, device='gpu')

    linear = torch.nn.Linear(k, n, bias=False)
    linear.requires_grad = False
    linear.cuda()
    check = time.time()
    with torch.no_grad():
        c_ = linear(a)
        print("Linear cost: {}".format(time.time() - check))
        import pdb;pdb.set_trace()

    #torch.set_num_interop_threads(1)
    #torch.set_num_threads(1)

if __name__ == '__main__':
    main()