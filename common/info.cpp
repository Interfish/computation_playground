#include <iostream>

#include <cuda_runtime.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个 block 的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个 block 的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "每个 block x 维最大：" << devProp.maxThreadsDim[0] << std::endl;
        std::cout << "每个 block y 维最大：" << devProp.maxThreadsDim[1] << std::endl;
        std::cout << "每个 block z 维最大：" << devProp.maxThreadsDim[2] << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
        cudaSharedMemConfig* pConfig;
        cudaDeviceGetSharedMemConfig(pConfig);
        std::cout << "Bank Size type: " << pConfig << std::endl;
        std::cout << "======================================================" << std::endl;
    }
    return 0;
}