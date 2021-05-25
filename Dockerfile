FROM nvidia/cuda:11.1-devel-ubuntu20.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y build-essential software-properties-common libssl-dev wget tar
## install cmake
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && tar -zxvf cmake-3.20.2.tar.gz \
    && cd cmake-3.20.2 && ./bootstrap && make -j && make install
ADD packages /tmp/packages
RUN dpkg -i /tmp/packages/nsys.2021.deb && bash /tmp/packages/nsight-compute-linux-2021.1.1.5.run -- -noprompt
RUN mkdir -p /root/computation_playground
WORKDIR /root/computation_playground