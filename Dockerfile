# Glenside Dockerfile
#
# the build_arg tvm_build_threads controls how many threads are used to build TVM.

FROM ubuntu:18.04

# Install needed packages
# Needed so that tzdata install will be non-interactive
# https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y git libgtest-dev libssl-dev cmake wget unzip libtinfo-dev libz-dev libcurl4-openssl-dev libopenblas-dev g++ sudo python3-dev libclang-dev curl lsb-release wget software-properties-common pkg-config python3-pip

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
# PR #125 enables us to use stable (by removing usage of Bencher)
# RUN rustup default nightly

# Install LLVM
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN sudo ./llvm.sh 10

# Needed by TVM Rust build process
ENV LLVM_CONFIG_PATH=/usr/lib/llvm-10/bin/llvm-config

# Build TVM with Rust bindings
# TODO(@gussmith23) Switch to TVM mainline once
# https://github.com/apache/tvm/pull/10063 lands.
RUN cd /root && git clone https://github.com/gussmith23/tvm tvm --recursive
WORKDIR /root/tvm
RUN git fetch
RUN git checkout f3f14a68b4adedfc48ad448d155ce8f720010317
RUN git submodule sync && git submodule update
RUN echo 'set(USE_LLVM $ENV{LLVM_CONFIG_PATH})' >> config.cmake
RUN echo 'set(USE_RPC ON)' >> config.cmake
RUN echo 'set(USE_SORT ON)' >> config.cmake
RUN echo 'set(USE_GRAPH_RUNTIME ON)' >> config.cmake
RUN echo 'set(USE_BLAS openblas)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD 14)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD_REQUIRED ON)' >> config.cmake
RUN echo 'set(CMAKE_CXX_EXTENSIONS OFF)' >> config.cmake
#RUN echo 'set(CMAKE_BUILD_TYPE Debug)' >> config.cmake
ARG tvm_build_threads=2
RUN bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j${tvm_build_threads}"

# Help the system find the libtvm library and TVM Python library
ENV TVM_HOME=/root/tvm
ENV PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="$TVM_HOME/build/"

# Set up Python
RUN pip3 install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Build Glenside.
WORKDIR /root/glenside
COPY . .

