# Glenside Dockerfile
#
# Uses `nproc+1` cores when running make.

FROM ubuntu:18.04

# Install needed packages
# Needed so that tzdata install will be non-interactive
# https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y git libgtest-dev cmake wget unzip libtinfo-dev libz-dev libcurl4-openssl-dev libopenblas-dev g++ sudo python3-dev libclang-dev curl lsb-release wget software-properties-common python3-pip

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN rustup default nightly

# Install LLVM
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN sudo ./llvm.sh 10

# Needed by TVM Rust build process
ENV LLVM_CONFIG_PATH=/usr/lib/llvm-10/bin/llvm-config

# Build TVM with Rust bindings
# TODO(@gussmith23) Switch this to TVM mainline
# once https://github.com/apache/incubator-tvm/pull/6563 is merged
RUN cd /root && git clone https://github.com/mwillsey/incubator-tvm tvm --recursive
WORKDIR /root/tvm
RUN git fetch
RUN git checkout 3b6edf9ec0b6b3ab6a91174e7e2aa321cd8ec9b2
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
RUN bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j`nproc`"

# Help the system find the libtvm library and TVM Python library
ENV TVM_HOME=/root/tvm
ENV PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="$TVM_HOME/build/"

# Set up Python
RUN pip3 install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Build Glenside with all features
WORKDIR /root/glenside
COPY . .

# At this point, you should be able to build Glenside with whatever features you
# want!
