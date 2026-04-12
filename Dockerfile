FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update && apt-get install -y \
    software-properties-common lsb-release apt-transport-https ca-certificates curl gnupg \
    build-essential git cmake python3-dev python3-pip zlib1g-dev \
    libprotobuf-dev protobuf-compiler && \
    pip3 install --no-cache-dir numpy protobuf

RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y cmake

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && rm cuda-keyring.deb && \
    apt-get update && apt-get -y install cuda-toolkit-12-8

RUN curl -fL https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.9.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8_1.0-1_amd64.deb \
    -o trt.deb
RUN dpkg -i trt.deb && rm trt.deb && \
    cp /var/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8/*.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y tensorrt-dev tensorrt

WORKDIR /opt

RUN apt-get -y install apt-utils cudnn9-cuda-12
RUN pip install --no-cache-dir psutil

RUN git clone --recursive https://github.com/microsoft/onnxruntime.git && \
    cd onnxruntime && \
    git checkout 986b66af96252488bcf885741623ba877964baca && \
    git submodule update --init --recursive

RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/lib
ENV CUDAARCHS=70;75;80;86;89;90a;100a;120a

RUN cd onnxruntime && ./build.sh --allow_running_as_root --config Release --build_shared_lib --parallel \
    --use_cuda --cuda_home /usr/local/cuda-12.8 --cudnn_home /usr/lib/x86_64-linux-gnu \
    --use_tensorrt --tensorrt_home /usr \
    --skip_tests \
    --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" ONNX_USE_LTO=OFF && \
    mkdir -p /usr/local/lib /usr/local/include/onnxruntime && \
    cp build/Linux/Release/libonnxruntime.so* /usr/local/lib/ && \
    cp -r include/* /usr/local/include/onnxruntime/ && \
    ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev curl git ca-certificates vim && \
    rm -rf /var/lib/apt/lists/*
RUN pkg-config --cflags --libs openssl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /workspace
COPY asr-onnx/ ./asr-onnx
COPY messages/  ./messages

RUN cd asr-onnx && cargo build --release

COPY asr-onnx/test_entrypoint.sh /test_entrypoint.sh
RUN chmod +x /test_entrypoint.sh

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-12.8/lib64

WORKDIR /workspace
CMD ["bash"]

