FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    ninja-build \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libopenblas-dev \
    graphviz \
    libpng-dev \
    libprotobuf-dev \
    protobuf-compiler \
    opencl-headers \
    libgoogle-glog-dev \
    libboost-all-dev \
    libdouble-conversion-dev \
    libevent-dev \
    libgflags-dev \
    libjemalloc-dev \
    libpthread-stubs0-dev \
    liblz4-dev \
    libzstd-dev \
    libsodium-dev \
    libfmt-dev \
    clang-13 \
    openjdk-17-jdk-headless \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up clang
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 100

# Install Python packages
RUN pip install --no-cache-dir \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    torchaudio==0.13.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    matplotlib>=3.5.0 \
    scikit-learn>=1.0.0 \
    scipy>=1.7.0 \
    dvc[ssh]>=3.18.0 \
    onnx>=1.14.0 \
    onnxruntime>=1.10.0 \
    captum>=0.6.0 \
    graphviz>=0.20.0 \
    seaborn>=0.12.0 \
    pyfunctional>=1.4.0 \
    lpips>=0.1.4 \
    ghidra-stubs>=11.1.2.1.0.4 \
    tqdm>=4.64.0 \
    requests>=2.26.0 \
    Pillow>=8.4.0 \
    opencv-python>=4.5.0 \
    networkx>=2.6.0

# Skip TVM - use ONNX instead for model conversion
# RUN pip install --no-cache-dir apache-tvm==0.10.0

# Set working directory
WORKDIR /workspace

# Create a simple environment script
RUN echo '#!/bin/bash' > /workspace/env.sh && \
    echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> /workspace/env.sh && \
    chmod +x /workspace/env.sh

# Default command
CMD ["/bin/bash"]
