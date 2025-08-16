FROM cnly/dotfiles-full:bullseye-20230109-c20df35

# TVM deps
RUN apt update && apt install -y \
    ninja-build zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev llvm-13 libopenblas-dev

# Glow deps
RUN apt install -y graphviz libpng-dev \
    libprotobuf-dev ninja-build protobuf-compiler wget \
    opencl-headers libgoogle-glog-dev libboost-all-dev \
    libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
    libjemalloc-dev libpthread-stubs0-dev liblz4-dev libzstd-dev libbz2-dev \
    libsodium-dev libfmt-dev clang-13
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100

# NNFusion deps
RUN apt install -y build-essential cmake git curl zlib1g zlib1g-dev libtinfo-dev unzip \
    autoconf automake libtool ca-certificates gdb sqlite3 libsqlite3-dev libcurl4-openssl-dev \
    libprotobuf-dev protobuf-compiler libgflags-dev libgtest-dev \
    libhwloc-dev libgmock-dev

# Ghidra deps
RUN apt install -y openjdk-17-jdk-headless unzip

# Python
RUN apt install -y libreadline-dev
RUN /home/linuxbrew/.linuxbrew/bin/pyenv install 3.8.12
