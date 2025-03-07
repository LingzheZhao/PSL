# Texture Reference is deprecated since CUDA 12.0
# See: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE.html
#
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG MAX_JOBS 32
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=89;86
ENV TORCH_CUDA_ARCH_LIST="8.9;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_CACHE_DIR=/tmp/

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # DeterminedAI requirements and common tools
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client nfs-common \
        ## Tools
        git curl wget unzip nano vim-tiny net-tools sudo htop iotop iputils-ping \
        cloc rsync screen tmux xz-utils software-properties-common \
        ## Deps
        ffmpeg \
        libassimp-dev \
        libatlas-base-dev \
        libavdevice-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-test-dev \
        libcereal-dev \
        libcgal-dev \
        libeigen3-dev \
        libembree-dev \
        libflann-dev \
        libfreeimage-dev \
        libgflags-dev \
        libglew-dev \
        libglfw3-dev \
        libgoogle-glog-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        liblapack-dev \
        libmetis-dev \
        libopenblas-dev \
        libprotobuf-dev \
        libqt5opengl5-dev \
        libsqlite3-dev \
        libsuitesparse-dev libcusparse11 \
        protobuf-compiler \
        python-is-python3 \
        python3.10-dev \
        python3-pip \
        qtbase5-dev \
        xorg-dev \
        intel-mkl \
        libtbb2 libtbb-dev \
        libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        # VTK
        libvtk9-dev \
        # OSMesa build dependencies
        libosmesa6-dev \
        # EGL build dependencies
        libopengl-dev libglvnd-dev libgl-dev libglx-dev libegl-dev \
        # x11 utils
        mesa-utils x11-apps \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Install Determined AI and python deps
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.1 &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install -U pip setuptools pathtools promise pybind11 &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog --single-branch &&\
    cd glog &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j `nproc` &&\
    make install &&\
    rm -rf /tmp/*
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver --single-branch &&\
    cd ceres-solver &&\
    git checkout $(git describe --tags) &&\
    mkdir build &&\
    cd build &&\
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF &&\
    make -j `nproc` &&\
    make install &&\
    rm -rf /tmp/*

ARG OPENEXR_VERSION=3.2.1
RUN wget https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v${OPENEXR_VERSION}.tar.gz &&\
    tar -xvzf v${OPENEXR_VERSION}.tar.gz &&\
    cd openexr-${OPENEXR_VERSION} &&\
    cmake -B build -GNinja &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# Install OpenCV
ARG OPENCV_VERSION="4.8.1"
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    cd opencv-${OPENCV_VERSION} &&\
    cmake \
        -B /tmp/opencv-${OPENCV_VERSION}/build \
        -GNinja \
        -DCMAKE_CXX_STANDARD=17 \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=${TORCH_CUDA_ARCH_LIST} \
        -DHAVE_FFMPEG=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# Install colmap.
RUN git clone https://github.com/colmap/colmap &&\
    cd colmap &&\
    # Fix CUDA 12.x compile for tag/3.8
    git checkout 1f80118456f4b587a44f288ce5874099fbfebc36 &&\
    mkdir build &&\
    cd build &&\
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} &&\
    make -j `nproc` &&\
    make install &&\
    rm -rf /tmp/*

# Install pycolmap, required by hloc.
RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap &&\
    cd pycolmap &&\
    pip install . &&\
    rm -rf /tmp/*

# Install pyceres from source
RUN git clone --branch v1.0 --recursive https://github.com/cvg/pyceres &&\
    cd pyceres &&\
    pip install -e . &&\
    rm -rf /tmp/*

RUN git clone --recursive https://github.com/pytorch/pytorch &&\
    cd pytorch &&\
    git checkout v2.1.1 &&\
    git submodule sync &&\
    git submodule update --init --recursive --jobs 0 &&\
    _GLIBCXX_USE_CXX11_ABI=1 TORCH_USE_CUDA_DSA=1 \
    USE_NUMPY=1 USE_CUDNN=1 USE_OPENCV=1 USE_BLAS=1 USE_LAPACK=1 USE_MKL=1 \
        python setup.py install &&\
    rm -rf /tmp/*

RUN pip install sympy &&\
    git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision --single-branch &&\
    cd torchvision &&\
    python setup.py install &&\
    rm -rf /tmp/*

RUN pip install git+https://github.com/cvg/Hierarchical-Localization@master &&\
    pip install git+https://github.com/cvg/pixel-perfect-sfm@main &&\
    pip install git+https://github.com/NVlabs/tiny-cuda-nn@master#subdirectory=bindings/torch &&\
    pip install open3d>=0.16.0 --ignore-installed &&\
    rm -rf /tmp/*

ARG FORCE_CUDA="1"
ARG CPATH="/usr/local/include:/usr/local/cuda/include:$CPATH"
RUN pip install torch-scatter &&\
    rm -rf /tmp/*

ADD . /opt/PSL
RUN cd /opt/PSL &&\
    cmake -B /tmp/build -GNinja &&\
    cmake --build /tmp/build -t install &&\
    cd python &&\
    pip install -e . &&\
    rm -rf /tmp/* \
