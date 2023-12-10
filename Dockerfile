# Texture Reference is deprecated since CUDA 12.0
# See: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE.html
#
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG MAX_JOBS 32
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=86;89
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Common
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client nfs-common \
        ## Tools
        git curl wget unzip nano vim-tiny net-tools sudo htop iotop \
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
        libopengl-dev \
        libglvnd-dev \
        libgl-dev \
        libglx-dev \
        libegl-dev \
        # x11 utils
        mesa-utils \
        x11-apps \
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

# # VTK
# ARG VTK_VERSION=v9.2.6
# RUN git clone https://gitlab.kitware.com/vtk/vtk.git --depth 1 --branch "v${VTK_VERSION}" vtk &&\
#     cd vtk &&\
#     cmake -B build -GNinja \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DVTK_BUILD_TESTING=OFF \
#         -DVTK_BUILD_DOCUMENTATION=OFF \
#         -DVTK_BUILD_EXAMPLES=OFF \
#         -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
#         -DVTK_WRAP_PYTHON=ON \
#         -DVTK_OPENGL_HAS_EGL:BOOL=ON \
#         -DVTK_OPENGL_HAS_OSMESA:BOOL=OFF \
#         -DVTK_USE_COCOA:BOOL=OFF \
#         -DVTK_USE_CUDA=ON \
#         -DVTK_USE_X:BOOL=OFF \
#         -DVTK_RENDERING_BACKEND="OpenGL2" \
#         -DPython3_EXECUTABLE=/usr/bin/python3 \
#         -DVTK_WHEEL_BUILD=ON \
#         -DVTK_PYTHON_VERSION=3 \
#         -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS:BOOL=ON \
#         -DVTK_DIST_NAME_SUFFIX="" \
#         -DVTK_VERSION_SUFFIX="post0+egl" &&\
#     cmake --build build -t install &&\
#     cd build &&\
#     pip install . &&\
#     ldconfig && rm -rf /tmp/*

# OpenEXR
ARG OPENEXR_VERSION=3.2.1
RUN wget https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v${OPENEXR_VERSION}.tar.gz &&\
    tar -xvzf v${OPENEXR_VERSION}.tar.gz &&\
    cd openexr-${OPENEXR_VERSION} &&\
    cmake -B build -GNinja &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# OpenCV
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
