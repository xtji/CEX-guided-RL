FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="Europe/London"
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    liblapack-dev \
    libblas-dev \
    libsuitesparse-dev \
    ca-certificates \
    p7zip-full \
    zip \
    libglpk-dev \
    wget \
    git \
    nano \
    gpg \
    software-properties-common \
    g++ \
    m4 \
    xz-utils \
    libgmp-dev \
    unzip \
    zlib1g-dev \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libboost-regex-dev \
    libboost-iostreams-dev \
    libtbb-dev libreadline-dev \
    pkg-config \
    libgsl-dev \
    flex \
    bison \
    libcliquer-dev \
    gfortran \
    file \
    dpkg-dev \
    libopenblas-dev \
    rpm \
    tar \
    libmetis-dev \
    liblapack64-dev \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    --no-install-recommends

RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /opt

ADD scipoptsuite-8.0.3.tgz /optx

RUN wget https://github.com/scipopt/papilo/archive/refs/tags/v2.1.2.tar.gz && \
    tar xfz v2.1.2.tar.gz && \
    rm -f v2.1.2.tar.gz && \
    cd papilo-2.1.2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j20 && \
    make install

RUN wget https://github.com/coin-or/Ipopt/archive/refs/tags/releases/3.14.12.tar.gz && \
    tar xfz 3.14.12.tar.gz && \
    rm -f 3.14.12.tar.gz && \
    cd Ipopt-releases-3.14.12 && \
    mkdir build && \
    cd build && \
    ../configure && \
    make -j20 && \
    make install

RUN cd scipoptsuite-8.0.3 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j20 && \
    make install

# # Set the working directory in the container to /app
WORKDIR /app
ADD . /app

ENV CVXOPT_BUILD_GLPK=1 \
    CVXOPT_GLPK_LIB_DIR=/usr/lib/x86_64-linux-gnu/ \
    CVXOPT_GLPK_INC_DIR=/usr/include/

# # Install any needed packages specified in requirements.txt
RUN pip install wheel && \
    pip install --no-cache-dir --no-dependencies -r requirements.txt && \
    pip install cvxpy[SCIP] && \
    pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    git submodule update --init --recursive && \
    pip install -e gym-gridworld/ && \
    pip install -e gym-marsrover/

ENTRYPOINT [ "/bin/bash" ]
