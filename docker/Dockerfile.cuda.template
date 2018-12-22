FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20
RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
         libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
     rm -rf /var/lib/apt/lists/*

### From previous Docker template
# Ubuntu Packages
RUN apt-get update && apt-get update -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
    software-properties-common \
    apt-utils \
    nano \
    vim \
    man \
    build-essential \
    wget \
    sudo \
    git \
    mercurial \
    subversion && \
    rm -rf /var/lib/apt/lists/* \ 
    nvidia-profiler # --no-install-recommends	

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get update && apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev libglfw3-dev

## Miniconda end

RUN conda create -y --name pytorch-py36 python=3.6.3 numpy pyyaml scipy ipython mkl

# RUN pip install line_profiler
RUN conda install pytorch torchvision cuda80 -c pytorch
RUN conda install -y mpi4py opencv
RUN pip install gym gym[atari] pandas hashfs pydevd remote_pdb rpdb matplotlib visdom 
RUN pip install sacred GitPython pymongo tinydb tinydb-serialization tensorflow pptree progressbar2 ipdb namedlist pyyaml cython

RUN pip install -e git+https://github.com/openai/mujoco-py.git#egg=mujoco_py
RUN pip install -e git+https://github.com/openai/baselines.git#egg=baselines 
RUN pip install 'gym[atari]'

# WORKDIR /workspace
# RUN chmod -R a+w /workspace

# Section to get permissions right, and avoid running inside as root {{
    # Create a user matching the UID, and create/chmod home dir (== project directory)
    # (uid corresponds to breord in CS network)
    RUN useradd -d /project -u <<UID>> --create-home user
    USER user
    WORKDIR /project/
    ADD . /project/

ENV PYTHONPATH "$PYTHONPATH:/project/"

ENTRYPOINT ["/opt/conda/bin/python"]
