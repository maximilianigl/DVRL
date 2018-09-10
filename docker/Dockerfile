### DO NOT EDIT DIRECTLY, SEE Dockerfile.template ###
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

# FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# # CUDA includes
# ENV CUDA_PATH /usr/local/cuda
# ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
# ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# ENV CUDNN_VERSION 7.0.5.15
# RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
#          build-essential \
#          cmake \
#          git \
#          curl \
#          ca-certificates \
#          libjpeg-dev \
#          libpng-dev \
#          libcudnn7=$CUDNN_VERSION-1+cuda9.1 \
#          libcudnn7-dev=$CUDNN_VERSION-1+cuda9.1 && \
#      rm -rf /var/lib/apt/lists/*



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

# Old Miniconda install
# RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh  && \
#      chmod +x ~/miniconda.sh && \
#      ~/miniconda.sh -b -p /opt/conda && \
#      rm ~/miniconda.sh && \
#      /opt/conda/bin/conda install conda-build && \
#      /opt/conda/bin/conda create -y --name pytorch-py36 python=3.6.3 numpy pyyaml scipy ipython mkl&& \
#      /opt/conda/bin/conda clean -ya
 

## Install Miniconda


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
# RUN /opt/conda/envs/pytorch-py36/bin/conda install pytorch torchvision cuda90 -c pytorch
#RUN /opt/conda/envs/pytorch-py36/bin/conda install pytorch torchvision cuda80 -c pytorch
RUN conda install pytorch torchvision cuda80 -c pytorch
RUN conda install -y mpi4py opencv
RUN pip install gym gym[atari] pandas hashfs pydevd remote_pdb rpdb matplotlib visdom 
RUN pip install sacred GitPython pymongo tinydb tinydb-serialization tensorflow==1.3.0 pptree progressbar2 ipdb namedlist pyyaml cython

RUN pip install -e git+https://github.com/openai/mujoco-py.git#egg=mujoco_py
RUN pip install -e git+https://github.com/openai/baselines.git#egg=baselines 

# WORKDIR /workspace
# RUN chmod -R a+w /workspace

# Section to get permissions right, and avoid running inside as root {{
    # Create a user matching the UID, and create/chmod home dir (== project directory)
    # (uid corresponds to breord in CS network)
    RUN useradd -d /project -u 12567 --create-home user
    USER user
    WORKDIR /project/
    ADD . /project/

ENV PYTHONPATH "$PYTHONPATH:/project/"

ENTRYPOINT ["/opt/conda/bin/python"]
