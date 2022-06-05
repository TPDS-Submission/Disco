# Disco
## Abstract 
Disco is an automatic deep learning compilation module for data-parallel distributed training.
The GNN in Disco is implemented using JAX and Tensorflow1.14 which is modified to support customized fusion strategies. 
The detailed software requirements are introduced in following sections. We will also introduce the dependencies needed and the procedures to install Disco.

## Software dependency
Dependency | Version 
--- | --- 
OS  | Ubuntu-16.04   
Linux Kernel | Linux 4.4.0-131-generic x86_64 
GCC | gcc 5.4.0
CUDA-Toolkit |  cuda-10.0
CUDNN | cudnn-7.6.0
NCCL |  nccl-2.6.4 
Python |  python3.7
TensorFlow |  1.14
JAX |  Modified version based on 0.2.3


The software dependency is listed in the table above. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 
CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 
The modified Tensorflow should be downloaded from our this repository.

## Installation
We will detailed introduced the installation steps of disco in this part.
### Install python environment. 
We recommand to use anaconda, please download the installation script to install anaconda through the link: https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh
After the installation, create an environment named disco with python3.7

`conda create -n disco python=3.7 scipy=1.7.3 flask `

Then activate the environment:

```
conda activate disco
pip3 install t5==0.9.0
```

### Install CUDA CUDNN and NCCL. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 

CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. 

NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 

### Build Disco from source
First, clone the Disco project and the submodule of it from repo:

`git clone https://github.com/yxd886/Disco.git --recursive`

Then step into jax folder:

`cd jax`
]
Then please check the dependency of python3, openmpi and curl in file WORKSPACE, and modify the path to these library accordingly based on your own environment:

```
new_local_repository(
name = "python_linux",
path = "/home/net/.conda/envs/jax",
build_file_content = """
cc_library(
name = "python37-lib",
srcs = ["lib/libpython3.7m.so"],
hdrs = glob(["include/python3.7m/*.h"]),
includes = ["include/python3.7m"],
visibility = ["//visibility:public"],
)
"""
)


new_local_repository(
name = "openmpi",
path = "/usr/local/openmpi",
build_file_content = """
cc_library(
name = "openmpi-lib",
srcs = ["lib/libmpi.so","lib/libmpi_mpifh.so"],
hdrs = glob(["include/*.h"]),
includes = ["include"],
visibility = ["//visibility:public"],
)
"""
)

new_local_repository(
name = "opencurl",
path = "/usr/",
build_file_content = """
cc_library(
name = "curl-lib",
srcs = ["lib/x86_64-linux-gnu/libcurl.so"],
hdrs = glob(["include/x86_64-linux-gnu/curl/*.h"]),
includes = ["include/x86_64-linux-gnu/curl"],
visibility = ["//visibility:public"],
)
"""
)
```
Then build JAX：

`sh build.sh`

### Install trax. 
```
cd ../trax
```
Execute the following cell (once) before running any of the code samples.
```
import os
import numpy as np

!pip install -q -U trax
import trax
```
