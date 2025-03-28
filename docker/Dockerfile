# Use the NVIDIA CUDA base image with CUDA 11.6 and Ubuntu 20.04
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt clean
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:$PATH

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Update PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Update Conda
RUN conda update -n base -c defaults conda -y

# Create a new Conda environment
RUN conda create -n materialseg3d-env python=3.9.15 -y

# Activate the environment in the PATH
ENV PATH=$CONDA_DIR/envs/materialseg3d-env/bin:$CONDA_DIR/bin:$PATH

# Remove libstdc++.so.6 from the Conda environment
RUN rm -f $CONDA_DIR/envs/materialseg3d-env/lib/libstdc++.so.6

# Clone the repository directly into /app
RUN git clone https://github.com/vantemoon/MaterialSeg3D.git

# Install Tex2Tex dependencies
# RUN conda install pytorch==1.13.1 torchvision==0.14.0 torchaudio==0.13.0 cudatoolkit=11.6 -c pytorch -c nvidia -y
RUN pip install torch==1.13.0+cu116 torchaudio==0.13.0+cu116 torchvision==0.14.0+cu116 --index-url https://download.pytorch.org/whl/cu116
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub -y
RUN conda install -c iopath iopath -y
RUN pip install --no-cache-dir numpy scipy matplotlib fvcore iopath -f https://download.pytorch.org/whl/cu116
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
RUN pip install xformers==0.0.16

RUN pip install -r /app/MaterialSeg3D/Text2Tex/requirements.txt
WORKDIR /app/MaterialSeg3D

# Download the pre-trained depth model from Hugging Face
RUN mkdir -p /app/MaterialSeg3D/Text2Tex/models/ControlNet/models && \
    wget --content-disposition "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth" \
    -O /app/MaterialSeg3D/Text2Tex/models/ControlNet/models/control_sd15_depth.pth

# Install GET3D dependencies
RUN pip install ninja xatlas gdown
RUN pip install git+https://github.com/NVlabs/nvdiffrast/
RUN pip install meshzoo ipdb imageio gputil h5py point-cloud-utils imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0
RUN pip install urllib3 scipy click tqdm opencv-python==4.5.4.58

# Build Blender environment
RUN wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender2.90/blender-2.90.0-linux64.tar.xz -O /tmp/blender.tar.xz
RUN tar -xf /tmp/blender.tar.xz -C /opt

# Add Blender to PATH
RUN ln -s /opt/blender-2.90.0-linux64/blender /usr/bin/blender
RUN cd /opt/blender-2.90.0-linux64/2.90/python/bin
RUN /opt/blender-2.90.0-linux64/2.90/python/bin/python3.7m -m ensurepip
RUN /opt/blender-2.90.0-linux64/2.90/python/bin/python3.7m -m pip install numpy
RUN rm -f /usr/bin/blender && ln -s /opt/blender-2.90.0-linux64/blender /usr/bin/blender

# Install required Blender dependencies
RUN apt update && apt install -y \
    libxi6 \
    libxrender1 \
    libxfixes3 \
    libxcursor1 \
    libxxf86vm1 \
    libgl1-mesa-glx \
    libdbus-1-3

# Install MMSegmentation
WORKDIR /app
RUN pip install -U openmim
RUN mim install mmengine
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

RUN cd /app/MaterialSeg3D/mmsegmentation && pip install -v -e .

# Download the model weights from Google Drive and unzip the file
RUN pip install gdown
RUN gdown 'https://drive.google.com/uc?id=1R323ODbLvg1jjJ6cXoE364ZGyzFeZ_w0' -O weights.zip
RUN unzip weights.zip -d /app/MaterialSeg3D/mmsegmentation/temp_dir
RUN rm weights.zip
RUN mv /app/MaterialSeg3D/mmsegmentation/temp_dir/* /app/MaterialSeg3D/mmsegmentation/work_dir
RUN rmdir /app/MaterialSeg3D/mmsegmentation/temp_dir

# Install required packages for frone-end
RUN pip install gunicorn
RUN pip install flask
RUN pip install flask-cors
RUN pip install pillow

# Install specific versions of dependencies
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir setuptools==60.2.0 && \
    pip install --no-cache-dir \
        gradio==3.37.0 \
        huggingface-hub==0.23.0 \
        accelerate==0.31.0 \
        diffusers==0.7.2 \
        transformers==4.20.0 \
        numpy==1.22.4 \
        torchmetrics==0.11.4 \
        opencv-python==4.5.5.64 && \
    pip install --no-cache-dir \
        torch==1.13.0+cu116 torchaudio==0.13.0+cu116 torchvision==0.14.0+cu116 \
        --index-url https://download.pytorch.org/whl/cu116 && \
    pip install --no-cache-dir \
        pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE 8080
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]