# Use the NVIDIA CUDA base image with CUDA 11.3 and Ubuntu 20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND noninteractive
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
RUN conda create -n environment_name python=3.9.15 -y

# Activate the environment in the PATH
ENV PATH=$CONDA_DIR/envs/materialseg3d-env/bin:$CONDA_DIR/bin:$PATH

# Remove libstdc++.so.6 from the Conda environment
RUN rm -f $CONDA_DIR/envs/materialseg3d-env/lib/libstdc++.so.6

# Clone the repository directly into /app
RUN git clone https://github.com/vantemoon/MaterialSeg3D.git

# Install Tex2Tex dependencies
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -y
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub nvidiacub -y
RUN conda install pytorch3d -c pytorch3d -y
RUN conda install xformers -c xformers -y
RUN cd /app/MaterialSeg3D/Text2Tex
RUN pip install -r requirements.txt
RUN cd /app/MaterialSeg3D

# Download the pre-trained depth file
RUN wget https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_depth.pth -O /app/Tex2Tex/models/ControlNet/models/control_sd15_depth.pth

# Install GET3D dependencies
RUN pip install ninja xatlas gdown
RUN pip install git+https://github.com/NVlabs/nvdiffrast/
RUN pip install meshzoo ipdb imageio gputil h5py point-cloud-utils imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0
RUN pip install urllib3
RUN pip install scipy
RUN pip install click
RUN pip install tqdm
RUN pip install opencv-python==4.5.4.58

# Build Blender environment
RUN wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender2.90/blender-2.90.0-linux64.tar.xz -O /tmp/blender.tar.xz
RUN tar -xf /tmp/blender.tar.xz -C /opt
RUN cd /opt/blender-2.90.0-linux64/2.90/python/bin
RUN ./python3.7m -m ensurepip
RUN ./python3.7m -m pip install numpy

# Install MMSegmentation
RUN cd /app
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN cd mmsegmentation
RUN pip install -v -e .

# Download the model weights from Google Drive and unzip the file
RUN pip install gdown
RUN gdown 'https://drive.google.com/uc?id=1R323ODbLvg1jjJ6cXoE364ZGyzFeZ_w0' -O weights.zip
RUN unzip weights.zip -d temp_dir
RUN rm weights.zip
RUN mv temp_dir/* ./work_dir
RUN rmdir temp_dir

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]