#!/bin/bash
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate materialseg3d-env

export USE_CUDA=1

exec "$@"