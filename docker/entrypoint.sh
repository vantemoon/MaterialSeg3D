#!/bin/bash
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate materialseg3d-env

export USE_CUDA=1

cd MaterialSeg3D
exec gunicorn --workers=1 --bind=0.0.0.0:8080 --timeout 1800 api:app