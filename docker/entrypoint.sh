#!/bin/bash
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate materialseg3d-env

export USE_CUDA=1

exec cd MaterialSeg3D
exec gunicorn --workers=4 --bind=0.0.0.0:8080 api:app