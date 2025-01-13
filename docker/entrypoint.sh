#!/bin/bash
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate materialseg3d-env

exec "$@"