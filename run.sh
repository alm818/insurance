#!/usr/bin/env bash
set -o pipefail

MY_ENV="insurance"          
CONDA_BASE=$(conda info --base)
conda config --add envs_dirs .conda/envs
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${MY_ENV}"
clear
# python src/final_data.py
python src/main.py
conda deactivate