#!/usr/bin/env bash
set -o pipefail
clear
#-----------------------------------------------------------------------------
# CONFIGURATION
#-----------------------------------------------------------------------------
MY_ENV="insurance"          # Name for your reproducible environment
BASE_ENV="none"                # If "none", creates environment from scratch; otherwise, clones this environment
PACKAGES=("pip" "tqdm" "numpy" "scipy" "pandas" "r-base" "r-tweedie" "rpy2")  # Additional packages to install
CONDA_CHANNEL="conda-forge"      # Channel to use (if needed)
CONDA_BASE=$(conda info --base)
HOME_FOLDER=$(pwd)
conda config --add envs_dirs .conda/envs

REQUIREMENT_FILE="requirements.txt"
ENVIRONMENT_FILE="environment.yml"
CONFIG_FILE="config.txt"
# Flag to track the current section
current_section=""
while IFS='=' read -r key value; do
  # Trim whitespace around 'key' and 'value'
  key=$(echo "$key" | xargs)
  value=$(echo "$value" | xargs)

  # Skip empty lines
  [[ -z "$key" ]] && continue

  # Detect section headers
  if [[ "$key" =~ ^\[.*\]$ ]]; then
    current_section="${key//[\[\]]/}" # Extract section name without brackets
    continue
  fi

  # Process keys based on the current section
  case "$current_section" in
    setup)
      case "$key" in
        rebuild_env) REBUILD_ENV=$value ;;
        rebuild_final_data) REBUILD_FINAL_DATA=$value ;;
      esac
      ;;
  esac
done < "$CONFIG_FILE"

# If rebuild is true, remove existing environment.yml and the conda environment
if [[ "$REBUILD_ENV" == true ]]; then
    echo "Rebuild requested: removing ${ENVIRONMENT_FILE} and the \"${MY_ENV}\" environment if they exist."
    
    if [[ -f "$ENVIRONMENT_FILE" ]]; then
        echo "Removing existing ${ENVIRONMENT_FILE} ..."
        rm "$ENVIRONMENT_FILE"
    fi

    # Check if environment exists, then remove it
    if conda env list | grep -q "^\s*${MY_ENV}\s"; then
        echo "Removing existing conda environment \"${MY_ENV}\" ..."
        conda remove -y --name "${MY_ENV}" --all
    fi
fi

if [[ ! -f "${ENVIRONMENT_FILE}" ]]; then
    echo "No ${ENVIRONMENT_FILE} found. Creating a new conda environment \"${MY_ENV}\"..."

    # 2) Create the conda env:
    if [[ "${BASE_ENV}" == "none" ]]; then
        echo "BASE_ENV is 'none'. Creating environment from scratch."
        conda create -y --name "${MY_ENV}" python=3.12
    else
        echo "BASE_ENV is \"${BASE_ENV}\". Cloning that environment."
        conda create -y --name "${MY_ENV}" --clone "${BASE_ENV}"
    fi
    # 3) Activate and install additional packages
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${MY_ENV}"
    echo "Installing additional packages: ${PACKAGES[*]}"
    conda install -y -c "${CONDA_CHANNEL}" "${PACKAGES[@]}"
    
    if [[ -f "${REQUIREMENT_FILE}" ]]; then
        pip install -r requirements.txt
    fi
    
    # 4) Export the environment to environment.yml
    echo "Exporting the environment to ${ENVIRONMENT_FILE} ..."
    conda env export --no-builds > "${ENVIRONMENT_FILE}"
    echo "Created ${ENVIRONMENT_FILE}"
    conda deactivate
else
    if ! conda env list | grep -q "^\s*${MY_ENV}\s"; then
        echo "Conda environment \"${MY_ENV}\" does not exist. Creating from ${ENVIRONMENT_FILE} ..."
        conda env create -f "$ENVIRONMENT_FILE"
    else
        echo "Conda environment \"${MY_ENV}\" already exists. Nothing to do."
    fi
fi

if [[ "$REBUILD_FINAL_DATA" == true ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${MY_ENV}"
    python src/final_data.py
    conda deactivate
fi