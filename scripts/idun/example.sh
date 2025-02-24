#!/bin/sh
#SBATCH --account="share-ie-idi"
#SBATCH --partition=CPUQ
#SBATCH --time=0-100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --job-name="multi-agent-rl"
#SBATCH --output=srun.out
#SBATCH --error=srun.err

if [ ! -f scripts/idun/clean.sh ]; then
    echo "No clean script found"
else
    echo "Cleaning up"
    sh scripts/idun/clean.sh
fi

if [ -z "$0" ]; then
    echo "Error: run with additional options."
    exit 1
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}" || exit 1
echo "Running from this directory: $(pwd)"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The jo was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module list

pip install --upgrade pip

pip install poetry
poetry install

poetry run wandb login

poetry run python -O examples

uname -a
