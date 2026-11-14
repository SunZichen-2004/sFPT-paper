#!/bin/bash
#SBATCH --job-name=sfpt-paper
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250000M
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MPLBACKEND=Agg PYTHONUNBUFFERED=1
: "${CASE:?Set CASE to a name in cases.json}"
python -u run_case.py "$CASE" --protocol "${PROTOCOL:-paper}" --seed "${SEED:-0}"
