#!/bin/bash
set -euo pipefail

mkdir -p Log_S1 Step1

N=$(wc -l < list_cram.txt)
echo "Submitting ${N} jobs as SLURM array..."
sbatch --array=1-${N}%200 Step1.sh
