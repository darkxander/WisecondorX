#!/bin/bash
#SBATCH -p ghfc
#SBATCH --qos=ghfc
#SBATCH --exclude=maestro-1128,maestro-1096
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH -J WisecondorX_S1
#SBATCH -o Log_S1/%a.out
#SBATCH -e Log_S1/%a.err

set -euo pipefail

module load R/4.4.0 WisecondorX/1.2.9

# Get the CRAM path for this array task
CRAM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" list_cram.txt)

# Extract sample name (e.g., B004QLO from B004QLO.GRCh38_GIABv3.cram)
name=$(basename "$CRAM" | cut -d'.' -f1)

echo "Processing sample: ${name} (${CRAM})"

WisecondorX convert "$CRAM" "Step1/${name}.npz" \
  --reference /pasteur/helix/projects/ghfc_wgs/references/GRCh38/chr/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta \
  --binsize 1000
