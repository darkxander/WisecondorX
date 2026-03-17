#!/bin/bash

name=$(echo $(basename $1 | cut -d '.' -f1))

mkdir -p output_method3

module load R/4.4.0
source /pasteur/helix/projects/ghfc_wgs/tools/wisecondorx_v2_venv/bin/activate

WisecondorX predict $1 /pasteur/helix/projects/ghfc_wgs/references/GRCh38/WisecondorX/1kb/reference_wisecondorX_hg38_1kb.npz output_method3/$name --plot --bed
