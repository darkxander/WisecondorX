#!/bin/bash

# Vérifie qu'un seul argument est fourni (le dossier)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

input_dir=$1
input_pattern="$input_dir/*_aberrations.bed"
output_file="merged_aberration_1kb.tsv"

# En-tête avec ordre souhaité
echo -e "#chr\tstart\tend\ttype\tSAMPLE\tratio\tzscore" > "$output_file"

# Boucle sur les fichiers .bed du dossier
for file in $input_pattern; do
    sample=$(basename "$file" _aberrations.bed)
    awk -v sample="$sample" 'FNR > 1 {
        type = $6;
        gsub(/gain/, "dup", type);
        gsub(/loss/, "del", type);
        print $1 "\t" $2 "\t" $3 "\t" type "\t" sample "\t" $4 "\t" $5
    }' "$file" >> "$output_file"
done
