#!/bin/bash

module load R/4.4.0

rm -Rf Log_S3
mkdir -p Log_S3

while read i;
do 
  myname=$(echo $(basename $i | cut -d '.' -f1))
  
sbatch -p ghfc --exclude=maestro-1128,maestro-1096 --mem=32G --qos=ghfc -e Log_S3/$myname.termlog -o Log_S3/$myname.termlog -J "Wisecondor" --wrap="bash Step3.sh $i"
done <  head.txt
