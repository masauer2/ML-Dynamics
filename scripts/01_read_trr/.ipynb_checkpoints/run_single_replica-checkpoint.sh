#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 0-00:20                  # wall time (D-HH:MM)
#SBATCH -J STEP01_READ_TRR

module load mamba/latest
source activate KEMP_ML
id=0
python run_single.py -id $id >& step1_$id.out
  
