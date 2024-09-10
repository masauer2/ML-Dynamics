#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH -J STEP01_READ_TRR

module load mamba/latest
source activate KEMP_ML
python run_replicas.py >& step1.out
  
