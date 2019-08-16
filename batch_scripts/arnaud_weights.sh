#!/usr/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --mem-per-cpu=5GB
#SBATCH -p priority
#SBATCH -o ../slurm_output/hostname_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ayshaw@g.harvard.edu
module load gcc/6.2.0 cuda/9.0 python/3.6.0
source ~/jupytervenv/bin/activate
python ../python_scripts/fast_weights_arnaud.py
#python multithreading.py 4FAZA.fas 4
