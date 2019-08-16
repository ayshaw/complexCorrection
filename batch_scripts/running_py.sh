#!/usr/bin/bash
#SBATCH -c 1 # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 0-23:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=40G                          # Memory total in MB (for all cores)
#SBATCH -o concat_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e concat_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ayshaw@g.harvard.edu   # Email to which notifications will be sent
module load gcc/6.2.0 python/3.6.0
source activate evcouplings_stable
echo "finished loading environment"
which python
python preparing_complexes.py
echo "I finished this python script"

