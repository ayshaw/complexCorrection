#!/usr/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 0-00:15                         # Runtime in D-HH:MM format
#SBATCH -p short                          # Partition to run in
#SBATCH --mem=10G                         # Memory total in MB (for all cores)
#SBATCH --mail-type=FAIL                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ayshaw@g.harvard.edu   # Email to which notifications will be sent
#SBATCH -o output_logs/gen_%j_allpdb0766_.out
#SBATCH -e output_logs/gen_%j_allpdb0766_.err
module load gcc/6.2.0 python/3.6.0
source activate evcouplings_stable
python /home/as974/ada/multimerCorrection/general_complex_sysargv.py allpdb0766