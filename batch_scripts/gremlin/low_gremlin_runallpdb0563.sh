#!/usr/bin/bash
#SBATCH -c 1 # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 0-05:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu                           # Partition to run in
#SBATCH --mem=40G                          # Memory total in MB (for all cores)
#SBATCH --gres=gpu:teslaV100:1 
#SBATCH --mail-type=FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ayshaw@g.harvard.edu   # Email to which notifications will be sent
#SBATCH -o output_logs/gen_%j_allpdb0563_.out
#SBATCH -e output_logs/gen_%j_allpdb0563_.err
module load gcc/6.2.0 python/3.6.0 cuda/9.0
source activate evcouplings_stable
source ~/jupytervenv/bin/activate
python /home/as974/ada/multimerCorrection/GREMLIN_TF_v2_weights_edit.py allpdb0563