#!/usr/bin/env python
# coding: utf-8

# In[21]:


lines = '''#!/usr/bin/bash
#SBATCH -c 1 # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
#SBATCH -t 05-00:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=40G                          # Memory total in MB (for all cores)
#SBATCH -o output_logs/gen_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e output_logs/gen_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ayshaw@g.harvard.edu   # Email to which notifications will be sent'''


# In[22]:


import pandas as pd
import os


# In[ ]:





# In[24]:


dir_names = pd.read_csv('high_precision_complexes.csv',header=None)[1].tolist()
for dn in dir_names:
    with open('batch_scripts/'+dn+'.sh','w+') as file:
        file.writelines(lines)
        file.write('\n')
        file.write('module load gcc/6.2.0 python/3.6.0\nsource activate evcouplings_stable\n')
        file.write('python /home/as974/ada/multimerCorrection/general_complex_sysargv.py '+dn)


# In[ ]:




