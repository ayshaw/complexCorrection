import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import pandas as pd
import os
import itertools
import glob
import math
import time
import sys

def return_msa_parsed(align_file):
	uid = []
	sequence = []
	spec = []
	lines = open(align_file, "r")
	for line in lines:
		line = line.rstrip()
		if line[0] == ">":
			uid.append(line[1:])
			spec.append(line[1:].split('/')[0].split('_')[1])
			sequence.append([])
		else:
			sequence[-1].append(line)
	lines.close()
	sequence = [''.join(seq) for seq in sequence]
	print('parsing finished for {} sequences'.format(len(uid)))
	msa_df = pd.DataFrame()
	msa_df['species']=pd.Series(spec,dtype='category')
	msa_df['seqs']=np.arange(len(uid),dtype=np.int)
	print('pd initialized')
	return msa_df,sequence,uid

def write_concat_a2m(msa1_df,sequence_1,uid1,msa2_df,sequence_2,uid2,filename):	   
	species_list = np.intersect1d(msa2_df['species'].unique(),msa1_df['species'].unique())  
	species_list_limited = species_list[:sys.argv[2]]
	print('total number of species:{}'.format(len(species_list)))
	start = time.time()
	counter = 0
	with open(filename,'w+') as f:
		for spec in species_list_limited:
			for prod in (itertools.product(msa1_df[msa1_df['species']==spec]['seqs'].values,msa2_df[msa2_df['species']==spec]['seqs'].values)):
				counter += 1
				f.write('>'+uid1[prod[0]]+'-'+uid2[prod[1]]+'\n')
				f.write(sequence_1[prod[0]]+sequence_2[prod[1]]+'\n')
	print('time elapsed: {} seconds'.format(time.time()-start))
	print('number of sequences: ',counter)

aligns = glob.glob('/home/as974/marks/users/kbrock/ecoli_complex/calibration/output/'+sys.argv[1]+'/align*')
align_1 = aligns[0]+'/'+sys.argv[1]+'.a2m'
align_2 = aligns[1]+'/'+sys.argv[1]+'.a2m'
file_dir = '/home/as974/ada/multimerCorrection/benchmark_limited/'+sys.argv[1]
try: os.mkdir(file_dir)
except: print(sys.argv[1] + ' already exists')
msa1_df,sequence_1,uid1 = return_msa_parsed(align_1)
msa2_df,sequence_2,uid2 = return_msa_parsed(align_2)
print('finished parsing individual alignments')
filename = file_dir+'/concatenation_'+sys.argv[2]+'species_'+counter+'_seqs.a2m'
write_concat_a2m(msa1_df,sequence_1,uid1,msa2_df,sequence_2,uid2,filename)
print('finished concatenating alignments')
