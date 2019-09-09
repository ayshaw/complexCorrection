import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import pandas as pd
import os
import itertools
import glob
print('starting python script')
# from fasta
def parse_fasta(filename):
	'''function to parse fasta'''
	header = []
	sequence = []
	lines = open(filename, "r")
	for line in lines:
		line = line.rstrip()
		if line[0] == ">":
			header.append(line[1:])
			sequence.append([])
		else:
			sequence[-1].append(line)
	lines.close()
	sequence = [''.join(seq) for seq in sequence]
	return np.array(header), np.array(sequence)
def return_msa_parsed(align_file):
	header = []
	sequence = []
	lines = open(align_file, "r")
	for line in lines:
		line = line.rstrip()
		if line[0] == ">":
			header.append(line[1:])
			sequence.append([])
		else:
			sequence[-1].append(line)
	lines.close()
	sequence = [''.join(seq) for seq in sequence]

	msa_df = pd.DataFrame()
	msa_df['full_names']=np.array(header)
	msa_df['seqs']=np.array(sequence)
	msa_df['names']=msa_df.full_names.str.split('/',expand=True)[0]
	msa_df['region'] = msa_df.full_names.str.split('/',expand=True)[1]
	msa_df['species']=msa_df.names.str.split('_',expand=True)[1]
	return msa_df
def write_concat_a2m(msa1_df,msa2_df,filename):
	species_list = np.unique(np.concatenate((msa2_df['species'].values,msa1_df['species'].values)))
	print("no. species: ",len(species_list),'\n')
	species_list_limited = species_list[:30]
	counter = 0
	with open(filename,'w+') as f:
		for spec in species_list_limited:
			for prod in (itertools.product(msa1_df[msa1_df['species']==spec]['full_names'].values,msa2_df[msa2_df['species']==spec]['full_names'].values)):
				counter += 1
				f.write('>'+msa1_df[msa1_df['full_names']==prod[0]]['names'].values[0]+'-'+msa2_df[msa2_df['full_names']==prod[1]]['names'].values[0]+'/'+msa1_df[msa1_df['full_names']==prod[0]]['region'].values[0]+'-'+msa2_df[msa2_df['full_names']==prod[1]]['region'].values[0]+'\n')
				f.write(msa1_df[msa1_df['full_names']==prod[0]]['seqs'].values[0]+msa2_df[msa2_df['full_names']==prod[1]]['seqs'].values[0]+'\n')
	print('no. sequences: ',counter)
print('loaded all packages and functions - running now')
hpc_df = pd.read_csv('high_precision_complexes.csv',header=None)
print('read hpc_df')				
for index,row in hpc_df.iterrows():
	aligns = glob.glob('/home/as974/marks/users/kbrock/ecoli_complex/calibration/output/'+row[1].values[1]+'/align*')
	align_1 = aligns[0]+'/'+row.values[1]+'.a2m'
	align_2 = aligns[1]+'/'+row.values[1]+'.a2m'
	file_dir = 'benchmark/'+row.values[1]
	try: os.mkdir(file_dir)
	except: print('file already exists')
	msa1_df = return_msa_parsed(align_1)
	msa2_df = return_msa_parsed(align_2)
	file_name = file_dir+'/concat_'+str(counter)+'.a2m'
	write_concat_a2m(msa1_df,msa2_df,file_name)
	print('finished ',row)
	del msa1_df
	del msa2_df
	break
	
print("completely finished")
