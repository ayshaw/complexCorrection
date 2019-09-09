import numpy as np
import sys
#import tensorflow as tf
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import pandas as pd
import os
import time
import pickle as pkl
from scipy.spatial import distance
from functools import partial
import multiprocessing as mp
import cProfile

offset = 0
alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
	a2n[a] = n
################
print(sys.argv)
def aa2num(aa):
	'''convert aa into num'''
	if aa in a2n: return a2n[aa]
	else: return a2n['-']
def parse_fasta(filename,limit=-1):
	'''function to parse fasta'''
	header = []
	sequence = []
	lines = open(filename, "r")
	for line in lines:
		line = line.rstrip()
		if line[0] == ">":
			if len(header) == limit:
				break
			header.append(line[1:])
			sequence.append([])
		else:
			sequence[-1].append(line)
	lines.close()
	sequence = [''.join(seq) for seq in sequence]
	return np.array(header), np.array(sequence)
def filt_gaps(msa,gap_cutoff=0.5):
	'''filters alignment to remove gappy positions'''
	tmp = (msa == states-1).astype(np.float)
	non_gaps = np.where(np.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
	return msa[:,non_gaps],non_gaps

def weights_serial(msa,start_ind,end_ind):
	rncol=(1/(msa.shape[1]))
	#output=[]
	weights_file = open('fastweights_'+str(start_ind)+'-'+str(end_ind)+'.txt','w+')
	for i in range(int(start_ind),int(end_ind)):
		weights_file.write(str(i)+'\t'+str(1/(np.sum(rncol*np.sum(msa==msa[i],axis=1,dtype=np.uint64)>=0.8,dtype=np.uint16)))+'\n')
	#if queue!=None:
	weights_file.close()
	#	queue.put(output)

def get_eff_lowmem(msa):
	msa = np.uint8(msa)
	nrow=np.uint32(msa.shape[0])
	#pool=mp.Pool(processes=mp.cpu_count())
	#mp.set_start_method('fork')
	#queue=mp.Queue()
	num_cpus = int(sys.argv[offset+2])
	print('num cpus:',num_cpus)
	start = np.arange(0,nrow-int(nrow/num_cpus)+1,int(nrow/num_cpus),dtype=np.uint32)[::-1]
	print('num_jobs: ',len(start))
	end = (start + int(nrow/num_cpus))
	print(end)
	end[0] = nrow
	processes = []
	
	for start_ind,end_ind in zip(start,end):
		p=mp.Process(target=weights_serial,args=(msa,start_ind,end_ind))
		print('processes initialized!')
		processes.append(p)
		p.start()
	#for p in processes:
		#print(queue.get())
	print('output appended!')
	for p in processes:
		p.join()

def get_eff(msa,eff_cutoff=0.8):
	'''compute effective weight for each sequence'''
	ncol = msa.shape[1]
	start = time.time()
	print('starting pdist!')
	# pairwise identity
	pdist_res = pdist(msa,"hamming") #need to sparsify this process
	#print('shape of pdist result before squareform: {}, sum of pdist: {}'.format(pdist(msa,'hamming').shape, np.sum(pdist(msa,'hamming'))))
	msa_sm = 1.0 - squareform(pdist(msa,"hamming")) #need to sparsify this process
	#print('finished hamming: {} seconds \t shape of msa_sm after squareform: {}'.format(time.time()-start,msa_sm.shape))
	# weight for each sequence
	msa_w = (msa_sm >= eff_cutoff).astype(np.float64)
	print(msa_w)
	#print('shape of msa_w after cutoff: {}, \t sum of msa_w:{} \t shape of sum: {}'.format(msa_w.shape,np.sum(msa_w,-1),np.sum(msa_w,-1).shape))
	msa_w = 1/np.sum(msa_w,-1)
	#print('shape of weights after sum normalization: {}'.format(msa_w.shape))
	return msa_w
def mk_msa(seqs):
	'''converts list of sequences to msa'''
	msa_ori = []
	for seq in seqs:
		msa_ori.append(list(map(aa2num,seq)))
	msa_ori = np.array(msa_ori,dtype=np.int)
	start=time.time()
	# remove positions with more than > 50% gaps
	msa, v_idx = filt_gaps(msa_ori,0.5)
	print('removed gaps!\t filt_gaps: {} seconds'.format(time.time()-start))
	# compute effective weight for each sequence

	start = time.time()
	msa_weights_fast = get_eff_lowmem(msa)
	print('fast pdist implementation: {} seconds'.format(time.time()-start))
	start = time.time()
	msa_weights = get_eff(msa,0.8)
	print('slow pdist implementation: {} seconds'.format(time.time()-start))
	# compute effective number of sequences
	#print('difference between pdist:{}'.format(sum(msa_weights_fast-msa_weights)))
#
## process input sequences
names, seqs = parse_fasta(sys.argv[offset+1])
msa = mk_msa(seqs)
