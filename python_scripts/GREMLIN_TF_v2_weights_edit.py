# ------------------------------------------------------------
# "THE BEERWARE LICENSE" (Revision 42):
# <so@g.harvard.edu> and <pkk382@g.harvard.edu> wrote this code.
# As long as you retain this notice, you can do whatever you want
# with this stuff. If we meet someday, and you think this stuff
# is worth it, you can buy us a beer in return.
# --Sergey Ovchinnikov and Peter Koo
# ------------------------------------------------------------
from functools import partial
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from scipy import stats
import pandas as pd
import os
import keras
import sys
import pickle as pkl
import multiprocessing as mp
alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
	a2n[a] = n
print(sys.argv)
msa_file = sys.argv[1]

complex_name = os.path.basename(msa_file).split('.')[0]
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
	tmp = (msa == states-1).astype(np.float32)
	non_gaps = np.where(np.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
	return msa[:,non_gaps],non_gaps
  
def f(msa,i): 
  rncol=(1/(msa.shape[1]))
  return 1/(np.sum(rncol*np.sum(msa==msa[i],axis=1,dtype=np.uint16)>=0.8,dtype=np.uint32))

def get_eff_lowmem(msa,eff_cutoff=0.8):
  return np.fromiter(map(partial(f,msa),np.arange(msa.shape[0])),dtype=float)
  

def weights_serial(msa,start_ind,end_ind):
	rncol=(1/(msa.shape[1]))
	#output=[]
	weights_file = open('fastweights_'+str(start_ind)+'-'+str(end_ind)+'.txt','w+')
	for i in range(int(start_ind),int(end_ind)):
		weights_file.write(str(i)+'\t'+str(1/(np.sum(rncol*np.sum(msa==msa[i],axis=1,dtype=np.uint64)>=0.8,dtype=np.uint16)))+'\n')
	#if queue!=None:
	weights_file.close()
	#	queue.put(output)

def mk_msa(seqs):
	'''converts list of sequences to msa'''

	msa_ori = []
	for seq in seqs:
		msa_ori.append(list(map(aa2num,seq)))
	msa_ori = np.array(msa_ori,dtype=np.int)

	# remove positions with more than > 50% gaps
	msa, v_idx = filt_gaps(msa_ori,0.5)
	print('removed gaps')
	# compute effective weight for each sequence
	msa_weights = get_eff_lowmem(msa,0.8)
	print('computed phylogeny weights')
	# compute effective number of sequences
	ncol = msa.shape[1] # length of sequence
	w_idx = v_idx[np.stack(np.triu_indices(ncol,1),-1)]
#	 msa_cluster = cluster(msa)
	return {"msa":msa,
		  "weights":msa_weights,
		  "neff":np.sum(msa_weights),
		  "v_idx":v_idx,
		  "w_idx":w_idx,
		  "nrow":msa.shape[0],
		  "ncol":ncol}

def sym_w(w):
	'''symmetrize input matrix of shape (x,y,x,y)'''
	x = w.shape[0]
	w = w * np.reshape(1-np.eye(x),(x,1,x,1))
	w = w + tf.transpose(w,[2,3,0,1])
	return w

def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
	# adam optimizer
	# Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
	# with sum(g*g) instead of (g*g). Furthmore, we find that disabling the bias correction
	# (b_fix=False) speeds up convergence for our case.

	if var_list is None: var_list = tf.trainable_variables() 
	gradients = tf.gradients(loss,var_list)
	if b_fix: t = tf.Variable(0.0,"t")
	opt = []
	for n,(x,g) in enumerate(zip(var_list,gradients)):
		if g is not None:
			ini = dict(initializer=tf.zeros_initializer,trainable=False)
			mt = tf.get_variable(name+"_mt_"+str(n),shape=list(x.shape), **ini)
			vt = tf.get_variable(name+"_vt_"+str(n),shape=[], **ini)

			mt_tmp = b1*mt+(1-b1)*g
			vt_tmp = b2*vt+(1-b2)*tf.reduce_sum(tf.square(g))
			lr_tmp = lr/(tf.sqrt(vt_tmp) + 1e-8)

			if b_fix: lr_tmp = lr_tmp * tf.sqrt(1-tf.pow(b2,t))/(1-tf.pow(b1,t))

			opt.append(x.assign_add(-lr_tmp * mt_tmp))
			opt.append(vt.assign(vt_tmp))
			opt.append(mt.assign(mt_tmp))

	if b_fix: opt.append(t.assign_add(1.0))
	return(tf.group(opt))


def normalize(x):
  x = stats.boxcox(x - np.amin(x) + 1.0)[0]
  x_mean = np.mean(x)
  x_std = np.std(x)
  return((x-x_mean)/x_std)

def get_mtx(mrf):
  '''get mtx given mrf'''
  
  # l2norm of 20x20 matrices (note: we ignore gaps)
  raw = np.sqrt(np.sum(np.square(mrf["w"][:,:-1,:-1]),(1,2)))
  raw_sq = squareform(raw)

  # apc (average product correction)
  ap_sq = np.sum(raw_sq,0,keepdims=True)*np.sum(raw_sq,1,keepdims=True)/np.sum(raw_sq)
  apc = squareform(raw_sq - ap_sq, checks=False)

  mtx = {"i": mrf["w_idx"][:,0],
		 "j": mrf["w_idx"][:,1],
		 "raw": raw,
		 "apc": apc,
		 "zscore": normalize(apc)}
  return mtx

def GREMLIN_weights(msa,l2_wb=0.01 ,wb_input=None,opt_type="adam", opt_iter=100, opt_rate=1.0, batch_size=512):
  
	##############################################################
	# SETUP COMPUTE GRAPH
	##############################################################
	# kill any existing tensorflow graph
	tf.reset_default_graph()

	ncol = msa["ncol"] # length of sequence
	nrow = msa["nrow"] # number of sequences
	print("ncol: {},n nrow: {}".format(ncol,nrow))
	if wb_input==None:
		wb_input = np.ones([nrow])
	# msa (multiple sequence alignment) 
	MSA = tf.placeholder(tf.int32,shape=(None,ncol),name="msa")

	# one-hot encode msa
	OH_MSA = tf.one_hot(MSA,states)

	# msa weights
	MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")
	idx = tf.placeholder(tf.int64,shape=[batch_size], name = 'idx')

	# 1-body-term of the MRF
	V = tf.get_variable(name="V", 
					  shape=[ncol,states],
					  initializer=tf.zeros_initializer)

	# 2-body-term of the MRF
	W = tf.get_variable(name="W",
					  shape=[ncol,states,ncol,states],
					  initializer=tf.zeros_initializer)

	# weights for concatenation
	wb = tf.get_variable(name="wb",
					  shape=[nrow],
					  initializer=tf.ones_initializer
					  )
	wb=tf.math.multiply(wb,wb_input)
	# symmetrize W
	W = sym_w(W)

	def L2(x): return tf.reduce_sum(tf.square(x))
	def L1(x): return tf

	########################################
	# V + W
	########################################
	VW = V + tf.tensordot(OH_MSA,W,2)

	# hamiltonian
	H = tf.reduce_sum(tf.multiply(OH_MSA,VW),axis=(1,2))

	# local Z (parition function)
	Z = tf.reduce_sum(tf.reduce_logsumexp(VW,axis=2),axis=1)

	# Psuedo-Log-Likelihood
	PLL = H - Z
	wb = tf.nn.relu(wb)
	# Regularization
	L2_V = 0.01 * L2(V)
	L2_W = 0.01 * L2(W) * 0.5 * (ncol-1) * (states-1)
	#L2_wb = 0.01 * L2(tf.gather(wb,idx))
	L2_wb = l2_wb*L2(tf.gather(wb,idx))
	# loss function to minimize
	#loss = -tf.reduce_sum(PLL*MSA_weights*tf.gather(wb,idx))/tf.reduce_sum(MSA_weights*tf.gather(wb,idx))
	loss = -tf.reduce_sum(PLL*MSA_weights*tf.gather(wb,idx))/tf.reduce_sum(MSA_weights*tf.gather(wb,idx))-tf.minimum(tf.reduce_min(wb),0)
	loss = loss + (L2_V + L2_W + L2_wb)/msa["neff"]
	##############################################################
	# MINIMIZE LOSS FUNCTION
	##############################################################
	if opt_type == "adam":  
		opt = opt_adam(loss,"adam",lr=opt_rate)

	# generate input/feed
	def feed(feed_all=False):
		if batch_size is None or feed_all:
			return {MSA:msa["msa"], MSA_weights:msa["weights"],idx:np.arange(len(msa['weights']))}
		else:
			idx_val = np.random.randint(0,msa["nrow"],size=batch_size)
			return {MSA:msa["msa"][idx_val], MSA_weights:msa["weights"][idx_val],idx:idx_val}

	# optimize!
	with tf.Session() as sess:
		# initialize variables V and W
		sess.run(tf.global_variables_initializer())
		feed_dict = feed()
		# initialize V
		msa_cat = tf.keras.utils.to_categorical(msa["msa"],states)
		pseudo_count = 0.01 * np.log(msa["neff"])
		V_ini = np.log(np.sum(msa_cat.T * msa["weights"],-1).T + pseudo_count)
		V_ini = V_ini - np.mean(V_ini,-1,keepdims=True)
		wb_ini = sess.run(wb)
		sess.run(V.assign(V_ini))

		

		# compute loss across all data
		get_loss = lambda: round(sess.run(loss,feed()) * msa["neff"],2)
		print("starting",get_loss())

		if opt_type == "lbfgs":
			lbfgs = tf.contrib.opt.ScipyOptimizerInterface
			opt = lbfgs(loss,method="L-BFGS-B",options={'maxiter': opt_iter})
			opt.minimize(sess,feed(feed_all=True))

		if opt_type == "adam":
			for i in range(opt_iter):
				sess.run(opt,feed())  
				if (i+1) % int(opt_iter/10) == 0:
					print("iter",(i+1),get_loss())

		# save the V and W parameters of the MRF
		V_ = sess.run(V)
		W_ = sess.run(W)
		wb_ =sess.run(wb)

	# only return upper-right triangle of matrix (since it's symmetric)
	tri = np.triu_indices(ncol,1)
	W_ = W_[tri[0],:,tri[1],:]

	mrf = {"v": V_,
		 "w": W_,
		 "wb": wb_,
		 "wb_ini":wb_ini,
		  'w_idx':msa['w_idx'],
		  'v_idx':msa['v_idx']}

	return mrf

def make_couplingScores_csv(mrf,l2_wb,complex_name=complex_name):
	# adding amino acid to index
	mtx_weights["i_aa"] = np.array([alphabet[msa['msa_ori'][0][i]]+"_"+str(i+1) for i in mtx_weights["i"]])
	mtx_weights["j_aa"] = np.array([alphabet[msa['msa_ori'][0][j]]+"_"+str(j+1) for j in mtx_weights["j"]])


	pd_mtx_weights = pd.DataFrame(mtx_weights,columns=["i","j","apc","zscore","i_aa","j_aa"])

	try:
		os.mkdir('../{}'.format(complex_name))
	except:
		pass
	mtx_weights = get_mtx(mrf_weights)
	pd_mtx_weights.to_csv('../{0}/{0}_l2{1}_couplings_score.csv'.format(complex_name,l2_wb))
# parse fasta
print(sys.argv[1])
names, seqs = parse_fasta(sys.argv[1])

# process input sequences
msa = mk_msa(seqs)
print('msa processed!')

mrf_weights = GREMLIN_weights(msa,l2_wb=0.01)
make_couplingsScores_csv(mrf_weights,0.01)
mrf_weights = GREMLIN_weights(msa,l2_wb=0.02)
make_couplingsScores_csv(mrf_weights,0.02)



