import time
import numpy as np

class DataHelper:
    def __init__(self,
                 dataset,
                 theta,
                 custom_dataset=False,
                 alignment_file="",
                 focus_seq_name="",
                 calc_weights=True,
                 working_dir=".",
                 load_all_sequences=True,
                 alphabet_type="protein"):

        """
        Class to load and organize alignment data.
        This function also helps makes predictions about mutations.

        Parameters
        --------------
        dataset: preloaded dataset names
                    We have found it easiest to organize datasets in this
                    way and use the self.configure_datasets() func
        alignment_file: Name of the alignment file located in the "datasets"
                            folder. Not needed if dataset pre-entered
        focus_seq_name: Name of the sequence in the alignment
                            Defaults to the first sequence in the alignment
        calc_weights: (bool) Calculate sequence weights
                        Default True, but not necessary if just loading weights
                            and doing mutation effect prediction
        working_dir: location of "params", "logs", "embeddings", and "datasets"
                        folders
        theta: Sequence weighting hyperparameter
                Generally: Prokaryotic and eukaryotic families =  0.2
                            Viruses = 0.01
        load_all_sequences:
        alphabet_type: Alphabet type of associated dataset.
                            Options are DNA, RNA, protein, allelic

        Returns
        ------------
        None
        """

        # np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type

        if theta == 0:
            self.calc_weights = False

        # Initalize the elbo of the wt to None
        #   will be useful if eventually doing mutation effect prediction
        self.wt_elbo = None

        # Alignment processing parameters
        self.theta = theta

        # If I am running tests with the model, I don't need all the
        #    sequences loaded
        self.load_all_sequences = load_all_sequences

        # Load necessary information for preloaded datasets

        if custom_dataset:
            self.alignment_file = dataset

        elif self.dataset != "":
            self.configure_datasets()

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == "protein":
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
            self.reorder_alphabet = "DEKRHNQSTPGAVILMCFYW"
        elif self.alphabet_type == "RNA":
            self.alphabet = "ACGU"
            self.reorder_alphabet = "ACGU"
        elif self.alphabet_type == "DNA":
            self.alphabet = "ACGT"
            self.reorder_alphabet = "ACGT"
        elif self.alphabet_type == "allelic":
            self.alphabet = "012"
            self.reorder_alphabet = "012"

        # then generate the experimental data
        self.gen_basic_alignment()

        if self.load_all_sequences:
            self.gen_full_alignment()

    def configure_datasets(self):

        if self.dataset == "small":
            self.alignment_file = self.working_dir + "4FAZA.a2m"

        elif self.dataset == "large":
            self.alignment_file = self.working_dir + "benchmark/allpdb0777/concatenation.a2m"
        #             self.theta = 0.2

        elif self.dataset == "PABP_YEAST":
            self.alignment_file = self.working_dir + "/datasets/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m"
        #             self.theta = 0.2

        elif self.dataset == "DLG4_RAT":
            self.alignment_file = self.working_dir + "/datasets/DLG4_RAT_hmmerbit_plmc_n5_m30_f50_t0.2_r300-400_id100_b50.a2m"
        #             self.theta = 0.2

        elif self.dataset == "BG505":
            self.alignment_file = self.working_dir + "/datasets/BG505_env_1_b0.5.a2m"
        #             self.theta = 0.2

        elif self.dataset == "BF520":
            self.alignment_file = self.working_dir + "/datasets/BF520_env_1_b0.5.a2m"
        #             self.theta = 0.01

        elif self.dataset == "trna":
            self.alignment_file = self.working_dir + "/datasets/RF00005_CCU.fasta"
            self.alphabet_type = "RNA"

    #             self.theta = 0.2

    def one_hot_3D(self, s):
        """ Transform sequence string into one-hot aa vector"""
        # One-hot encode as row vector
        x = np.zeros((len(s), len(self.alphabet)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i, self.aa_dict[letter]] = 1
        return x

    def gen_basic_alignment(self):
        """ Read training alignment and store basics in class instance """
        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        # Read alignment
        self.seq_name_to_sequence = defaultdict(str)
        self.seq_names = []

        name = ""
        INPUT = open(self.alignment_file, "r")
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith(">"):
                name = line
                self.seq_names.append(name)
            else:
                self.seq_name_to_sequence[name] += line
        INPUT.close()

        # If we don"t have a focus sequence, pick the one that
        #   we used to generate the alignment
        if self.focus_seq_name == "":
            self.focus_seq_name = self.seq_names[0]

        # Select focus columns
        #  These columns are the uppercase residues of the .a2m file
        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # We also expect the focus sequence to be formatted as:
        # >[NAME]/[start]-[end]
        focus_loc = self.focus_seq_name.split("/")[-1]
        start, stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_cols_list \
            = [idx_col + int(start) for idx_col in self.focus_cols]
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col + int(start): self.focus_seq[idx_col] for idx_col in self.focus_cols}
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col + int(start): idx_col for idx_col in self.focus_cols}

    def gen_full_alignment(self):

        # Get only the focus columns
        for seq_name, sequence in self.seq_name_to_sequence.items():
            # Replace periods with dashes (the uppercase equivalent)
            sequence = sequence.replace(".", "-")

            # then get only the focus columns
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have bad characters
        alphabet_set = set(list(self.alphabet))
        seq_names_to_remove = []
        for seq_name, sequence in self.seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != "-":
                    seq_names_to_remove.append(seq_name)

        seq_names_to_remove = list(set(seq_names_to_remove))
        for seq_name in seq_names_to_remove:
            del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print("Encoding sequences")
        self.x_train = np.zeros((len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)))
        self.x_train_name_list = []
        for i, seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            self.x_train_name_list.append(seq_name)
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    self.x_train[i, j, k] = 1.0

        # Very fast weight computation

        self.seqlen = self.x_train.shape[1]
        self.datasize = self.x_train.shape[0]

        if self.calc_weights and self.theta > 0:
            print("effective weights")
            weights = []
            seq_batch = 1000
            x_train_flat = self.x_train.reshape(self.x_train.shape[0], -1)
            nb_seq = x_train_flat.shape[0]
            nb_iter = int(nb_seq / seq_batch)
            rest = nb_seq % seq_batch
            xtfs_t = x_train_flat
            for i in range(nb_iter):
                #weights.append(1.0 / (((torch.div(
                #   torch.mm(xtfs_t[i * seq_batch:(i + 1) * seq_batch], xtfs_t.transpose(0, 1)),
                #   xtfs_t[i * seq_batch:(i + 1) * seq_batch].sum(1).unsqueeze(1))) > (1 - self.theta)).sum(1).float()))
                weights.append(1.0/ (((np.div(
                    np.dot(xtfs_t[i *seq_batch: (i+1) * seq_batch], xtfs_t.transpose(0, 1)),
                    np.expand_dims(xtfs_t[i * seq_batch:(i+1) * seq_batch].sum(1),1))) > (1-self.theta)).sum(1).astype(float)))

            #weights.append(1.0 / (((torch.div(torch.mm(xtfs_t[-rest:], xtfs_t.transpose(0, 1)),
            #                                  xtfs_t[-rest:].sum(1).unsqueeze(1))) > (1 - self.theta)).sum(1).float()))
            weights.append(1.0 / (((np.div(np.dot(xtfs_t[-rest:], xtfs_t.transpose(0, 1)),
                                           np.expand_dims(xtfs_t[-rest:].sum(1))))> (1-self.theta)).sum(1).astype(float)))
            weights_tensor = np.concatenate(weights)
            self.weights = weights_tensor
            #         self.weights = weights_tensor.cpu().numpy()
            self.Neff = weights_tensor.sum()
        #             print(self.Neff)
        else:
            #             # If not using weights, use an isotropic weight matrix
            self.weights = np.ones(self.x_train.shape[0])
            self.Neff = self.x_train.shape[0]

        print("Neff =", str(self.Neff))
        print("Data Shape =", self.x_train.shape)

    #         # Fast sequence weights with Theano
    #         if self.calc_weights:
    #             print ("Computing sequence weights")
    #             # Numpy version
    #             import scipy
    #             from scipy.spatial.distance import pdist, squareform
    #             x_train_flat = self.x_train.reshape(self.x_train.shape[0], -1)
    #             print(x_train_flat.shape)
    # #             self.weights = 1.0 / np.sum(squareform(pdist(x_train_flat[:10000], metric="hamming")) < self.theta, axis=0)
    #             self.weights = 1.0 / np.sum(squareform(pdist(x_train_flat, metric="hamming")) < self.theta, axis=0)
    #             #
    #             # Theano weights
    #             # X = T.tensor3("x")
    #             # cutoff = T.scalar("theta")
    #             # X_flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    #             # N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
    #             # weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list],allow_input_downcast=True)
    #             #
    #             # self.weights = weightfun(self.x_train, self.theta)[0]

    #         else:
    #             # If not using weights, use an isotropic weight matrix
    #             self.weights = np.ones(self.x_train.shape[0])

    #         self.Neff = np.sum(self.weights)

    #         print ("Neff =",str(self.Neff))
    #         print ("Data Shape =",self.x_train.shape)
start = time.time()
DataHelper("small",0.2,working_dir="/home/as974/ada/multimerCorrection")
print('Datahelper: ',time.time()-start,' seconds')
