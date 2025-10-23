import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class DataReader:
    def __init__(self,data_path,window_size):
        self.data_path = data_path
        self.sequenceList = []
        self.sequences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.read_words()
        self.calc_neg_prob(window_size)

    def read_words(self):
        with open(self.data_path,'rb') as f:
            self.sequenceList = pickle.load(f)
        self.sequences_count = len(self.sequenceList)
        for sequence in self.sequenceList:
            for node in sequence:
                self.token_count += 1
                self.word_frequency[str(node)] = self.word_frequency.get(str(node),0) + 1
        wid = 0
        for w,c in self.word_frequency.items():
            self.word2id[w] = wid
            self.id2word[wid] = w
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def calc_neg_prob(self,window_size):
        self.co_occurrence = np.zeros((len(self.word2id),len(self.word2id)))
        for sequence in self.sequenceList:
            seq_len = len(sequence)
            for i in range(seq_len-1):
                neighbors = sequence[(i+1):(i+1+window_size)]
                for j in neighbors:
                    self.co_occurrence[self.word2id[str(sequence[i])]][self.word2id[str(j)]] += 1

        self.probs = {}
        for i in tqdm(range(len(self.word2id))):
            counts = self.co_occurrence[i]
            prob_pernode = {}
            for j in range(len(self.word2id)):
                if i != j:
                    center_count = counts[j]
                    dist_counts = center_count - counts
                    low_counts_pos = np.where(dist_counts>0)[0]
                    if len(low_counts_pos) != 0:
                        low_counts = dist_counts[low_counts_pos]
                        low_counts = low_counts/low_counts.sum()
                        label = [self.id2word[pos] for pos in low_counts_pos]
                        prob_pernode[self.id2word[j]] = {'prob':low_counts,'word':label}
            self.probs[self.id2word[i]] = prob_pernode

    def getNegatives(self,u,v,nneg):
        probs = self.probs[self.id2word[u]][self.id2word[v]]['prob']
        labels = self.probs[self.id2word[u]][self.id2word[v]]['word']
        tmp = np.random.choice(labels,p=probs,size=nneg)
        tmp = [self.word2id[str(t)] for t in tmp]
        return np.array(tmp)

class PoincareDataset(Dataset):
    def __init__(self,data,window_size,nneg):
        self.data = data
        self.window_size = window_size
        self.nneg = nneg
        self.samples = []
        self.create_sample()
    
    def create_sample(self):
        sequenceList = self.data.sequenceList
        for sequence in tqdm(sequenceList,desc="Sampling Negative Samples"):
            word_ids = [self.data.word2id[str(w)] for w in sequence]
            self.samples.extend([(u,v,self.data.getNegatives(u,v,self.nneg))
                for i,u in enumerate(word_ids)
                for j,v in enumerate(word_ids[(i+1):(i+self.window_size+1)])
                if (u!=v) and (self.data.id2word[v] in self.data.probs[self.data.id2word[u]])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        return self.samples[idx]