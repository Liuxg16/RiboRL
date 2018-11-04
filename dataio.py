import sys, os
import re, fileinput, math
import numpy as np
import random
# import h5py
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import pandas as pd
import cPickle as pickle, json

class Dataset_yeast(object):

    def __init__(self, rnd_seed=1234,testrate = 0.3,vocab=None, relative_offset = 0,\
            path = 'data/A-codon-site.txt'):
        self.path = path
        self.rnd_seed = 1234

        test_data, test_labels, vocab1 = self.readfile()
        if vocab is not None:
            self.vocab = vocab
        else:
            # self.vocab = self.genVocab(self.rawdata)
            self.vocab = vocab1

        char_vocab = {'A':4,'C':1,'G':2, 'T':3}
        self.nt_matrix = [[] for i in range(len(self.vocab))]
        for word, id in self.vocab.items():
            if 'k' in word:
                self.nt_matrix[id] = [0,0,0]
            else:
                self.nt_matrix[id] = [char_vocab[c] for c in word]

        self.window_condon = relative_offset
        test_data = self.word2index(self, test_data, self.vocab)

        dist5, dist3 = None, None

        self.test_data, self.test_labels,_ =   \
            process_ribo_normal(test_data ,test_labels, self.vocab, self.window_condon, dist5,
                    dist3)


        print('----yeast Dataset----loaded data-----')

    def readfile(self):

        test_data, test_labels =[],[]
        testf = open('./data/yeast-dataset/testlabels.txt','rb')
        docs_= open('./data/yeast-dataset/testdata.txt',"rb").read()
        labels = testf.read().strip().split('\n')

        docs = docs_.strip().split('\n')
        n_sample = int(len(docs)/2)
        for i in range(n_sample):
            name = docs[2*i]
            text = docs[2*i+1].strip().split()
            test_data.append(text)
            label = [float(x) for x in labels[i].split()]
            test_labels.append(label)



        with open('./data/yeast-dataset/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)

        # print 'loaded train {}'.format(len(train_data))
        # print 'loaded valid {}'.format(len(valid_data))
        print 'loaded test {}'.format(len(test_data))



        return  test_data, test_labels,vocab

    def genVocab(self,lines, maskid=0):
        char_vocab = {'A':4,'C':1,'G':2, 'T':3}
        """generate vocabulary from contents"""
        #lines = [' '.join(line) for line in lines]
        wordset = set(item for line in lines for item in line)
        freq = {word: 0 for index, word in enumerate(wordset)}
        for line in lines:
            line_wordset = set(item for item in line)
            for word in line_wordset:
                freq[word] += 1
        high_word = []
        for (word,count) in freq.items():
            if count >0 and word != '<mask>':
                high_word.append(word)
        
        word2index = {word: index + 1 for index, word in enumerate(high_word)}
        print word2index

        nt_matrix = [[] for i in range(len(word2index)+2)]
        for word, id in word2index.items():
            nt_matrix[id] = [char_vocab[c] for c in word]
        word2index['<mask>'] = maskid
        word2index['unk'] = len(word2index) 
        nt_matrix[word2index['<mask>']] =  [0,0,0]
        nt_matrix[word2index['unk']] =  [0,0,0]

        self.nt_matrix = nt_matrix
        

        return word2index

    def getIndex(self,word):
        if self.vocab.has_key(word):
            return self.vocab[word]
        else:
            return self.vocab['unk']

    @staticmethod
    def word2index(self, docs, vocab):
        #docs = [' '.join(line) for line in docs]
        index_docs = [[self.getIndex(char) for char in doc] for doc in docs]
        # max_len = max([len(doc) for doc in index_docs])
        # index_docs = [doc+[vocab['mask']]*(max_len - len(doc)) for doc in index_docs]
        # index_docs = np.array(index_docs)
        return index_docs

    def getData(self):
        return  self.test_data, self.test_labels

    def getVocab(self):
        return self.vocab

def process_ribo_normal(docs_condons, docs_counts, vocab, window_condon, dist5=None,dist3=None):
    '''
    given the condon sequence and footprints counts
    return the condon context for each footprint
    add min-max normalization
    '''

    # rna_offset = (window_condon-1)/2
    offset_relative = int((window_condon-1)/2)
    offset_l =  offset_relative+int(window_condon-1)%2
    offset_r = offset_relative

    data = []
    labels = []
    genes_offsets = []
    offsett = 0
    for condon_seq,asite_seq1 in zip(docs_condons,docs_counts):
        genes_offsets.append(offsett)
        # print condon_seq
        assert len(condon_seq)==len(asite_seq1)
        n_seq = len(condon_seq)
        value_t = np.array(asite_seq1)
        # asite_seq = np.log(1+value_t.astype(float))
        asite_seq = value_t.astype(float)
        asite_sum = np.sum(asite_seq)
        n_valid_asite = np.sum(asite_seq>0.5) # each rna,total records
        asite_seq = asite_seq/(asite_sum/n_valid_asite)  # sum
        # asite_seq = (asite_seq-minv)/(np.max(asite_seq)-minv)  # min-max
        for i in range(n_seq):
            footprints = asite_seq[i]
            if footprints>1e-6:
                start = (i-offset_l)
                end = i+offset_r+1
                if start <0:
                    continue
                    #start = 0
                if end > n_seq:
                    continue
                    # end = n_seq
                datum = [vocab['<mask>']]*(offset_l-i+start)+[x for x in condon_seq[start:end]]\
                        +[vocab['<mask>']]*(offset_r+1-end+i)
                data.append(datum)
                labels.append(footprints)
                offsett +=1

    
    if dist3 != None:

        idx_minus5 = (window_condon-1)/2-5
        idx_plus3 =  (window_condon-1)/2+3
        ys = []
        for seq, count in zip(data,labels):

            cd_minus5 = seq[idx_minus5]
            cd_plus3 = seq[idx_plus3]
            res = count/(dist5[cd_minus5]*dist3[cd_plus3])
            ys.append(res)

        labels = ys

    assert len(data)==len(labels)

    data = np.array(data).astype(int)
    labels = np.array(labels)
    
    # print('loaded roseshape data:')
    # print(data.shape)
    # print(labels.shape)
   
    return data, labels, genes_offsets



if __name__ == "__main__":

    dataset = Dataset_asite('default',1234,0.3)


