# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:04:48 2017

@authors: Kimia Nadjahi & Rachid Riad 
"""
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings("ignore")

import os.path
import numpy as np
import matplotlib.pyplot as plt

import codecs
from scipy import ndimage, misc
from scipy import sparse

from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import hamming_loss
from sklearn.multiclass import OneVsRestClassifier

import string
import pdb

#Global dictionary for letter classes

dictionary = dict(zip(string.ascii_lowercase, range(0,26)))

class DAgger(object):
    """Class for DAgger algorithm for Imitation Learning"""
    
    def __init__(self,ocr_path = 'letter.data'):
        self.ocr_path = os.path.join("Dataset/", str(ocr_path))
        self.words = []
        self.words_fold = []
        self.sequences = []
        self.dataset = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        self.labels = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
        
    def process_ocr(self):
        """Process ocr dataset"""
        print('Process ocr dataset')
        letters = []
        sequence = []
        f = open(self.ocr_path, 'r')
        for line in f.readlines():
            line = line.split('\t')
            if int(line[2]) == -1:
                line.remove('\n')
                sequence.append(dictionary[line[1]])
                image = np.array(map(lambda letter: int(letter), line[6:]))
                letters.append(image)
                self.words.append(letters)
                self.sequences.append(sequence)
                self.words_fold.append(int(line[5]))
                letters = []
                sequence = []
                continue
            line.remove('\n')
            line = map(lambda letter: letter.encode('utf-8'), line)
            sequence.append(dictionary[line[1]])
            image = np.array(map(lambda letter: int(letter), line[6:]))
            letters.append(image)
        f.close()
        
    def build_initial_dataset(self):
        """Build Initial dataset of trajectories to mimic"""
        print('Build dataset and labels')
        self.process_ocr()
        num_words = len(self.words)
        for idx in range(num_words): 
            for jdx in range(1,len(self.words[idx])): 
                X = np.zeros((128,26))
                X[:,self.sequences[idx][jdx-1]] = self.words[idx][jdx]
                X = X.reshape((1,3328))
                X = sparse.coo_matrix(X)
                self.dataset[self.words_fold[idx]].append(X)
                self.labels[self.words_fold[idx]].append(self.sequences[idx][jdx])        
        
    def aggregate_dataset(self, hat_policy, agg_data, test_fold = 9):
        """Aggregate original trajectories dataset with new generated from policy"""
        print('Aggregate dataset')
        num_words = len(self.words)
        for idx in range(num_words):
            if self.words_fold[idx] != test_fold: 
                for jdx in range(1,len(self.words[idx])):
                    X = np.zeros((128,26))
                    X[:,self.sequences[idx][jdx-1]] = self.words[idx][jdx]
                    X = X.reshape((1,3328))
                    X = sparse.coo_matrix(X)
                    y_pred = hat_policy.predict(X)
                    if y_pred != self.sequences[idx][jdx] and jdx != (len(self.words[idx])-1):
                        X_new = np.zeros((128,26))                        
                        X_new[:,y_pred[0]] = self.words[idx][jdx+1]
                        
                        X_new = X_new.reshape((1,3328))
                        X_new = sparse.coo_matrix(X_new)

                        agg_data[0].append(X_new)
                        agg_data[1].append(self.sequences[idx][jdx+1])
        print("Dataset size after aggregation")
        print(len(agg_data[0]))
        return agg_data
                    
    def run(self, N = 10):
        """Fit the policy classifier trained on a given dataset"""
        print('full run')
        self.build_initial_dataset()        
        final_scores = np.zeros(N)
        test_fold = 9
        hat_policies = []
        X_train = sum([self.dataset[i] for i in self.dataset.keys() if i != test_fold], [])
        y_train = sum([self.labels[i] for i in self.labels.keys() if i != test_fold], [])
        agg_data = [X_train,y_train]
        print("Dataset size at the beginning")
        print(len(agg_data[0]))
        hat_policy = svm.SVC(C = 10,kernel='linear')
#        hat_policy = linear_model.SGDClassifier(loss="squared_hinge", penalty="l2", n_iter=30)
        hat_policy.fit(sparse.vstack(agg_data[0]),np.vstack(agg_data[1]))
        hat_policies.append(hat_policy)
        
        hamming_scores = []
        y_pred = hat_policy.predict(sparse.vstack(self.dataset[test_fold]))
        print(1-hamming_loss(y_pred,self.labels[test_fold]))
        hamming_scores.append(hamming_loss(y_pred,self.labels[test_fold]))
        
        for iter in range(1,N):
            print(iter)
            agg_data = self.aggregate_dataset(hat_policy, agg_data, test_fold)
            hat_policy = svm.SVC(C = 10,kernel='linear')
#            hat_policy = linear_model.SGDClassifier(loss="squared_hinge", penalty="l2", n_iter=30)

            hat_policy.fit(sparse.vstack(agg_data[0]),agg_data[1])
            hat_policies.append(hat_policy)
            y_pred = hat_policy.predict(sparse.vstack(self.dataset[test_fold]))
            print(1-hamming_loss(y_pred,self.labels[test_fold]))
            hamming_scores.append(hamming_loss(y_pred,self.labels[test_fold]))
        final_scores = 1-np.array(hamming_scores)
        t = np.array(range(N))+1
        svm_struct_scores = np.zeros(N) + final_scores[0]
        plt.plot(t,final_scores,label='DAgger')
        plt.plot(t,svm_struct_scores,label = 'SVM_struct')
        plt.legend()
        plt.ylabel('Average character accuracy')
        plt.xlabel('Training iteration')
        plt.show()
        return final_scores, hat_policies
        
if __name__ == "__main__":
    dag = DAgger()
    scores, hat_policies = dag.run()
        
        