# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:04:48 2017

@authors: Kimia Nadjahi & Rachid Riad 
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt

import codecs
from scipy import ndimage, misc

from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import hamming_loss

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
        self.process_ocr()
        num_words = len(self.words)
        for idx in range(num_words): 
            for jdx in range(1,len(self.words[idx])): 
                X = np.concatenate([self.words[idx][jdx], np.array([self.sequences[idx][jdx-1]])],  axis = 0)   
#                pdb.set_trace()
                self.dataset[self.words_fold[idx]].append(X.tolist())
                self.labels[self.words_fold[idx]].append(self.sequences[idx][jdx])        
        
    def aggregate_dataset(self, agg_data = [], hat_policy, test_fold = 9):
        """Aggregate original trajectories dataset with new generated from policy"""
        num_words = len(self.words)
        for idx in range(num_words):
            if self.words_fold[idx] != test_fold: 
                for jdx in range(1,len(self.words[idx])):
                    X = np.concatenate([self.words[idx][jdx], np.array([self.sequences[idx][jdx-1]])],  axis = 0)
                    y_pred = hat_policy.predict(X)
                    if y_pred != self.sequences[idx][jdx] and jdx != (len(self.words[idx]-1)):
                        X_new = np.concatenate([self.words[idx][jdx+1], y_pred],  axis = 0)
                        agg_data.append([X_new,self.sequences[idx][jdx+1]])
        return agg_data
                    
    def run(self, X,y):
        """Fit the policy classifier trained on a given dataset"""
        clf = linear_model.SGDClassifier()
        clf.fit(X,y)
        return clf
        
    
        
        