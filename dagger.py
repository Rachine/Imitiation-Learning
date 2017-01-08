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
        self.dataset = []
        self.labels = []
        
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
#        traj_word = []
        #TODO build good indexing for the train and test
        for idx in range(num_words): 
            for jdx in range(1,len(self.words[idx])): 
                X = np.concatenate([self.words[idx][jdx], np.array([self.sequences[idx][jdx-1]])],  axis = 0)   
#                pdb.set_trace()
                self.dataset.append(X.tolist())
                self.labels.append(self.sequences[idx][jdx])
                
    def aggregate_dataset(self,policy):
        """Aggregate original trajectories dataset with new generated from policy"""
                    
    def fit_policy(self):
        """Fit the policy classifier trained on a given dataset"""
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(self.dataset,self.labels)
        return clf
        
        
        
        
        