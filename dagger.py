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
        
    def build_iniial_dataset(self):
        """Build Initial dataset of trajectories to mimic"""

    def aggregate_dataset(self,D,clf):
        """Aggregate original trajectories dataset with new generated"""
        return D
        
    def fit_policy(self,D):
        """Fit the policy classifier trained on a given dataset"""
        clf = svm.SVC()
        return clf
        
        
        
        
        