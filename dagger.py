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
#import utils 

class DAgger(object):
    """Class for DAgger algorithm for Imitation Learning"""
    
    def __init__(self,ocr_path = 'letter.data'):
        self.ocr_path = os.path.join("Dataset/", str(ocr_path))
        self.words = []
        self.words_fold = []
        self.sequences = []
    
    def process_ocr(self):
        """Process ocr dataset"""
        letters = []
        sequence = []
        f = open(self.ocr_path, 'r')
        for line in f.readlines():
            line = line.split('\t')
            if int(line[2]) == -1:
                line.remove('\n')
                sequence.append(line[1])
                image = np.array(map(lambda letter: int(letter), line[6:])).reshape((16,8))
                letters.append(image)
                self.words.append(letters)
                self.sequences.append(sequence)
                self.words_fold.append(line[5])
                letters = []
                sequence = []
                continue
            line.remove('\n')
            line = map(lambda letter: letter.encode('utf-8'), line)
            sequence.append(line[1])
            image = np.array(map(lambda letter: int(letter), line[6:])).reshape((16,8))
            letters.append(image)
        f.close()
        
    