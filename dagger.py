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
        letters = []
        f = open(self.ocr_path, 'r')
        for line in f.readlines():
            line = line.split('\t')
            if int(line[2]) == -1:
                line.remove('\n')
                letters.append(map(lambda letter: letter.encode('utf-8'), line))
                self.words.append(letters)
                letters = []
                continue
            line.remove('\n')
            letters.append(map(lambda letter: letter.encode('utf-8'), line))
        f.close()
