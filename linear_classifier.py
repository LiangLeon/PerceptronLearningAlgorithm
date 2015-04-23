# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 16:29:03 2015

@author: LiangLeon
"""

import numpy as np
 
class Classifier:
    '''Class to represent a linear function and associated linear 
    classifier on n-dimension space.'''
 
    def __init__(self, vect_w=None):
        '''Initializes coefficients, if None then
        must be initialized later.
 
        :param vect_w: vector of coefficients.'''
        self.vect_w = vect_w
 
    def init_random_last0(self, dimension, seed=None):
        '''
        Initializes to random vector with last coordinate=0,
        uses seed if provided.
 
        :params dimension: vector dimension;
 
        :params seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)
        self.vect_w = -1 + 2*np.random.rand(dimension)
        self.vect_w[-1] = 0  # exclude class coordinate
 
    def value_on(self, vect_x):
        '''Computes value of the function on vector vect_x.
 
        :param vect_x: the argument of the linear function.'''
        return sum(p * q for p, q in zip(self.vect_w, vect_x))
 
    def class_of(self, vect_x):
        '''Computes a class, one of the values {-1, 1} on vector vect_x.
 
        :param vect_x: the argument of the linear function.'''
        return 1 if self.value_on(vect_x) >= 0 else -1
        
    def intersect_aabox2d(self, box=None):
        if box is None:
            box = ((-1,-1),(1,1))
 
        minx = min(box[0][0], box[1][0])
        maxx = max(box[0][0], box[1][0])
        miny = min(box[0][1], box[1][1])
        maxy = max(box[0][1], box[1][1])
 
        intsect_x = []
        intsect_y = []
 
        for side_x in (minx, maxx):
            ya = -(self.vect_w[0] + self.vect_w[1] * side_x)/self.vect_w[2]
            if ya >= miny and ya <= maxy:
                intsect_x.append(side_x)
                intsect_y.append(ya)
        for side_y in (miny, maxy):
            xb = -(self.vect_w[0] + self.vect_w[2] * side_y)/self.vect_w[1]
            if xb <= maxx and xb >= minx:
                intsect_x.append(xb)
                intsect_y.append(side_y)
        return intsect_x, intsect_y