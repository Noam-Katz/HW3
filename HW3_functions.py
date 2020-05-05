import cv2
import numpy as np
import numpy.matlib
from certifi.__main__ import args

from HW3_functions import *
import os
import argparse
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from skimage import io
from sklearn.cluster import KMeans

"""
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y
"""


def compNormHist(I, S):
    # INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    # OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1 VECTOR. NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)
    patch = I[(S[1] - S[3]):(S[1] + S[3]), (S[0] - S[2]):(S[0] + S[2])]
    n_colors = 16
    RGBvector = [0]*4096
    for i in range(3):
        indexRange = (np.max(patch[:, :, i]) - np.min(patch[:, :, i]))
        quantUnit = indexRange // n_colors
        patch[:, :, i] = (patch[:, :, i] - np.min(patch[:, :, i])) // quantUnit

    for x in range(patch.shape[1]):         # shape[1] is width
        for y in range(patch.shape[0]):     # shape[0] is height
            R = patch[y, x, 1]
            G = patch[y, x, 1]
            B = patch[y, x, 1]
            RGBvector[R+(G*15)+(B*15*15)] += 1

    return RGBvector / np.sum(RGBvector)

def predictParticles(S_next_tag):
    # INPUT  = S_next_tag (previously sampled particles)
    # OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
    S_next = S_next_tag
    return S_next


def compBatDist(p, q):
    # INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    # OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
    i = 1
    """IMPORTANT - YOU WILL USE THIS FUNCTION TO UPDATE THE INDIVIDUAL WEIGHTS
 OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
 THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
 AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
 NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1"""


def sampleParticles(S_prev, C):
    # INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
    # OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)
    i = 1


def showParticles(I, S):
    # INPUT = I (current frame), S (current state vector)
    #        W (current weight vector), i (number of current frame)
    #        ID
    i = 1
