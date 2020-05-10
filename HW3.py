from math import exp

import cv2
import numpy as np
import numpy.matlib
from HW3_functions import *
import os

"""
 HW 3, COURSE 0512-4263, TAU 2020

 PARTICLE FILTER TRACKING

 THE PURPOSE OF THIS ASSIGNMENT IS TO IMPLEMENT A PARTICLE FILTER TRACKER
 IN ORDER TO TRACK A RUNNING PERSON IN A SERIES OF IMAGES.

 IN ORDER TO DO THIS YOU WILL WRITE THE FOLLOWING FUNCTIONS:

 compNormHist(I, S)
 INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
 OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1
                    VECTOR. NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)


 predictParticles(S_next_tag)
 INPUT  = S_next_tag (previously sampled particles)
 OUTPUT = S_next (predicted particles. weights and CDF not updated yet)


 compBatDist(p, q)
 INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
 OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)

 IMPORTANT - YOU WILL USE THIS FUNCTION TO UPDATE THE INDIVIDUAL WEIGHTS
 OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
 THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
 AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
 NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1

 sampleParticles(S_prev, C)
 INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
 OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)


 showParticles(I, S)
 INPUT = I (current frame), S (current state vector)
         W (current weight vector), i (number of current frame)
         ID 
"""
ID1 = "204251144"
ID2 = "200940500"

ID = "HW3_{0}_{1}".format(ID1, ID2)
IMAGE_DIR_PATH = "{0}\\Images".format(os.getcwd())

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,  # x center
             139,  # y center
             16,  # half width
             43,  # half height
             0,  # velocity x
             0]  # velocity y

# CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
S = predictParticles(np.matlib.repmat(s_initial, N, 1).T)

# LOAD FIRST IMAGE
I = cv2.imread(IMAGE_DIR_PATH + os.sep + "001.png")

# COMPUTE NORMALIZED HISTOGRAM
q = compNormHist(I, s_initial)

# COMPUTE BAT DISTANCE (W)
W = np.zeros(100)
for j in range(N):
    p = compNormHist(I, S[:, j])
    W[j] = compBatDist(p, q)

# COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
W = W / np.sum(W)
C = (np.cumsum(W)).tolist()

images_processed = 1

# MAIN TRACKING LOOP
image_name_list = os.listdir(IMAGE_DIR_PATH)
for image_name in image_name_list[1:]:

    S_prev = S

    # LOAD NEW IMAGE FRAME
    image_path = IMAGE_DIR_PATH + os.sep + image_name

    I = cv2.imread(image_path)

    # SAMPLE THE CURRENT PARTICLE FILTERS
    S_next_tag = sampleParticles(S_prev, C)

    # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
    S = predictParticles(S_next_tag)

    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(I, s_initial)
    # p = compNormHist(I, S[:,0])

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:

    # COMPUTE BAT DISTANCE (W)
    for j in range(N):
        p = compNormHist(I, S[:, j])
        W[j] = compBatDist(p, q)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = W / np.sum(W)

    C = [0] * N
    C[0] = W[0]
    C = (np.cumsum(W)).tolist()

    # CREATE DETECTOR PLOTS
    images_processed += 1
    i = images_processed
    if 0 == images_processed % 10:
        showParticles(I, S, W, i, ID)

    if images_processed > 110:
        break
