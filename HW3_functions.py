import math
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy.matlib
import os

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
    n_colors = 16
    RGBvector = np.zeros((n_colors, n_colors, n_colors), dtype = int)

    I_B, I_G, I_R = cv2.split(I)

    # keep the patch in the image boundaries
    ylow = np.clip(S[1] - S[3], 0, I.shape[0] - 1)
    yhigh = np.clip(S[1] + S[3], 1, I.shape[0])
    xlow = np.clip(S[0] - S[2], 0, I.shape[1] - 1)
    xhigh = np.clip(S[0] + S[2], 1, I.shape[1])

    patch_R = (I_R[ylow:yhigh, xlow:xhigh]/n_colors).astype(int)  # cut the patch from image
    patch_B = (I_B[ylow:yhigh, xlow:xhigh]/n_colors).astype(int)  # cut the patch from image
    patch_G = (I_G[ylow:yhigh, xlow:xhigh]/n_colors).astype(int)  # cut the patch from image


    # Making a 4096 vector for every permutation of RGB values
    for x in range(patch_R.shape[1]):  # shape[1] is width
        for y in range(patch_R.shape[0]):  # shape[0] is height
            RGBvector[patch_R[y, x], patch_G[y, x], patch_B[y, x]] += 1

    RGBvector = np.reshape(RGBvector, (1,4096))
    # Return normalized vector
    if np.sum(RGBvector) == 0:
        return RGBvector
    return RGBvector / np.sum(RGBvector)


def predictParticles(S_next_tag):
    # INPUT  = S_next_tag (previously sampled particles)
    # OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
    mean, sigma = 0, 2
    S_next = np.copy(S_next_tag)

    # Add velocity to pixels value
    S_next[0, :] += S_next[4, :]  # X + Vx
    S_next[1, :] += S_next[5, :]  # Y + Vy

    # Add normal noise, each particle gets different noise
    for s in range(S_next.shape[0]):
        if s != 2 and s != 3:
            noise = np.round(np.random.normal(mean, sigma, S_next.shape[1])).astype(int)
            S_next[s, :] += noise

    # Keeping the boundaries
    S_next[0][S_next[0] < S_next[2][0]] = S_next[2][0]
    S_next[1][S_next[1] < S_next[3][0]] = S_next[3][0]

    return S_next


def compBatDist(p, q):
    # INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    # OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
    return np.exp(20 * np.sum(np.sqrt(np.multiply(p, q))))


"""IMPORTANT - YOU WILL USE THIS FUNCTION TO UPDATE THE INDIVIDUAL WEIGHTS
 OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
 THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
 AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
 NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1"""


def sampleParticles(S_prev, C):
    # INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
    # OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)
    S_next_tag = np.zeros_like(S_prev)
    for n in range(len(C)):
        # r is random number - uniform distribution
        r = np.random.uniform(0, 1)
        minValue = 100

        # find the minimal value which is bigger than r
        for c in C:
            if c >= r:
                break

        # Get the index and store it in the next S
        j = C.index(c)
        S_next_tag[:, n] = S_prev[:, j]

    return S_next_tag


def showParticles(I, S, W, i, ID):
    # INPUT = I (current frame), S (current state vector)
    #        W (current weight vector), i (number of current frame)
    #        ID

    """ ANOTHER OPTION TO SHOW AVERAGE PARTICLE
    # Finding the average particle from weight vector
    average_particle_weight = np.mean(W)
    average_particle_index = np.argmin((np.abs(W - average_particle_weight)))
    S_average_particle = S[:, average_particle_index]

    green_start_point = (S_average_particle[0] - S_average_particle[2], S_average_particle[1] - S_average_particle[3])
    green_stop_point = (S_average_particle[0] + S_average_particle[2], S_average_particle[1] + S_average_particle[3])

    left = np.sum(S[0, :] @ W[:]) - S[2, 1]
    bot = np.sum(S[1, :] @ W[:]) - S[3, 1]
    right = np.sum(S[0, :] @ W[:]) + S[2, 1]
    top = np.sum(S[1, :] @ W[:]) + S[3, 1]"""

    # Adding green rectangle around the average particle
    left = np.mean(S[0, :]) - S[2, 1]
    bot = np.mean(S[1, :]) - S[3, 1]
    right = np.mean(S[0, :]) + S[2, 1]
    top = np.mean(S[1, :]) + S[3, 1]

    green_start_point = (int(left), int(bot))
    green_stop_point = (int(right), int(top))

    I = cv2.rectangle(I, green_start_point, green_stop_point, (0, 255, 0), 2, lineType=8, shift=0)  # color in BGR

    # Finding the maximal particle from weight vector
    maximal_particle_index = np.argmax(W)
    S_maximal_particle = S[:, maximal_particle_index]

    # Adding red rectangle around the maximal particle
    red_start_point = (S_maximal_particle[0] - S_maximal_particle[2], S_maximal_particle[1] - S_maximal_particle[3])
    red_stop_point = (S_maximal_particle[0] + S_maximal_particle[2], S_maximal_particle[1] + S_maximal_particle[3])
    I = cv2.rectangle(I, red_start_point, red_stop_point, (0, 0, 255), 2, lineType=8, shift=0)  # color in BGR

    # Displaying image
    figure, axs = plt.subplots(1)
    figure = axs.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

    plt.title("{0}- Frame number = {1}".format(ID, int(i)))

    # saving and showing the wanted images
    plt.savefig("{0}-{1}.png".format(ID, int(i)))
    plt.show(block=False)
    plt.pause(0.01)   # otherwise figure is not responding due to "block = False"

