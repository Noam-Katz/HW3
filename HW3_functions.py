import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    patch = I[(S[1] - S[3]):(S[1] + S[3]), (S[0] - S[2]):(S[0] + S[2])] # cut the patch from image
    n_colors = 16
    RGBvector = [0] * 4096

    # Quantization to 4 bits (0-15)
    for i in range(3):
        indexRange = (np.max(patch[:, :, i]) - np.min(patch[:, :, i]))
        quantUnit = indexRange // n_colors
        patch[:, :, i] = (patch[:, :, i] - np.min(patch[:, :, i])) // quantUnit

    # Making a 4096 vector for every permutation of RGB values
    for x in range(patch.shape[1]):  # shape[1] is width
        for y in range(patch.shape[0]):  # shape[0] is height
            R = patch[y, x, 1]
            G = patch[y, x, 1]
            B = patch[y, x, 1]
            RGBvector[R + (G * 15) + (B * 15 * 15)] += 1

    # Return normalized vector
    return RGBvector / np.sum(RGBvector)


def predictParticles(S_next_tag):
    # INPUT  = S_next_tag (previously sampled particles)
    # OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
    mean, sigma = 0, 1
    S_next = S_next_tag

    # Add velocity to pixels value
    S_next[0, :] = np.round(S_next[0, :] + S_next[4, :])  # X + Vx
    S_next[1, :] = np.round(S_next[1, :] + S_next[5, :])  # Y + Vy

    # Add normal noise, each particle gets different noise
    for s in range(max(S_next.shape)):
        noise = np.round(np.random.normal(mean, sigma, 6)).astype(int)
        S_next[:, s] += noise

    return S_next


def compBatDist(p, q):
    # INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    # OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
    return math.exp(20 * np.sum(np.sqrt(np.multiply(q, p))))


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
        minValue = 1

        # find the minimal value which is bigger than r
        for c in C:
            if c >= r:
                if c < minValue:
                    minValue = c

        # Get the index and store it in the next S
        j = C.index(minValue)
        S_next_tag[:, n] = S_prev[:, j]

    return S_next_tag


def showParticles(I, S, W, i, ID):
    # INPUT = I (current frame), S (current state vector)
    #        W (current weight vector), i (number of current frame)
    #        ID

    # Finding the average particle from weight vector
    average_particle = np.mean(W)
    average_particle_index = np.min(np.argwhere(W = average_particle))
    S_average_particle = S[: ,average_particle_index]

    # Adding green rectangle around the average particle
    x_start, y_start = S_average_particle[0] - S_average_particle[2] , S_average_particle[1] - S_average_particle[3]
    x_stop, y_stop = S_average_particle[0] + S_average_particle[2] , S_average_particle[1] + S_average_particle[3]
    #rect_average_green = patches.Rectangle((x_start, y_start) ,S_average_particle[2] * 2, S_average_particle[3] * 2, lw = 1, ec = 'g', fc = 'none')
    rect_average_green = cv2.rectangle(I, (x_start, y_start), (x_stop, y_stop), (0, 255, 0), 1, lineType=8, shift=0) # color in BGR


    # Finding the maximal particle from weight vector
    maximal_particle = np.argmax(W)
    maximal_particle_index = np.min(np.argwhere(W = maximal_particle))
    S_maximal_particle = S[:, maximal_particle_index]

    # Adding red rectangle around the maximal particle
    x_start, y_start = S_maximal_particle[0] - S_maximal_particle[2] , S_maximal_particle[1] - S_maximal_particle[3]
    x_stop, y_stop = S_maximal_particle[0] + S_maximal_particle[2] , S_maximal_particle[1] + S_maximal_particle[3]
    #rect_max_red = patches.Rectangle((x_start, y_start) ,S_average_particle[2] * 2, S_average_particle[3] * 2, lw = 1, ec = 'r', fc = 'none')
    rect_max_red = cv2.rectangle(I, (x_start, y_start), (x_stop, y_stop), (0, 0, 255), 1, lineType=8, shift=0) # color in BGR

    # Displaying image
    figure, axs = plt.subplots(1)
    figure = axs.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

    figure.axes.get_xaxis().set_visible(False)
    figure.axes.get_yaxis().set_visible(False)
    #figure.axis('off')

    # axs.add_patch(rect_max_red)
    # axs.add_patch(rect_average_green)

    plt.title("HW3_{0}- Frame number = {1}".format(ID, int(i)))

    # saving the wanted images
    if i % 10 == 0:
        plt.savefig("HW3_{0}-{1}.png".format(ID, int(i)))

    plt.show(block = False)