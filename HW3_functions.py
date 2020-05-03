def compNormHist(I, S):
 #INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
 #OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1 VECTOR. NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)


def predictParticles(S_next_tag):
 #INPUT  = S_next_tag (previously sampled particles)
 #OUTPUT = S_next (predicted particles. weights and CDF not updated yet)


def compBatDist(p, q):
 #INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
 #OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)

 """IMPORTANT - YOU WILL USE THIS FUNCTION TO UPDATE THE INDIVIDUAL WEIGHTS
 OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
 THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
 AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
 NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1"""

def sampleParticles(S_prev, C):
# INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
# OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)


def showParticles(I, S):
 #INPUT = I (current frame), S (current state vector)
 #        W (current weight vector), i (number of current frame)
 #        ID