import numpy as np

def awgn_capacity(Pchan):
    return 1/2 * np.log2(1 + Pchan)

def minimum_Pchan(C):
    return 2 ** (2*C) - 1