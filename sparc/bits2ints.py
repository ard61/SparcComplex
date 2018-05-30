import numpy as np

def bits2ints(bits, width):
    """
    Transform an array of bits (ints in {0,1})
    into an array of ints
    where each int is sampled from `width` bits.
    """
    assert(bits.size % width == 0)
    L = bits.size // width
    ints = np.zeros(L, dtype=int)
    for l in range(L):
        cur_int = 0
        for bit in bits[l*width: (l+1)*width]:
            cur_int = (cur_int << 1) | bit
        ints = ints.append(cur_int)
    return ints

def ints2bits(ints, width):
    """
    Transform an array of ints
    into an array of bits
    where each int maps to `width` bits.
    """
    bits = np.zeros(width * ints.size, dtype=int)
    for l in range(ints.size):
        bits[l*width : (l+1)*width] = np.array([int(bit) for bit in np.binary_repr(ints[l], width=width)])
    return bits