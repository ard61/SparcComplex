import numpy as np
from sparc.bits2ints import ints2bits

def error_rate(sparc):
    """
    Test a SPARC, giving the section and bit error rates as output.
    """
    logKM = sparc.p.logK + sparc.p.logM

    input = np.random.randint(0, 2**logKM, size=sparc.p.L)
    x = sparc.encode(input)
    y = x + np.random.normal(size=x.shape)
    output = sparc.decode(y)

    # Calculate codeword error rate
    CER = np.any(output != input)

    # Calculate section error rate
    l_err = np.sum(output != input)
    SER = l_err / sparc.p.L

    # Calculate bit error rate
    b_err = np.sum(ints2bits(input, logKM) != ints2bits(output, logKM))
    BER = b_err / (sparc.p.inputbits)

    return (CER, SER, BER)
