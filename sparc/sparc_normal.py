"""
AMP SPARC decoder with Hadamard matrices.
"""

import numpy as np

class Sparc:
    def __init__(self, sparc_params, Ax, Ay):
        """
        Initialise a SPARC with codeword matrix provided as the Ax and Ay functions
        and power allocation P.
        This defines the constants n, L, M.
        """
        self.p = sparc_params

        # Vector of length M x L with each entry equal to sqrt(nP_l)
        self.sqrtnPl = np.sqrt(self.p.n * np.repeat(self.p.Palloc, self.p.M))

        # Average power
        self.P_total = np.sum(self.p.Palloc)

        # Functions to calculate Ax, A^T y
        self.Ax, self.Ay = Ax, Ay

    def encode(self, x):
        """
        Encode the length-L vector of integers x between 1 and M into a codeword.
        """
        assert(x.size == self.p.L)
        beta = np.zeros(self.p.L * self.p.M)

        for l in range(self.p.L):
            beta[l * self.p.M + x[l]] = 1
        beta = beta * self.sqrtnPl

        codeword = self.Ax(beta)
        return codeword

    def eta(self, s, tau_sq):
        u = s * self.sqrtnPl / tau_sq
        max_u = u.max()  # Regularise argument to exponential so that it's smaller than 1.
        numerator = np.exp(u - max_u)
        denominator = np.zeros(self.p.L)
        for l in range(self.p.L):
            denominator[l] = np.sum(numerator[l*self.p.M : (l+1)*self.p.M])
        eta = self.sqrtnPl * numerator / np.repeat(denominator, self.p.M)
        return eta

    def decode(self, y):
        """
        AMP decoder. Decoding terminates when tau has stopped changing. 
        """
        assert(y.size == self.p.n)
        # Setup!
        beta = np.zeros(self.p.M * self.p.L)  # beta_0 = 0
        z = y  # z_0 = y
        s = beta + self.Ay(z)
        tau_sq = np.dot(z,z) / self.p.n
        tau_sq_prev = tau_sq + 1

        # Iterate!
        t = 1
        decoding_threshold = 5*self.p.Palloc[self.p.L-1]
        while tau_sq_prev - tau_sq >= decoding_threshold:
            #print('t = {}, tau_sq = {}, avg(beta^2) = {}'.format(t, tau_sq, np.dot(beta, beta) / self.p.n))
            
            # Calculate beta^t = eta^(t-1) (s_(t-1))
            beta = self.eta(s, tau_sq)

            # Calculate z_t = y - A beta^t - z_(t-1) / tau_(t-1)^2 * (P_total - (beta^t)^2 / n)
            z = y - self.Ax(beta) + z / tau_sq * (self.P_total - np.dot(beta, beta) / self.p.n)

            # Calculate s^t = beta^t + A^T z^(t)
            s = beta + self.Ay(z)

            # Calculate tau_t^2 = z_t^2 / n
            tau_sq_prev = tau_sq
            tau_sq = np.dot(z,z) / self.p.n

            t += 1

        # Declare the maximum value in each section to be the decoded '1'.
        x = np.zeros(self.p.L, dtype=int)

        for l in range(self.p.L):
            index = beta[l * self.p.M : (l+1) * self.p.M].argmax()
            x[l] = index
        return x

if __name__ == "__main__":
    import sys
    sys.path.append('/home/antoine/Documents/IIB Engineering/Project/Code')
    import sparc as sp

    ## Setup
    # Use SNR of 15 for C = 2 bits
    P_total = 15

    C = sp.awgn_capacity(P_total)
    R = 0.8*C
    L = 1024
    logM = 9
    logK = 1

    Palloc = sp.modified_power_alloc(0.7, 0.6, P_total, L, C)

    p = sp.SparcParams.from_rate(R, 1, L, logK, logM, Palloc)

    (Ax, Ay) = sp.sparc_transforms(p.n, p.M, p.L, sp.gen_ordering(p.n, p.M, p.L))
    sparc = Sparc(p, Ax, Ay)

    ## Generate vector to transmit.
    input = np.random.randint(0, p.M, size=L)

    ## Encode it
    x = sparc.encode(input)

    ## Transmit it
    y = x + np.random.normal(size=p.n)

    ## Decode it
    output = sparc.decode(y)

    print(input)
    print(output)