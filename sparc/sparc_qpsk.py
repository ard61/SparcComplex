"""
AMP Complex QPSK SPARC decoder with DFT matrices.
"""

import numpy as np

def abs2(z):
    return np.sum(np.real(z)**2 + np.imag(z)**2)

class SparcQpsk:
    def __init__(self, sparc_params, Ax, Ay):
        """
        Initialise a modulated SPARC with codeword matrix provided as the Ax and Ay functions
        and power allocation P.
        This defines the constants n, L, K, M.
        """

        self.p = sparc_params
        self.sqrtnPl = np.sqrt(self.p.n * self.p.Palloc)

        # Optimal modulation constants
        self.a = np.exp(2 * np.pi * 1j * np.array(range(self.p.K)) / self.p.K)
        self.a = self.sqrtnPl.reshape(self.p.L, 1).repeat(self.p.K, axis=1) * self.a.reshape(1, self.p.K).repeat(self.p.L, axis=0)
        #print(self.a)
        assert(self.a.shape == (self.p.L, self.p.K))

        # Functions to calculate Ax, A^T y
        self.Ax, self.Ay = Ax, Ay

    def encode(self, x):
        """
        Encode the length-L vector of integers x between 1 and M into a codeword.
        """
        assert(x.size == self.p.L)
        beta = np.zeros((self.p.L, self.p.M), dtype=complex)

        ind = (x / self.p.K).astype(int)
        beta[range(self.p.L), ind] = self.a[range(self.p.L), x % self.p.K]

        codeword = self.Ax(beta.ravel())
        # Make sure average power per transmission is within 10% of P_total.
        #assert(abs2(codeword) / self.p.n - self.p.Pchan < self.p.Pchan/10)

        codeword = np.stack([np.real(codeword), np.imag(codeword)], -1)
        return codeword

    def eta(self, s, tau_sq):
        a = self.a.reshape(self.p.L, 1, self.p.K).repeat(self.p.M, axis=1)
        s = s.reshape(self.p.L, self.p.M, 1).repeat(self.p.K, axis=2)
        u = 2 * (np.real(s)*np.real(a) + np.imag(s)*np.imag(a)) / tau_sq
        u = np.exp(u - u.max())
        numerator = np.sum(a * u, axis=2)
        denominator = np.sum(np.sum(u, axis=2), axis=1)
        eta = numerator / denominator.reshape(self.p.L, 1).repeat(self.p.M, axis=1)
        return eta.ravel()

    def decode(self, y):
        """
        AMP decoder. Decoding terminates when tau has stopped changing. 
        """
        assert(y.shape == (self.p.n, self.p.D))

        # Recover y from the concatenated real and imaginary parts
        y = y[:,0] + 1j * y[:,1]
        
        # Setup!
        beta = np.zeros(self.p.L * self.p.M, dtype=complex)  # beta_0 = 0
        z = y  # z_0 = y
        s = beta + self.Ay(z)
        tau_sq = abs2(z) / self.p.n
        tau_sq_prev = tau_sq + 1

        # Iterate!
        t = 1
        decoding_threshold = 5*self.p.Palloc[self.p.L-1]
        while tau_sq_prev - tau_sq >= decoding_threshold:
            #print('t = {}, tau_sq = {}, avg(beta^2) = {}'.format(t, tau_sq, abs2(beta) / self.p.n))
        
            # Calculate beta^t = eta^(t-1) (s_(t-1))
            beta = self.eta(s, tau_sq)

            # Calculate z_t = y - A beta^t - z_(t-1) / tau_(t-1)^2 * (P_total - (beta^t)^2 / n)
            z = y - self.Ax(beta) + z / tau_sq * (self.p.Pchan - abs2(beta) / self.p.n)

            # Calculate s^t = beta^t + A^T z^(t)
            s = beta + self.Ay(z)

            # Calculate tau_t^2 = z_t^2 / n
            tau_sq_prev = tau_sq
            tau_sq = abs2(z) / self.p.n

            t += 1

        #print('Final tau^2: {}'.format(tau_sq))

        #final_stat = beta.reshape(self.L, self.M)
        final_stat = s.reshape(self.p.L, self.p.M)
        ind = np.abs(final_stat).argmax(axis=1)  # The indices of the largest-magnitude elements in each section.
        max_vals = final_stat[range(self.p.L),ind]
        #print(max_vals)
        mod_const = np.abs(self.a - max_vals.reshape(self.p.L, 1).repeat(self.p.K, axis=1)).argmin(axis=1)  # The modulation constant closest to that element

        x = ind * self.p.K + mod_const
        return x

if __name__ == "__main__":
    import sys
    sys.path.append('/home/antoine/Documents/IIB Engineering/Project/Code')
    import sparc as sp

    ## Setup
    P_total = 15  # Power bound: average power per real transmission
    sigma_sq = 1  # Average noise per real transmission

    SNR_per_bit = 4
    R = 1
    P_total = 2 * R * (10 ** (SNR_per_bit / 10))
    C = sp.awgn_capacity(P_total)
    #R = 0.8*C  # Number of bits / real transmission

    D = 2
    L = 1024
    logK = 2
    logM = 8

    a = 0
    f = 0

    Palloc = sp.modified_power_alloc(a, f, P_total, L, C)  # Power allocation: average power per real transmission in section l
    p = sp.SparcParams.from_rate(R, D, L, logK, logM, Palloc)

    (Ax, Ay) = sp.complex_sparc_transforms(p.n, p.M, p.L, sp.gen_ordering_dft(p.n, p.M, p.L))
    sparc = SparcQpsk(p, Ax, Ay)

    print({'C': p.C, 'R': p.R, 'n': p.n, 'P_total': p.Pchan})

    ## Generate vector to transmit.
    input = np.random.randint(0, p.M, size=p.L)

    ## Encode it
    x = sparc.encode(input)

    ## Transmit it
    y = x + np.random.normal(size=x.shape)

    ## Decode it
    output = sparc.decode(y)

    print(input)
    print(output)

    section_errors = input != output
    index_errors = input // p.K != output // p.K
    modulation_errors = section_errors - index_errors
    assert(np.any(modulation_errors == -1) == False)

    print('Section error rate: {}'.format(np.sum(section_errors) / L))
    print('Index errors: {}  ; at sections {}'.format(np.sum(index_errors), np.nonzero(index_errors)))
    print('Modulation constant errors {}  ; at sections {}'.format(np.sum(modulation_errors), np.nonzero(modulation_errors)))
