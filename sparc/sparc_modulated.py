"""
AMP Modulated SPARC decoder with Hadamard matrices.
"""

import numpy as np

class SparcModulated:
    def __init__(self, sparc_params, Ax, Ay):
        """
        Initialise a modulated SPARC with codeword matrix provided as the Ax and Ay functions
        and power allocation P.
        This defines the constants n, L, K, M.
        """
        self.p = sparc_params
        assert(self.p.K % 2 == 0)  # Symmetric modulation scheme

        self.sqrtnPl = np.sqrt(self.p.n * self.p.Palloc)

        # Optimal modulation constants
        self.a = [np.sqrt(3 / (self.p.K**2 - 1)) * (2*R - 1) for R in range(1, self.p.K//2 + 1)]
        self.a = np.array(self.a + [-a for a in self.a])
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
        beta = np.zeros((self.p.L, self.p.M))

        ind = (x / self.p.K).astype(int)
        beta[range(self.p.L), ind] = self.a[range(self.p.L), x % self.p.K]

        codeword = self.Ax(beta.ravel())
        #assert(np.abs(codeword.dot(codeword) / self.p.n - self.p.Pchan) < self.p.Pchan/5)  # Make sure average power is within 20% of the channel power constraint

        return codeword
        
    def eta(self, s, tau_sq):
        a = self.a.reshape(self.p.L, 1, self.p.K).repeat(self.p.M, axis=1)
        s = s.reshape(self.p.L, self.p.M, 1).repeat(self.p.K, axis=2)
        u = a / tau_sq * (s - a / 2)
        u = np.exp(u - u.max())
        self.u = u  # cache result
        numerator = np.sum(a * u, axis=2)
        denominator = np.sum(np.sum(u, axis=2), axis=1)
        eta = numerator / denominator.reshape(self.p.L, 1).repeat(self.p.M, axis=1)
        return eta.ravel()

    def onsager_frac(self, s):
        section_sum = np.sum(self.u, axis=1)
        num = np.sum(self.a**2 * section_sum, axis=1)
        den = np.sum(section_sum, axis=1)
        result = np.sum(num / den, axis=0)
        #if self.p.K == 2:
        #    #assert(np.abs(result/self.p.n - self.p.Pchan) < self.p.Pchan/10)  # Make sure our result is within 10% of the total energy
        return result

    def decode(self, y):
        """
        AMP decoder. Decoding terminates when tau has stopped changing. 
        """
        assert(y.size == self.p.n)
        # Setup!
        beta = np.zeros(self.p.L * self.p.M)  # beta_0 = 0
        z = y  # z_0 = y
        s = beta + self.Ay(z)
        tau_sq = np.dot(z,z) / self.p.n
        tau_sq_prev = tau_sq + 1

        # Iterate!
        t = 1
        decoding_threshold = 5*self.p.Palloc[self.p.L-1]
        while tau_sq_prev - tau_sq >= decoding_threshold:
            #print('t = {}, tau_sq = {}, avg(beta^2) = {}'.format(t, tau_sq, beta.dot(beta) / self.p.n))
            
            # Calculate beta^t = eta^(t-1) (s_(t-1))
            beta = self.eta(s, tau_sq)

            # Calculate z_t = y - A beta^t - z_(t-1) / tau_(t-1)^2 * (P_total - (beta^t)^2 / n)
            z = y - self.Ax(beta) + z / tau_sq * (self.onsager_frac(s) - beta.dot(beta)) / self.p.n

            # Calculate s^t = beta^t + A^T z^(t)
            s = beta + self.Ay(z)

            # Calculate tau_t^2 = z_t^2 / n
            tau_sq_prev = tau_sq
            tau_sq = z.dot(z) / self.p.n

            t += 1

        #print('Final tau^2: {}'.format(tau_sq))

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
    P_total = 63
    sigma_sq = 1

    #SNR_per_bit = 5
    #R = 1
    #P_total = 2 * R * (10 ** (SNR_per_bit / 10))
    C = sp.awgn_capacity(P_total)
    R = 0.8*C
    R_pa = 1.1 * R

    L = 1024
    logK = 2
    logM = 10

    Palloc = sp.iterative_power_alloc(P_total, L, R_pa)

    p = sp.SparcParams.from_rate(R, 1, L, logK, logM, Palloc)

    print({'C': C, 'R': R, 'R_pa': R_pa, 'n': p.n, 'P_total': P_total})

    (Ax, Ay) = sp.sparc_transforms(p.n, p.M, p.L, sp.gen_ordering(p.n, p.M, p.L))
    sparc = SparcModulated(p, Ax, Ay)

    ## Generate vector to transmit.
    input = np.random.randint(0, p.M, size=p.L)

    ## Encode it
    x = sparc.encode(input)

    ## Transmit it
    y = x + np.sqrt(sigma_sq) * np.random.normal(size=p.n)

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

    if p.K == 4:
        # Check whether errors are more likely to happen when transmitting the
        # lower-amplitude modulation constants?
        low_amplitude = np.bitwise_or(input % p.K == 0, input % p.K == 2)
        high_amplitude = np.bitwise_or(input % p.K == 1, input % p.K == 3)

        print('low amplitude error rate: {}'.format(np.sum(np.bitwise_and(section_errors, low_amplitude)) / np.sum(low_amplitude)))
        print('high amplitude error rate: {}'.format(np.sum(np.bitwise_and(section_errors, high_amplitude)) / np.sum(high_amplitude)))