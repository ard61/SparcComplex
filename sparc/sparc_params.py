import numpy as np
import json
from sparc.awgn_capacity import awgn_capacity

class SparcParams:
    def __init__(self, n, D, L, logK, logM, Palloc):
        assert (Palloc.shape[0] == L)

        self.n = n              # codeword length
        self.D = D              # codeword dimensionality (1 for real, 2 for complex)
        self.L = L              # #(sections)
        self.logK = logK        # log #(modulation constants in each section)
        self.logM = logM        # log #(columns per section)
        self.Palloc = Palloc    # Power allocation (power per section per transmission)

        self.compute_consts()

    def compute_consts(self):
        self.K = 2**self.logK
        self.M = 2**self.logM
        self.Pchan = np.sum(self.Palloc)                        # Average power per transmission
        self.inputbits = self.L * (self.logK + self.logM)       # Number of bits input to the block code
        self.R = self.inputbits / self.n                        # Code rate (bits per transmission)
        self.C = awgn_capacity(self.Pchan)                      # Channel capacity per transmission
        self.Ctotal = self.L * self.n * self.C                  # Total channel capacity
    
    def __str__(self):
        return "n: {}, D:{}, L:{}, K:{}, M:{}, R:{}, C:{}, Pchan: {}".format(
            self.n,
            self.D,
            self.L,
            self.K,
            self.M,
            self.R,
            self.C,
            self.Pchan
        )

    def __repr__(self):
        return "n{},D{},L{},logK{},logM{},P{}".format(
            self.n,
            self.D,
            self.L,
            self.logK,
            self.logM,
            self.Pchan
        )

    def __eq__(self, other):
        return (self.n == other.n 
            and self.D == other.D
            and self.L == other.L
            and self.logK == other.logK
            and self.logM == other.logM
            and np.all(self.Palloc == other.Palloc))

    def comparable_with(self, other):
        """
        To be comparable, two SPARCs need:
        (1) to be transmitted in the same number of real transmissions, n * D.
        (2) to code the same number of input bits, L * (logK + logM)
        (3) to have the same power constraint in each signal dimension, Pchan / D.
        """
        return (self.n * self.D == other.n * other.D            # Same number of real-valued transmissions
            and self.inputbits == other.inputbits               # Same number of input bits
            and self.Pchan / self.D == other.Pchan / other.D)   # Same power constraint per dimension

    @classmethod
    def from_rate(cls, R, D, L, logK, logM, Palloc):
        # n = int(np.round(L * (logK + logM) / R))

        # Make sure n is even if D=1!
        if D == 1:
            n = int(np.round(L * (logK + logM) / R / 2)) * 2
        else:
            n = int(np.round(L * (logK + logM) / R))
        return cls(n, D, L, logK, logM, Palloc)
    
    def to_json(self):
        return json.dumps({
            'n': int(self.n),
            'D': int(self.D),
            'L': int(self.L),
            'logK': int(self.logK),
            'logM': int(self.logM),
            'Palloc': self.Palloc.tolist()
        })

    @classmethod
    def from_json(cls, string):
        p = json.loads(string)
        return cls(p['n'], p['D'], p['L'], p['logK'], p['logM'], np.array(p['Palloc']))
