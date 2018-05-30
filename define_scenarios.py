import numpy as np
import sparc

# Generate loads of different SPARC parameters, that are all comparable. 

# Parameters common to all simulations
R_per_Ds = np.array([1, 1.5])               # Rate per dimension R/D
R_pa_per_Ds = np.array([0, 1.05]) * R_per_Ds      # Power allocation rate per dimension R_pa/D
L = 1024
logK = 0
logKMs = np.array([9])

# Channel powers at which we will test the SPARCs for each rate
ebn0s = np.stack([
    np.arange(2, 6.1, 0.5),
    np.arange(4, 8.1, 0.5),
])

# All the parameters we want to test
p_K1D1 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)
p_K2D1 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)
p_K4D1 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)
p_K4D2 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)
p_K8D2 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)
p_K16D2 = np.zeros((R_per_Ds.size, ebn0s.shape[1], logKMs.size), dtype=sparc.SparcParams)

for i in range(R_per_Ds.size):
    for j in range(ebn0s.shape[1]):
        for k in range(logKMs.size):
            R_per_D = R_per_Ds[i]
            R_pa_per_D = R_pa_per_Ds[i]
            ebn0 = ebn0s[i,j]
            logKM = logKMs[k]

            Pchan_per_D = sparc.ebn0_to_Pchan(ebn0, R_per_D)
            Palloc_per_D = sparc.iterative_power_alloc(Pchan_per_D, L, R_pa_per_D)

            # Unmodulated real
            D = 1
            logK = 0
            logM = logKM - logK
            p_K1D1[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)

            # 2-ary real
            logK = 1
            logM = logKM - logK
            p_K2D1[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)

            # 4-ary real
            logK = 2
            logM = logKM - logK
            p_K4D1[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)

            # 4-ary complex
            D = 2
            logK = 2
            logM = logKM - logK
            p_K4D2[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)

            # 8-ary complex
            logK = 3
            logM = logKM - logK
            p_K8D2[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)

            # 16-ary complex
            logK = 4
            logM = logKM - logK
            p_K16D2[i,j,k] = sparc.SparcParams.from_rate(R_per_D * D, D, L, logK, logM, Palloc_per_D * D)


            # Check they're all comparable!
            assert(p_K1D1[i,j,k].comparable_with(p_K2D1[i,j,k]))
            assert(p_K1D1[i,j,k].comparable_with(p_K4D1[i,j,k]))
            assert(p_K1D1[i,j,k].comparable_with(p_K4D2[i,j,k]))
            assert(p_K1D1[i,j,k].comparable_with(p_K8D2[i,j,k]))
            assert(p_K1D1[i,j,k].comparable_with(p_K16D2[i,j,k]))

scenarios = np.stack([p_K1D1, p_K2D1, p_K4D1, p_K4D2, p_K8D2, p_K16D2])
