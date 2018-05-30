import numpy as np

def modified_power_alloc(a, f, P_total, L, C):
    P = np.zeros(L)
    for l in range(L):
        if l/L < f:
            P[l] = np.exp2(-2*a*C*l/L)
        else:
            P[l] = np.exp2(-2*a*C*f)
    P *= P_total / sum(P)  # Scale P so that it sums to P_total

    return P

def iterative_power_alloc(P_total, L, R):
    P = np.zeros(L)
    P_remain = P_total
    for l in range(L):
        tau_sq = 1 + P_remain
        P[l] = 2*np.log(2)*R*tau_sq/L

        if P[l] < P_remain / (L-l):
            P[l:] = P_remain / (L-l)
            break
        P_remain = P_remain - P[l]
    return P

def Pchan_to_ebn0(Pchan, R):
    return 10 * np.log10(Pchan / (2*R))

def ebn0_to_Pchan(ebn0, R):
    return 2 * R * (10 ** (ebn0 / 10))