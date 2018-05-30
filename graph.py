import os
import numpy as np
import matplotlib.pyplot as plt

import sparc as sp
from define_scenarios import *

## Load data
results_dir = 'results'

data_points = np.zeros_like(scenarios, dtype=sparc.DataPoint)
scenario_inds = [index for index in np.ndindex(scenarios.shape)]
for index in scenario_inds:
    data_points[index] = sparc.DataPoint.none(scenarios[index])

for index in scenario_inds:
    filename = '{}/{}.json'.format(results_dir, repr(scenarios[index]))
    try:
        with open(filename) as file:
            data_points[index] = sp.DataPoint.from_json(file.read())
    except FileNotFoundError:
        pass


# Create 'figures' directory
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


# Plot graphs.

###############################################################################
# Unmodulated real vs 2-ary real for constant rate 1.0, increasing Eb/N0.
###############################################################################
ax = plt.axes()

ebn0 = [ebn0s[0, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=1
data2 = [data_points[1, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=2

line1 = plt.semilogy(ebn0, data1, label='K=1, M=512')
line2 = plt.semilogy(ebn0, data2, label='K=2, M=256')

R = data_points[0, 0, 0, 0].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend()
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,R=1,K=1,2,KM=512.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 2-ary real for constant rate 1.5, increasing Eb/N0.
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[1, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=1
data2 = [data_points[1, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=2

line1 = plt.semilogy(ebn0, data1, label='K=1, M=512')
line2 = plt.semilogy(ebn0, data2, label='K=2, M=256')

R = data_points[0, 1, 0, 0].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend()
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,R=1.5,K=1,2,KM=512.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 4-ary real for constant rate 1.0, increasing Eb/N0.
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[0, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=1
data2 = [data_points[2, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # K=4

line1 = plt.semilogy(ebn0, data1, label='K=1, M=512')
line2 = plt.semilogy(ebn0, data2, label='K=4, M=128')

R = data_points[0, 0, 0, 0].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend()
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,R=1,K=1,4,KM=512.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 4-ary, 8-ary, 16-ary complex for constant rate 1.0, increasing Eb/N0.
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[0, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=1 K=1
data2 = [data_points[3, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=4
data3 = [data_points[4, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=8
data4 = [data_points[5, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=16

line1 = plt.semilogy(ebn0, data1, label='D=1, K=1, M=512')
line2 = plt.semilogy(ebn0, data2, label='D=2, K=4, M=128')
line3 = plt.semilogy(ebn0, data3, label='D=2, K=8, M=64')
line4 = plt.semilogy(ebn0, data4, label='D=2, K=16, M=32')

R = data_points[0, 0, 0, 0].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend(loc='center left')
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,2,R=1,K=1,4,8,16,KM=512.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 4-ary, 8-ary, 16-ary complex for constant rate 1.5, increasing Eb/N0.
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[1, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=1 K=1
data2 = [data_points[3, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=4
data3 = [data_points[4, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=8
data4 = [data_points[5, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=16

line1 = plt.semilogy(ebn0, data1, label='D=1, K=1, M=512')
line2 = plt.semilogy(ebn0, data2, label='D=2, K=4, M=128')
line3 = plt.semilogy(ebn0, data3, label='D=2, K=8, M=64')
line4 = plt.semilogy(ebn0, data4, label='D=2, K=16, M=32')

R = data_points[0, 1, 0, 0].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend(loc='center left')
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,2,R=1.5,K=1,4,8,16,KM=512.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 4-ary, 8-ary, 16-ary complex for constant rate 1.0, increasing Eb/N0, for KM = 1024
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[0, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 0, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=1 K=1
data2 = [data_points[3, 0, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=4
data3 = [data_points[4, 0, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=8
data4 = [data_points[5, 0, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=16

line1 = plt.semilogy(ebn0, data1, label='D=1, K=1, M=1024')
line2 = plt.semilogy(ebn0, data2, label='D=2, K=4, M=256')
line3 = plt.semilogy(ebn0, data3, label='D=2, K=8, M=128')
line4 = plt.semilogy(ebn0, data4, label='D=2, K=16, M=64')

R = data_points[0, 0, 0, 1].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend(loc='center left')
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,2,R=1,K=1,4,8,16,KM=1024.png', bbox_inches='tight')
plt.close()


###############################################################################
# Unmodulated real vs 4-ary, 8-ary, 16-ary complex for constant rate 1.5, increasing Eb/N0, for KM=1024
###############################################################################
plt.figure()
ax = plt.axes()

ebn0 = [ebn0s[1, j] for j in range(ebn0s.shape[1])]
data1 = [data_points[0, 1, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=1 K=1
data2 = [data_points[3, 1, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=4
data3 = [data_points[4, 1, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=8
data4 = [data_points[5, 1, j, 1].avg_BER for j in range(ebn0s.shape[1])]  # D=2 K=16

line1 = plt.semilogy(ebn0, data1, label='D=1, K=1, M=1024')
line2 = plt.semilogy(ebn0, data2, label='D=2, K=4, M=256')
line3 = plt.semilogy(ebn0, data3, label='D=2, K=8, M=128')
line4 = plt.semilogy(ebn0, data4, label='D=2, K=16, M=64')

R = data_points[0, 1, 0, 1].sparc_params.R
ebn0_min = sp.Pchan_to_ebn0(sp.minimum_Pchan(R), R)
plt.axvline(ebn0_min)  # Shannon capacity
capacity_text = 'Minimum SNR: %.3f dB' % ebn0_min
capacity_bbox = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

# Text box for minimum power
ax.text(0.07, 0.1, capacity_text, transform=ax.transAxes, verticalalignment='bottom', bbox=capacity_bbox)

plt.legend(loc='center left')
plt.xlabel('Signal-to-noise ratio Eb/N0 (dB)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,2,R=1.5,K=1,4,8,16,KM=1024.png', bbox_inches='tight')
plt.close()
