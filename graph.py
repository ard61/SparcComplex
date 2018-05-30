import os
import numpy as np
import matplotlib.pyplot as plt

import sparc as sp
from define_scenarios import *

# Load data
results_dir = 'results'
data_points = np.zeros_like(scenarios, dtype=sparc.DataPoint)
scenario_inds = [index for index in np.ndindex(scenarios.shape)]
end_index = (2, 0, 3, 0)  # The last scenario we got results for, if simulations are incomplete
for index in scenario_inds[:scenario_inds.index(end_index)]:
    filename = '{}/{}.json'.format(results_dir, repr(scenarios[index]))
    with open(filename) as file:
        data_points[index] = sp.DataPoint.from_json(file.read())


# Create 'figures' directory
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

## Plot graphs:
# Unmodulated real vs 2-ary real for constant rate 1.0, increasing Eb/N0.
line1 = plt.semilogy([ebn0s[0, j] for j in range(ebn0s.shape[1])],
                     [data_points[0, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])],
                     label='K=1, M=512')
line2 = plt.semilogy([ebn0s[0, j] for j in range(ebn0s.shape[1])],
                     [data_points[1, 0, j, 0].avg_BER for j in range(ebn0s.shape[1])],
                     label='K=2, M=256')
plt.legend()
plt.xlabel('Signal-to-noise ratio (Eb / N0)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,R=1,K=1,2.png', bbox_inches='tight')
plt.close()

# Unmodulated real vs 2-ary real for constant rate 1.5, increasing Eb/N0.
plt.figure()
line1 = plt.semilogy([ebn0s[1, j] for j in range(ebn0s.shape[1])],
                     [data_points[0, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])],
                     label='K=1, M=512')
line2 = plt.semilogy([ebn0s[1, j] for j in range(ebn0s.shape[1])],
                     [data_points[1, 1, j, 0].avg_BER for j in range(ebn0s.shape[1])],
                     label='K=2, M=256')
plt.legend()
plt.xlabel('Signal-to-noise ratio (Eb / N0)')
plt.ylabel('Average Bit Error Rate')
plt.savefig(figures_dir + '/D=1,R=1.5,K=1,2.png', bbox_inches='tight')
plt.close()
#plt.show()
