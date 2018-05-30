import os
import sys
sys.path.append('/home/zelda/ard61/project/code')

import numpy as np
import sparc

import simulate
from define_scenarios import *

scenario_inds = [index for index in np.ndindex(scenarios.shape)]

# Initialise data points list
data_points = np.zeros_like(scenarios, dtype=sparc.DataPoint)
for index in scenario_inds:
    data_points[index] = sparc.DataPoint.none(scenarios[index])

# Create 'results' directory
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Gather data
start_index = (0, 0, 0, 0)  # Change this to resume from partially-interrupted simulations

max_trials = 400  # For robust results at high SNR, will need to be much larger
num_trials_increment = 400
for index in scenario_inds[scenario_inds.index(start_index):]:
    print('Scenario index: {}'.format(index))
    print(scenarios[index])

    # Keep repeating trials until we have good enough estimates, or we hit max_trials trials.
    while (data_points[index].num_trials < max_trials   # Definitely stop if we hit max_trials trials
        and (data_points[index].avg_BER == 0   # Otherwise, keep going if still at 0 trials
            or data_points[index].stddev_BER / np.sqrt(data_points[index].num_trials) > data_points[index].avg_BER / 4)):  # or if estimated standard deviation of average BER estimate is more than 25%
        print("trial {}".format(data_points[index].num_trials))

        # Run the additional trials
        data_points[index] = sparc.DataPoint.combine([
            data_points[index],
            simulate.simulate(scenarios[index], num_trials_increment)
        ])

    # Display statistics
    print(data_points[index])

    # Save the data in a file.
    filename = '{}/{}.json'.format(results_dir, repr(scenarios[index]))
    with open(filename, 'w') as file:
        file.write(data_points[index].to_json())
