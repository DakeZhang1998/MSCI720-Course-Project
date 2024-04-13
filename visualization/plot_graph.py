import pickle
import numpy as np
import matplotlib.pyplot as plt


algorithms = [
    'Popular',
    'ImplicitMF',
    'IIIm_120_15_0001',
    'UUIm_30_2_001',
]

data = []
means = []
stds = []

for algorithm in algorithms:
    means_per_algo = []
    stds_per_algo = []
    for gamma in [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
        with open(f'scores/scores_{algorithm}_gamma_{gamma}.bin', 'rb') as file:
            record = np.array(pickle.load(file))
            means_per_algo.append(np.mean(record))
            stds_per_algo.append(np.std(record, ddof=1))
    means.append(means_per_algo)
    stds.append(stds_per_algo)

# Create figure and axis
fig, ax = plt.subplots()

# Plot each set of data
for i in range(4):
    ax.plot([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            means[i], label=f'Data Point {i+1}')

# Labeling the axes
ax.set_xlabel('X value')
ax.set_ylabel('Y value')
ax.set_title('Curves Connecting Points with Error Bars')

# Adding legend
ax.legend()

# Show grid
ax.grid(True)

# Display the plot
plt.show()