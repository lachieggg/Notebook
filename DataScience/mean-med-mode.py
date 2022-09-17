%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# Create normal distribution with
# mean of 100
# standard deviation of 20.0
# 10,000 data points
incomes = np.random.normal(100.0, 20.0, 10000)

# Plot the distribution as a histogram with
# a granularity of 50
plt.hist(incomes, 50)
plt.show()

# Find the mean of the distribution
mean = np.mean(incomes)

# Find the median of the distribution
median = np.median(incomes)

# Find the mode of the distribution
mode = stats.mode(incomes)

print("Mean " + str(mean))
print("Median " + str(median))
print("Mode " + str(mode))

