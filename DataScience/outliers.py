%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Create a normal distribution representing income with:
#
# mean of 27000
# std-dev of 15000
# 1000 data points
#
incomes = np.random.normal(27000, 15000, 1000)

# Create an outlier with an income of 1 billion
incomes = np.append(incomes, [10**9])

# Plot a histogram using matplotlib
# Set the number of bins to be 50
plt.hist(incomes, 50)
plt.show()
