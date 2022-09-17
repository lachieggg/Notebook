%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Create a normal distribution with
# mean of 100
# standard deviation of 20
# sample size of 10,000
incomes = np.random.normal(100.0, 20.0, 10000)

# Plot a histogram
plt.hist(incomes, 50)
plt.show()

# Get the standard deviation of the income distribution
incomes.std()

# Get the variance of the income distribution
incomes.var()
