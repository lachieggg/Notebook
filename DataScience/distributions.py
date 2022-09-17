%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Uniformly distrubited, equal chance of any value occurring
#
# Equal distribution between -10.0 and 10.0 as the variable
# Plotted with 10,000 data points
values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, 50)
plt.show()

# Visualizing a normal distribution
# The normal distribution is also known as a Gaussian distribution
x = np.arange(-3, 3, 0.001)
plt.plot(x, norm.pdf(x))



# Generate a custom normal distribution
# With certain parameters
#
# mu    -> mean
# sigma -> stddev
#
mu = 5.0
sigma = 2.0
num_datapoints = 10000
num_bins = 1000
values = np.random.normal(mu, sigma, num_datapoints)
plt.hist(values, num_bins)
plt.show()


# Exponential fall off probability density function
#
step = 0.001
l = 3
x = np.arange(0, l, step)
plt.plot(x, expon.pdf(x))


# Binomial
# Probability mass function
# i.e. for discrete data
#
from scipy.stats import binom

# n -> number of samples
# p -> probability of each outcome
n, p = 10, 0.5
step = 0.001
x = np.arange(0, 10, step)
plt.plot(x, binom.pmf(x, n, p))


# Poisson
# Can be skewed!
# Not the same as normal distribution
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Mean
mu = 450
step = 0.5
end = 600
x = np.arange(400, end, step)
plt.plot(x, poisson.pmf(x, mu))

# The equivalent of a PDF for using discrete data is:
#
# A probability mass function
