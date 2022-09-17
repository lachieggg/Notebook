%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

# Create a normal distribution with
# mean of 0
# std-dev of 0.5
# 10,000 data points
vals = np.random.normal(0, 0.5, 10000)

# Plot a histogram
# with a granularity of 500
plt.hist(vals, 500)
plt.show()

# Mean is not influenced by changing the std-dev
np.mean(vals)

# Variance is not influenced by changing the mean
# but is influenced by changing the std-dev
np.var(vals)


# Skew is not influenced by changing the mean
# nor influenced by changing the std-dev
sp.skew(vals)

# Kurtosis is not influenced by changing the mean,
# and not influenced by changing the std-dev
sp.kurtosis(vals)

# Note
#
# Percentile: A number of which a given percent of observations fall below
# For instance, 95th percentile of IQ is the top 5% of IQ scores
# 95th percentile in IQ is 125, which means that only 5% of people
# have an IQ score of above 125.
