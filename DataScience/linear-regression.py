%matplotlib inline
import numpy as np
from pylab import *

# Hard code a linear relationship between
# page speeds and the amount of money spent
# on a website. These should be inversely correlated
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

# Render a scatterplot for this relationship
scatter(pageSpeeds, purchaseAmount)

from scipy import stats

# Run an actual linear regression model calculation using stats library
# This will return the slope (m), y intercept (c)
# The r-value, p-value (determines whether null hypothesis is valid)
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

# The null hypothesis is rejected if p < 0.05. 
# This would mean that there is no statistically significant 
# relationship between the two variables that are being measured

import matplotlib.pyplot as plt

def predict(x):
    # Calculate y = mx + c from given variables
    return slope * x + intercept

fitLine = predict(pageSpeeds)

# Plot the scatter
plt.scatter(pageSpeeds, purchaseAmount)
# Plot the resulting fit line from the linear regression 
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()

# Exercise
# Increase the random variation in the data and observe the impact
# it has on the r-squared error value.

# Mean of 3.0
# Standard deviation of 10 (was 1)
# Sample size of 1000
pageSpeeds = np.random.normal(3.0, 10, 1000)
# Mean of 0
# Standard deviation of 5 (was 0.1)
# Sample size of 1000
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 5, 1000)) * 3

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

fitLine = predict(pageSpeeds)

# Plot the scatter
plt.scatter(pageSpeeds, purchaseAmount)
# Plot the resulting fit line from the linear regression 
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()

# The resulting r-value from the modified values is lower
# Which indicates that less of the variance is encompassed
# by the line of best fit.
print(r_value**2)
print(std_err)

