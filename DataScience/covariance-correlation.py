# Exercise

# Work out if there is a relationship between page load times
# and the amount spent
#
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

# Make the purchase amount a function of pageSpeed
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

# Find the covariance
np.cov(pageSpeeds, purchaseAmount)

# Find the correlation coefficient
np.corrcoef(pageSpeeds, purchaseAmount)

