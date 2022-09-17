%matplotlib inline

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Linear x
x = np.arange(-3, 3, 0.01)

# Linear graph
plt.plot(x, x)

# Plot x versus the normal distribution of X
#
# Where the normal distribution of X is a function of x
#
# In other words create a data set that is normally
# distributed over -3 to 3
plt.plot(x, norm.pdf(x))
plt.show()

# Plot the same graph as before
plt.plot(x, norm.pdf(x))
# Plot a graph that has a different mean and std-deviation
mu = 1.0
stddev = 0.5
plt.plot(x, norm.pdf(x, mu, stddev))
plt.show()

# Saving to files
import os
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.savefig(os.getcwd() + '/MyPlot.png', format='png')


# Adjusting the axes
#
# Plot them
axes = plt.axes()
# Set the range for the x axes
axes.set_xlim([-5, 5])
# Set the range for the y axes
axes.set_ylim([0, 1.0])
# Set the tick marks (can skip some if you want to)
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# Set the y ticks, i.e. the markings and scale for the axes
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# Plot the normal PDF versus x
plt.plot(x, norm.pdf(x))
# Plot the normal PDF versus x for different values of mu and sigma
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()


# Adding grid lines
axes.grid()
# Re-plot
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# Change the style of the line
# The dash means "Solid line"
# b means blue
plt.plot(x, norm.pdf(x), 'b-')
# The colon means plot with little vertical hashes
# r means red
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')
plt.show()


# Adding labels and a legend
#
# Add labels to both axes
plt.xlabel('Greebles')
plt.ylabel('Probability')
# Same as before
plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')
# Add a legend in
plt.legend(['Sneetches', 'Gacks'], loc=4)
plt.show()


# Pie chart
values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]
labels = ['India', 'United States', 'Russia', 'China', 'Europe']
plt.pie(values, colors= colors, labels=labels, explode = explode)
plt.title('Student Locations')
plt.show()

# Bar chart
values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
plt.bar(range(0,5), values, color= colors)
plt.show()

# Scatter plot
#
# Two random distributions
X = randn(500)
Y = randn(500)
# Plot both X and Y
plt.scatter(X,Y)
plt.show()

# Histogram
mu = 27000
sigma = 15000
n = 10000
incomes = np.random.normal(mu, sigma, n)
plt.hist(incomes, 50)
plt.show()

# Box and whisker plot
#
uniformSkewed = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100
# Brings all the data sets together
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
# Plot the combined data set
plt.boxplot(data)
plt.show()


# Exercise
# Create a scatterplot from your own custom data
#
# Age vs time spent watching TV
#
from pylab import randn
from random import randint

age = [x for x in range(100)]

time_spent = np.random.rand(100)

time_spent = []
for x in range(100):
    time_spent.append(randint(0,x))
print(time_spent)

plt.scatter(age, time_spent)
