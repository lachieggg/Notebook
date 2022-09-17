%matplotlib inline

import pandas as pd
import seaborn as sns

# Fuel efficiency csv file
url = "http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv"
# Read the file in a data frame
df = pd.read_csv(url)
# Extract out the gear numbers
gear_counts = df['# Gears'].value_counts()
# Plot as bar chart
gear_counts.plot(kind='bar')
# Initialize seaborn
sns.set()
# Re-plot
gear_counts.plot(kind='bar')

# Show top of data
df.head()

# Histogram with smooth average
sns.displot(df['CombMPG'])

# Extract certain columns
df2 = df[['Cylinders', 'CityMPG', 'HwyMPG', 'CombMPG']]
df2.head()

# Plot every combination of columns on a scatterplot
# MPG stands for miles per gallon
# The graphs show that the number of cylinders is inversely correlated
# with the MPG
sns.pairplot(df2, height=2.5);

# Plot just combined MPG and engne displacement
sns.scatterplot(x="Eng Displ", y="CombMPG", data=df)
# Inverse or negative correlation

# Plot the same graph with the volume of data overlayed
sns.jointplot(x="Eng Displ", y="CombMPG", data=df)

# Scatter plot but with a linear regression applied to it
# Linear regression is basically fitting a line to the data
sns.lmplot(x="Eng Displ", y="CombMPG", data=df)

# Set the figure size
sns.set(rc={'figure.figsize':(15,5)})
# Plot the boxplot itself
# Manufacturer on X axis
# Combined MPG on the Y axis
# Set the X axis labels, keeping them unchanged, and add a slant
ax=sns.boxplot(x='Mfr Name', y='CombMPG', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# Box and whisker plot:
# Whisker ends are the minimum and maximum values in the distribution
# However these ends exclude outliers
#
# The boxes are Q1 to Q3 of the data
# The box length is the inter-quartile range (IQR)
#
# The median is the middle value, which is the centre
# of the boxplot, also known as Q2
#
# Q1 is the poin at which 25% of the data is below
# and 75% of the data is above
#
# Quartiles only care about the number of data points above
# and below, not how that data is distributed
#
# For instance, we could have Q1 of value 5, and all of the data
# below value 5 be of value 1, or alternatively value 4, and Q1 is unchanged
#
# Outliers are points that are outside 1.5 times the IQR on Q1 and Q3
#
# For instance:
#
# x is an outlier if x < Q1 - 1.5*IQR
# y is an outlier if x > Q3 + 1.5*IQR
# The IQR is defined as being the absolute difference in value between
# Q1 and Q3
#

# Plot a swarm plot
# Which shows the distribution of data better than a plain
# box plot
ax=sns.swarmplot(x='Mfr Name', y='CombMPG', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)


# Countplot
# Same thing as a histogram except for categorical data
ax=sns.countplot(x='Mfr Name', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
# This is just showing the number of data points for each
# make of car

# Heat map
df2 = df.pivot_table(index='Cylinders', columns='Eng Displ', values='CombMPG', aggfunc='mean')
sns.heatmap(df2)

# Exercises
#
# Scatter Plot
#
df2 = df[['# Gears', 'CombMPG']]
sns.scatterplot(x='# Gears', y='CombMPG', data=df2)

# Linear model
sns.lmplot(x="# Gears", y="CombMPG", data=df2)

# Joint plot
sns.jointplot(x="# Gears", y="CombMPG", data=df2)

# Box plot
sns.boxplot(x="# Gears", y="CombMPG", data=df2)

# Swarm plot
sns.swarmplot(x="# Gears", y="CombMPG", data=df2)

# From the graphs, we can see that
# bombined MPG correlates negatively with the number of gears.
# That is, the more gears we have, on average the car will
# have a lower number of miles per gallon.
#

