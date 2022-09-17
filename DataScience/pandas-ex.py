%matplotlib inline
import numpy as np
import pandas as pd

# Load the data file into a data frame
df = pd.read_csv("PastHires.csv")

# Show the head of the data frame
df.head()

# Show the first 10 rows of the data frame
df.head(10)

# Show the last 4 rows of the data frame
df.tail(4)

# Show the shape of the data
# i.e. number of rows and columns respectively
df.shape

# Total size of the data frame
# i.e. number of cells
df.size

# Length of the data frame
# i.e. number of rows
len(df)

# List the columns of the data frame
df.columns

# Extract a single column from the data frame
# as a "series"
df['Hired']

# Extract a given range of rows from the column
df['Hired'][:5]

# Extract a single cell from the data frame
df['Hired'][5]

# Extract more than one column
df[['Years Experience', 'Hired']]

# Extract more than one column and certain rows
df[['Years Experience', 'Hired']][:5]

# Sort your data frame by a column
# i.e. sort by highest number of "Years Experience"
df.sort_values(['Years Experience'])

# Calculate the number of unique values in a particular column
# and then list it
#
# For example, get the number of instances of each level (eg. PhD)
# and then list the instance number next to the education level
degree_counts = df['Level of Education'].value_counts()
degree_counts


# Plot the number of instances of each education level as a
# vertical bar graph
degree_counts.plot(kind='bar')

#
# Exercises
#

# Extract rows 5-10 from the data frame
# With only certain columns
x = df[['Previous employers', 'Hired']][5:11]
print(x)

# Plot them as a vertical bar graph
x.plot(kind='bar')
