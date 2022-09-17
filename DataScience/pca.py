# Code for PCA
#

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle

# Load in the famed Iris dataset 
iris = load_iris()

numSamples, numFeatures = iris.data.shape
# Print the data so we have an intuition about
# what we are looking at
print(numSamples)
print(numFeatures)
print(list(iris.target_names))
print(iris)

X = iris.data
# Transform or "flatten" it to two dimensions
#
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)

print(pca.components_)

# Determine how much variance we have preserved
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))