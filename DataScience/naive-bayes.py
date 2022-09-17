from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Naive Bayes

# Assume that 'data' is a dataframe containing a set of 
# messages and an outcome of whether they are 'spam' or 'ham'
# i.e. illegitimate or legitimate emails
#
# This will be our training data set for NB
#

# Tokenizes words into a number, where the numbers are keys to the
# word, which is more compact
# 
# Then turns each message into a list of words
#
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values) 
print(counts)

# Train our classifier using the input data
# 
# That is, get the set of probabilities over a set of words
# that is is spam. We can then use those probabilities to 
# classify future data.
#
classifier = MultinomialNB()
targets = data['class'].values
# First argument is the set of words in each email as a vector
#
# And the second is a set of targets i.e. the correct results
# that we can train our model against
classifier.fit(count, targets)

examples = [
    'Free Viagra now!!!', 
    "Hi Bob, how about a game of golf tomorrow?", 
    "Free Stuff Now Now Buy This!!"
]

# Tokenize each word and then create a 
# matrix that maps each token to a number of instances
# of each word
#
# Note that the word "Now" 
# occurs twice and this is shown in the output
#
example_counts = vectorizer.transform(examples)
print(example_counts) 
predictions = classifier.predict(example_counts)
predictions