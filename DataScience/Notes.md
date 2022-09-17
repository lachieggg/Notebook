

Pandas
=
Pandas is a Python library for importing data from files and cleaning it up and manipulating it.

To learn about Pandas, open up:

```PandasTutorial.ipynb```


Statistics
=

**Mean** - the average, the sum of the possible outcomes divided by the number of outcomes

**Median** - the middle number, where 50% of the values are above and 50% below

**Mode** - the most common number

Probability
=

**Probability density function** - for continuous data, take the integral to find the probability of a random outcome being between two values.

**Probability mass function** - for discrete data.


Moments
=
Moments are ways to measure the shape of the distribution

The first moment is the *mean*.

The second moment is the *variance*.

The third moment is how lopsided the distribution is. It is called *skew*. 

The fourth moment is about how accentuated the peak of the distribution is. This value is known as *kurtosis*.

Covariance
=

Covariance measures how two variables vary together from their means.

Measuring covariance is done by first converting the data into two variables as high dimensional vectors.

Small covariance, close to 0, means that there isn't much correlation between the two variables.

Large covariance, that is, the absolute difference from 0 is large, indicates that there is a correlation. But what is "large"?

Correlation
=

Correlation is the covariance divided by the product of the standard deviations of the variables. This standardizes the covariance and gives us a more accurate picture of how the two variables behave. 

Correlation of 0, means that there is no relationship between the two variables.

A correlation of 1 is a perfect correlation, and -1 is a perfect inverse correlation. 

Correlation does not imply causation.

You cannot say anything about causation without running an experiment, but correlation will tell you what kind of experiments you might want to run.


Variables
=

In an experiment, there are the independent and dependent variables. The independent variables are the variables that you are varying. The dependent variable is the variable that you are measuring.

For instance, consider an experiment measuring the relationship between maths scores and room temperature.

The maths score would be the dependent variable, the room temperature is the independent variable or the variable that is being modified and allowed to vary. 

Conditional Probability
=

$Pr(B|A) = \dfrac{Pr(A,B)}{Pr(A)}$

$Pr(A,B) = Pr(B|A)*Pr(A)$

This is intuitive, since if $A$ has already occurred, the sample space is limited to situations where $A$ occurs. Therefore we should divide by the probability that $A$ occurs which *increases* the value of the fraction.

$Pr(A,B) = P(A)*P(B) \Longleftrightarrow A$ and $B$ are independent.

$Pr(A|B) = Pr(A)$ if $A$ and $B$ are independent

**Derivation**

$Pr(A|B) = \dfrac{Pr(A,B)}{Pr(B)}$

$=> Pr(A|B) = \dfrac{Pr(A)*Pr(B)}{Pr(B)}$ when $A$ and $B$ are independent

$=> Pr(A|B) = Pr(A)$

Another probability law:

$P(A|B) + P(¬A|B) = 1$

Bayes Theorem
=

$Pr(A|B) = \dfrac{Pr(A)Pr(B|A)}{Pr(B)}$

Event $A$ $->$ Person $X$ is a user of a drug $Y$

Event $B$ $->$ Person $X$ tests positive to drug $Y$

$P(B|A) = 0.99$

$P(¬B|¬A) = 0.99$

$P(A) = 0.003$

$P(B) = P(B,A) + P(B, ¬A)$

$= P(B|A)*P(A) + P(B|¬A)*P(¬A)$

$= 0.99 * 0.003 + P(B|¬A)*0.097$

$= 0.99 * 0.003 + (1-P(¬B|¬A))*0.097$

$= 0.99 * 0.003 + 0.01 * 0.097$

$= 0.013$

**Applying Bayes theorem**: 

$P(A|B) = \dfrac{P(A)*P(B|A)}{P(B)}$

$=\dfrac{0.003*0.99}{0.013}$

$= 0.228 $

In summary, from the above, it is clear that the probability that an event $A$ occurs given an event $B$ has occurred is not the same as the converse, namely the probability that an event $B$ occurs given an event A has occurred. 

This can lead to very misleading results. In this case, the probability that a person is a user of a drug given that they tested positively for it is still only 22.8%. Phrased another way, this means that even if someone tests positively for the drug it is still more likely that they are not a user of that drug. 

We can see from this that the probabilities of $A |B$ and $B|A$ depend on the base probabilities. If the base probability of $B$ or $A$ is relatively low, this can impact the difference in the outcome of the probabilities of $A|B$ versus $B|A$ 

Linear Regression
=

Linear Regression is just fitting a line to a set of observations.

We can use this to predict unobserved values.

The name "linear regression" is misleading, it doesn't necessarily have anything to do with time or backtracking. **It is just fitting a line to a set of data points.**

Linear regression uses the least squares algorithm.

It is a model in which we minimize the sum of squared errors from a line. That is, we compute $m$ and $c$ in the equation $y = mx + c$, for which the sum of squared differences from the data points to $y$ is minimized.

It is also known as the "maximum likelihood estimation". 

**Measuring error with r-squared**

How well does the line computed by the linear regression fit the data?

R-squared will vary from 0 to 1.

0 means that no variance in the model is captured, 1 means all of the variance is captured.

Polynomial Regression
=

Important to note is that in polynomial regression, we want to avoid overfitting or underfitting to our data.

That is, we want to find a value of n, for which our polynomial is of degree n, that fits our data sufficiently but does not overvalue outliers in its shape.


Multiple Regression
=

When developing a model for predicting the price of a car, there are multiple variables that you would want to consider.

This is an instance in which we would use multiple regression, that is, regression across multiple variables.

Multiple regression is therefore when the outcome we are calculating depends on multiple input variables or independent variables.

For instance:

$price = \alpha + \beta_{1}*mileage + \beta_{2}*age + \beta_{3}*doors$


Multivariate Regression
=

Multivariate Regression is when we are predicting multiple outcomes, for instance, not just the price of a car, but perhaps estimate the time it will run for before needing a new engine.

In this case there could be several input variables as well as several output variables as well.

Articulated concisely, this is basically just having multiple dependent variables.

Multi-Level Models
=

From the lecture, some things happen at various different levels in a hierarchy.

Your health, is a function of how healthy the cells in your body are. The health of your cells is a function of how healthy the organs they are contained within are. The health of your organs are a function of how healthy your body is, which is in turn a function of the state of the society in which you live.

Predicting SAT scores for instance, might be based on the home environment of the child, how educated their parents are, how much money their parents were willing to invest in their education, whether they were available to tutor their kids in the course, amongst many other factors.

Machine Learning
=

From the lecture, machine learning is the study and implementation of algorithms that can learn from observational data, and can make predictions on it.

**Unsupervised Learning**

The model is not given any "answers" to learn from, it just tries to make sense of the data.

An example of something an unsupervised machine learning algorithm could be useful for, is clustering images into sets without telling the algorithm what characteristics the output should have or what should distinguish one image from being in one group versus another.

If you don't know what you are looking for, then unsupervised learning can bring out the "latent variable", that is, an underlying variable or connection between inputs that is not intuitive.

An example from the lectures is clustering users on a dating site based on their information and behaviour. Maybe there are groups of people that emerge that don't conform to your known stereotypes.

**Supervised Learning**

In supervised learning, the data the algorithm learns from comes with the correct answers.

Then we are actually learning from that data to be able to predict the outcomes of certain input data.

You can train your model against 80% of your data, and then test your model against the remaining 20% to see how the model performs. 

It is important to ensure that the sets you are using are large enough to check for overfitting of outliers in testing. That is, there should be enough variability in the test and train sets to protect against accidentally overfitting.

**K-fold Cross Validation**

From the lectures:

- Split data into $K$ randomly assigned segments
- Reserve one segment as your test data
- Train on each of the remaining $K-1$ segments and measure their performance
- Take the average of the $K-1$ r-squared scores

**Bayesian Methods**

Recall:

$Pr(A|B) = \dfrac{Pr(A)Pr(B|A)}{Pr(B)}$

$Pr(A) = Pr(A|B)Pr(B) + Pr(A|¬B)Pr(¬B)$

**Naive Bayes**

$Pr(Spam|Free) = \dfrac{Pr(Spam)Pr(Free|Spam)}{Pr(Free)}$

The above equation is the probability that an email is spam given that it contains the word free.

Naive Bayes just takes a set of these probabilities for all the words, and takes the product of all of them. 

For an example of spam classification algorithm, i.e. an algorithm that determines the probability of whether or not a given email is spam or not.

Take an individual word in the email, for instance, the word "free", and use Bayes theorem to determine the probability that the email is spam given that it contains that word. Obviously we need a training data set of emails, and then we can work out fairly quickly what the probability is by sampling our data.

Then we can take all the different words in an email, and multiply the probabilities together to get an outcome probability. This is the probability that the whole email is spam. 

It is called "Naive Bayes" because the probability of each word occurring is not necessarily independent of one another. That is, given two words $A$ and $B$, the probability that $A$ is in any email is not necessarily independent on the probability that $B$ is in any email. In fact, in many cases they are not independent. 

Nevertheless this is a technique that seems to at least have some utility, considering the fact that it is fairly easy to implement and does basic spam classification fairly well.


**K-Means Clustering**

Unsupervised Machine Learning technique that splits data into K groups.

For instance, given a scatter plot of data, we can group that data into 3 groups, based on how close certain data points are to one another.

Centroids are at the center of each of the data sets that we split the group up into. We measure whether a data point is in a group based
on which centroid it is closest to.

Algorithm:
- Pick K randomly positioned centroids
- Assign all data points to their respective centroid
- Find new centroids based on the average of each current cluster
- Repeat this process until each point converges on one centroid.

This is essentially just trying to find the right centroids through iteration. 

Notes:

*Choosing K* 

Increase K until your sum of squared errors converges or stops being reduced significantly across iterations.

*Avoiding local minima*

Since our initial choice of centroids is random, we want to make sure that the initial choice did not produce a result that was starkly different to the rest.

Therefore, we should iterate a few times with different values of our centroid in order to prevent "weird" outcomes.

*Labelling clusters*

K-Means classifies data to a limited extent. It does not give us meaning to the data. It's up to the data scientist to work out what the meaning is behind the clustering. 


**Entropy**

Entropy is a measure of a data set's disorder. If a data set of integers in an array consists of all 1s, then that data set will have an entropy of 0.

If everything in the data set is of a particular type or class, then that contributes no entropy to the set. If nothing in the data set is of that particular type or class, then that also contributes nothing to the entropy, because there are no instances of that class in the dataset.


**Decision Trees**

A decision tree is a form of supervised learning that outputs a decision tree based on input data and classifications of that data. For example, given a set of data that has a set of weather conditions and whether or not I went out to play in the park on a given day, we can generate a deicison tree algorithm that will determine, for any given future data, whether or not I am likely to go outside and play on that day.

Another example would be writing a program that will filter out resumes for a given job application, based on historical data.

ID3 is the algorithm for generating decision trees. At each step it minimizes the entropy of the data at the next step.

*Random Forests (Bootstrap aggregating)*

In order to prevent overfitting in our model, we can construct several decision trees from different random subsets of our dataset and attribute set, and then get each tree to "vote" on the best tree based on the outcomes, which as a set give a kind of set of probabilities for the best tree.

**Ensemble Learning**

Random forests are an example of *ensemble learning*.

It is where you use multiple models at once and let them vote on the results.

Boostrap aggregating is where we pick subsets of the data and vote on the best model.

Boosting is a technique where certain attributes that are misclassified by the previous model is made more important, so that future models will "pay more attention" to these attributes. In other words, we find the weaknesses in the model and try to rectify that by boosting the importance of an attribute.

A bucket of models trains different models and then picks the one that works best with the data.

Whereas stacking runs several models at once, and then uses each model to vote on the outcome. 

**XGBoost**

eXtreme Gradient Boosted trees.

Boosting is an ensemble method.

Creates a set of trees, each of which boosts the attributes that led to incorrect classifications of the previous tree.

XGBoost has several useful features

- Overfit prevention
- Missing value imputation
- Parallel processing (threading) and runs on clusters
- Cross Validation (finding the optimal number of iterations)
- Incremental training (stop/start the training process)
- Plug in custom variables to optimize for 
- Tree pruning (generating optimized trees)

*Hyperparameters*

We can choose various "hyperparameters" for XGBoost to get the best reuslts.

Booster: `gbtree` or `gblinear` 

Objective: `multi:softmax` for finding the best classification, or `multi:softprob` for set of proabilities.

Eta: Learning rate, adjusts the weights at each iteration.

Depth: Maximum depth of the tree. Too small and it will be inaccurate, too high and we will overfit our model.

Min Child Weight: Controls overfitting, if set too high it will underfit.

XGBoost is (almost) all that you need to know for basic ML problems in practical terms for simple classification or regression problems.

**Support Vector Machines**

Supervised learning technique that is helpful for classifying higher dimensional data with lots of attributes or features.

Mathematically complicated, but essentially finds "higher dimensional support vectors across which to divide the data". These support vectors are mathematically known as "hyperplanes".

Implements something known as the "kernel trick".

Despite doing clustering, just like K-Means, it is a supervised technique. 

*Support Vector Classification*

You can use different kernels to project down from these hyperdimensions into 2 dimensions, to essentially simplify the data from several dimensions and show it in $R^2$ 

For example, we can have a linear kernel, that breaks up our two dimensions using linear functions. We can also have SVC with a polynomial function of degree $n$. The example of the lecture slides is a kernel with polynomial of degree 3.

Recommender Systems
=

Recommender systems are used all over the Internet. The most well known examples include companies like Amazon and Netflix, that recommend new items based on your purchase and watch history respectively. These systems help users discover things that the system believes they might be interested in by using a variety of techniques in data science and machine learning depending on the application. 

**User based collaborative filtering**

A fancy term for just taking in behaviour across all user's, looking at similarities between users, and then making predictions about what user X might click on given that user X shares similar search history to user Y. In other words, making predictions about a user based on a combination of what you like and what everybody else likes.

Essentially we build up a matrix of content/products/websites/people that each user had an interaction with, which could be represented as a boolean value, with the user as the row, and the "interacted with" item as the column.

Then we can use this matrix to compute the similarity between two users. 

As an interesting idea, potentially this relationship between users could be represented as a "similarity coefficient" as a number from 0 to 1. And then we could create a set of similarities between users. In this case, if we had n users, then there would be n-1 "similarity coefficients" between each user. Since there are n users, and there only needs to be one coefficient for two users, there would be $n*(n-1)/2$ total coefficients for the set of users. So the "coefficient" matrix would scale at $O(n^2)$ in space used relative to the number of users. This is quite large if we have a lot of users. Perhaps we could take a random sample of users in practice instead of iterating over every single other user, which would keep the size of this "similarity coefficient matrix" down.


*Problems*

- Fickleness, people's tastes change over time
- More people than things, focusing on users can be expensive. 
- Shilling attacks, fake likes or clicks

**Item Based Collaborative Filtering**

Instead of basing recommendations on whether or not user X is similar to user Y and finding things user X might like based on what user Y likes, we can instead find relationships between items. So now we can say, user X bought item A, and item A is similar to item B, so user X might also like item B. 

This results in less computation, because there are less relationships to manage.

Items are not fickle, they do not change.

The process is something like, find every pair of movies that were watched by the same person, then measure the similarity of their ratings (out of 5 stars) over all users who watched both movies, then sort by the similarity of average rating strength to find a ranked set of similar movies. 

[Movielens](https://movielens.org/) is a dataset of movies and ratings of those movies to find recommendations of new movies that you might like. 

**Cosine Similarty**

For this example, we are just rating Star Wars.
 
So we are finding correlations between Star Wars and other movies.

We will be taking an average of the correlation between ratings for 
every user

That is, for one pair, we would be taking the ratings for that movie, and finding all of the places in the dataset where a user has rated both movies, then taking the correlation for each of those
i.e. 1 and 5 would be low correlation, 3 and 4 would be high, 1 and 1 would be perfect.

Then take all of those coefficients and take the mean, i.e. sum them all up and divide by the number of terms.

Then do that process for every other movie for which there is at least one user that has a rating for both.

In practical terms we may want more than just one rating to make sure our sample size is large enough for it to be representative or useful as a prediction mechanism.

So for movies that are quite obscure, we will need to remove them from the dataset, because if someone has rated Star Wars and one obscure movie both 5 stars, and they are the only person with that pairwise connection, then that will skew our data, because it will give a perfect correlation. So for this example we will be removing all movies that do not have at least N=100 ratings, in order to make sure that our sample size is sufficient. 

**K-Nearest Neighbour**

Given a scatterplot, we can compute the distance between any two points.
K-Nearest neighbours is, given a new point that we want to classify, and a set of points that we have already classified, then we can classify that new data point by computing the K-Nearest Neighbours of that point, and then finding the highest number of occurrences for a class in that set. 

**Principle Component Analysis (PCA)**

Dimensionality reduction is the attempt to distill higher dimensional sinformation into lower dimensional information.

K-means clustering is an example of a dimensionality reduction algorithm. It reduces data down to K dimensions, since the data becomes merely a set of distances from each centroid. 

PCA finds "hyperplanes" which we project our data on in our lower dimensional data space. We do this in a way that preserves most of the variance in the data. These hyperplanes are defined by eigenvectors of the data. 

In the context of the Iris dataset, we can take 4 dimensional data, which in our case is the length and width of the sepal and petal respectively. We can project this 4D data down into a 2D data space using this technique. 

A common application of PCA is image compression, which allows us to preserve a lot of the information in an image while also reducing the size stored on disk.

With the Iris dataset, it is possible to compress, flatten or transform the 4D data down to a 2D data space. This works well because the different variables are likely highly correlated, that is, it is possible that one variable can be formed quite easily out of two other variables. In this instance, we can mostly just measure the overall size of the whole flower instead of having to measure the petal and sepal widths, and PCA has "distilled" the dataset down into a simpler form using this correlation.

**Data Warehousing**

A data warehouse is essentially just a big database that takes in data from a variety of different sources. It is commonly used for analysis in a business.

Data normalization and cleaning can be a difficult problem when working with a lot of data. 

**ETL: Extract, Transform, Load**

Raw data is first extracted from the system.

The data is then transformed into the database schema that is required by the data warehouse.

The data is then loaded into the warehouse. 

The problem becomes this transform step, when dealing with "big data", this can be an expensive operation. 

**ELT: Extract, Load, Transform**

The idea behind ELT is to instead extract the data, load it straight into our data warehouse, and then use the power of distributed computing and software to transform the data "in place".

This takes advantage of a NoSQL database, where we do not need to worry about schema anymore and transforming the data before it hits our system. 

*Examples:* Hadoop, Hive, SparkSQL, Mapreduce.

Data warehousing is an entire discipline in and of itself.

**Reinforcement Learning**
---
A form of Machine Learning that operates by 'reinforcing' certain behaviours or actions by giving them a higher score than other less desirable actions.

*Q Learning* is a particular form of reinforcement learning. In Q-Learning, we have a set of states $s$, a set of actions $a$, and a value of each state and action $Q$. 

The algorithm is as follows. Begin with $Q$ values of 0. Explore the environment. If something bad happens, reduce the $Q$ value for the previous actions. If something good happens, increase the $Q$ value for the previous action or set of actions. Reinforcement learning is analogous to classical conditioning in Psychology (Pavlovian conditioning).


How should we explore the environment? We could always choose the best Q value and use that to explore states. But this is quite a rigid approach and could end up missing the global optima, since it is a greedy approach. How do we discover all possible states if we only follow a greedy approach of following the highest Q value? Similarly to hill climbing, this greedy approach may land us at a local optima  as opposed to a global one. 

The answer to this is to use randomness. We introduce an epsilon term $\epsilon$ and effectively roll a dice to determine which path to follow. Now the new question becomes, how do we pick our $\epsilon$, also known as our exploration rate.

Q-Learning is a kind of *Markov Decision Process* and it is also an application of *Dynamic Programming*, since we are storing the solutions to subproblems in our exploration of the environment to find the optimal set of weights for each of the possible action/state pairings. 

**Confusion Matrices**
---

                  |  Actual YES    |  Actual NO
    -----------------------------------------------
    Predicted Yes |  True Positive    False Positive
    Predicted No  |  False Negative   True Negative

A confusion matrix is essentially just a breakdown for the accuracy of a model across different input data.

Confusion matrices can sometimes be represented as heat maps with the colour mapping to a value. The confusion matrix can also represent more than just yes and no outcomes, like "I got a 1 but expected an 8" for instance. We would expect a dark line through the middle of the matrix in this case.

**Measuring Classifiers**

*Recall*:

Let $TP$ be the number of true positives in a test data set. Let $TN$ be the number of true negatives. Let $FP$ be the number of false positives. Let $FN$ be the number of false negatives.

Recall is defined as
$\dfrac{TP} {TP + FN} $

Also known as Sensitivity, True Positive rate, completeness. It is a good choice of metric when you care a lot about false negatives, like fraud detection (want to be leaning toward false positives in that case).

*Precision*:

$\dfrac{TP} {TP + FP}$

Also known as Correct Positives. It is a good choice of metric when you care a lot about false positives, like in drug testing. 

*Specificity*:

$\dfrac{TN} {TN + FP}$

*F1 score*:

$2*\dfrac {Precision*Recall} {Precision+Recall}$

For when you care about precision and recall.

**ROC Curve (Receiver Operating Characteristic Curve)**

Plot of the true positive rate (recall) vs. false positive rate at various threshold settings. The points above the diagonal represent good classification, since they are better than random, because random in a boolean setting is just an accuracy of 0.5.

How bent the curve is towards the upper left corner is an indication of how good the model is. A perfect curve would just be a point in the upper left corner, where there is a true positive rate of 1 and a false positive rate of 0.

**AUC (Area under the ROC curve)**

The area under the curve is equal to the probability that the model (classifier) will rank a randomly chosen positive instance higher than a randomly chosen negative one. An AUC of 1.0 is perfect, because it will rank a positive instance higher than a negative one always, and an AUC of 0.5 is useless, because in that case we might as well be flipping a coin and using that as our model. An AUC of less than 0.5 is worse than useless because it is more likely to predict the wrong outcome.

As previously stated, the ROC curve with of $y=x$ refers to a useless classifier since the average true positive is just as common as the true negative for different inputs.

Real World Data
=

Bias / Variance Tradeoff
---

Taking the mean of the predicitions, are they all close to the mean of the expected answers? 

The bias is a measure of how close the mean of the predicted values is to the expected outcome. Bias is akin to skew.

The variance is a measure of how scattered the individual values are from the expected outcome. Variance is akin to probabilistic variance. 

That is, the bias will take the mean of the outcomes, and determine if that is close to the expected outcome. And the variance will just take the sum of squared errors from the expected outcome, instead of averaging all the outcomes out first.

In the context of a dartboard, with the darts representing the outcomes and the bullseye representing the expected outcome.

Low bias and high variance occurs in the case that the darts are scattered around the bullseye, where the average position of all darts combined is on the bullseye but no individual dart hits it.

High bias and low variance would occur in the case that all the outcomes occur in the same area, but are far away from the expected outcome.

High bias and high variance would be where average position when summing the positions of all the darts is far away from the bullseye, but the sum of the variance of each dart is also high.

Low bias and low variance would occur when the darts are on the bullseye.

Essentially variance is a measure of how consistent we are with our predictions. Bias is a measure of how accurate we are to the expected outcome. 

Sometimes in the real world you need to choose between bias and variance. This is akin to overfitting or underfitting your data.

Let $B = Bias$, $V = Variance$ and $E = Error$

Then $E = B^2 + V$ 

Error is what we wish to minimize, not $B$ or $V$.

A model that is too complicated will have a high variance and a low bias (overfit).

A model that is too simple will have a low variance and a high bias.

