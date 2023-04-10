
"""
* sdf
! sdf
TODO
FIXME
//

"""

"""
* Supervised learning is used whenever we want to predict a
* certain outcome from a given input, and we examples of
* input/output pair.

* We build a MLM from these input/output pairs, which comprise
* our training set. Our goal is to make accurate predictions
* for new, never-before-seen data.

* Supervised learning often requires human effort to build the 
* training set, but afterward automates and often speeds up a
* very taxing task.

"""
"""
* There are two types of supervised ML problems:
*     - classfication
*     - regression

* Classification - goal is to predict class label.
!Example: yes/no question; language of a website

* Regression - goal is to predict a continuous number.
! Example: predicting person's income
"""

"""
* When building a model, we always want simple over complex.
* This helps prevent overfitting.
* But we don't want it to be too simple or it underfits.
! We need a sweet spot.
"""

"""
Complexity of model is tied to the variation of inputs.
The larger variety of data points, the more complex a model
you can use without overfitting.
"""

# * ALGORITHMS

# generate dataset for two-class classification
import matplotlib.pyplot as plt
import mglearn
import numpy as np
X, y = mglearn.datasets.make_forge()

# plot dataset
mglearn.discrete_scatter(X[:,0],
                         X[:,1],
                         y)
plt.legend(['Class 0', 'Class 1'], loc = 4)
plt.xlabel('First Feature')
plt.ylabel('Second feature')
print('X.shape:', X.shape)

# generate dataset for regression
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print('cancer.keys():\n', cancer.keys())
print('Shape of cancer data:', cancer.data.shape)
print('Sample counts per class:\n',
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print('Feature names:\n', cancer.feature_names)

from sklearn.datasets import load_boston
boston = load_boston()
print('Data shape:', boston.data.shape)

X, y = mglearn.datasets.load_extended_boston()
print('X.shape:', X.shape)

mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)
mglearn.plots.plot_knn_classification(n_neighbors=7)

# * use new data, using methods from Chapter 1 to split out data
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# * now fit the classifier using the training set
clf.fit(X_train, y_train)

# * to make a prediction on the test data, we call the predict()
# * method. For each data point in the test set, this computes
# * its nearest neighbors in the training set and finds
#* the common class among these

print('Test set predictions:', clf.predict(X_test))

#* to evaluate how well model generalizes, we can call score()
#* method
print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

#* we are able to visualize the decision boundary for each k

fig, axes = plt.subplots(1, 3, figsize = (10,3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    #* the fit() method returns the object self, so we can
    #* instantiate and fit in one line
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5,
                                    ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title('{} neighbor(s)'.format(n_neighbors))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend(loc=3)

#* trying this on breast cancer data
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify=cancer.target,
                                                    random_state=66)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # build model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record test generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

