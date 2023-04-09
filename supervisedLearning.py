
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