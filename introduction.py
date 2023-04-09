import numpy as np

x = np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))

from scipy import sparse
# create a 2D NumPy array with a diagonal
# of ones, and zeroes everywhere else
eye = np.eye(4)
print('NumPy array:\n', eye)

# convert the NumPy array to a SciPy sparse matrix
# in CSR format
# only nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print('\nSciPy sparse CSR matrix:\n', sparse_matrix)

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO representation:\n', eye_coo)

# matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

# generate a sequence of numbers from -10 to 10
# with 100 steps in between
x = np.linspace(-10, 10, 100)

# create a second array using sine
y = np.sin(x)

# this plot function makes a line chart of one array
# against another
plt.plot(x, y, marker = 'x')
plt.show()

# pandas
import pandas as pd

# create a simple dataset of people
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age': [24, 13, 53, 33]
        }

data_pandas = pd.DataFrame(data)
data_pandas

data_pandas[data_pandas['Age'] > 30]

# IRIS DATASET
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print('Keys of iris_dataset:\n', iris_dataset.keys())

print(iris_dataset['DESCR'][:193] + '\n...')

print('Target names:', iris_dataset['target_names'])
print('Feature names:\n', iris_dataset['feature_names'])

print('Type of data:', type(iris_dataset['data']))
print('Shape of data:', iris_dataset['data'].shape)

print('First five rows of data:\n', iris_dataset['data'][:5])

print('Type of target:', type(iris_dataset['target']))
print('Shape of target:', iris_dataset['target'].shape)
print('Target:\n', iris_dataset['target'])
# 0 = setosa, 1 = versicolor, 2 = virginica

# We want to build a ML model from this data that can
# predict the species of iris for a new set of
# measurements. But before we can apply our model
# to new measurements, we need to know whether it
# actually works--that is, whether we should trust
# its predictions

# we can't just use the data we used to build the model
# to evaluate it...it would just remember the whole
# training set and therefore always predict the
# correct label for any point in the set.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# look at data
# create dataframe from data in X_train
# label the columns using strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create scatter matrix from the dataframe; color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train,
                           figsize=(15,15), marker='o',
                           hist_kwds={'bins': 20},
                           s=20, alpha=0.8)

# KNN looks at all the datapoints and uses a distance calculation
# to determine which points are closest to it.
# The fixed 'k; determines the neighborhood of points

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# the knn object encapsulates the algorithm used
# to make predictions on new datapoints. It also
# holds info the algorithm extracts from training
# data

# to build the model on the training set, we call
# the fit method of the knn object. This takes
# the NumPy array X_train (containing the training
# data) and the NumPy array y_train (containing
# corresponding training labels) as arguments

knn.fit(X_train, y_train)

# making predictions
# we can now make predictions using this model
# on new data for which we might not know the
# correct labels.

# imagine we found an iris in the wild with
# sepal length 5cm, sepal width 2.9cm, petal
# length 1cm, and petal width 0.2cm.

# we can put this an a NumPy array
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape', X_new.shape)

# to make prediction, we call the predict() method
prediction = knn.predict(X_new)
print('Prediction:', prediction)
print('Predicted target name:',
      iris_dataset['target_names'][prediction])

# how do we know this is even right?

# Evaluating the model
# this is where the test set we created earlier
# comes in handy. This data was not used to build
# the model, but we do know the correct species
# is for each iris in the test set

# we can measure how well the model works by
# computing its accuracy

y_pred = knn.predict(X_test)
print('Test set predictions:\n', y_pred)