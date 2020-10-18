import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


# Importing the dataset
dataset = pd.read_csv('Glass_dataset/glass.data')
dataset.info()
dataset = shuffle(dataset)
# print(dataset)
X = dataset.iloc[:,1:10].values
X = (X - np.min(X)) / (np.max(X) - np.min(X))
Y = dataset.iloc[:,10].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.50, random_state = 0)

# from keras.utils import to_categorical
# print(y_train)
# y_train = to_categorical(y_train)
# print(y_train)
# y_test = to_categorical(y_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
my_x = x_train
my_y = y_train
my_xx = x_test
my_yy = y_test
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 2), random_state=1,max_iter=1000)
clf.fit(my_x, my_y)
y_pred = clf.predict(my_xx)
print("Accuracy:",metrics.accuracy_score(my_yy, y_pred))