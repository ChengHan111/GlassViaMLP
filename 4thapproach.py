import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

dataset = pd.read_csv('Glass_dataset/glass.data')
dataset.info()
dataset = shuffle(dataset)
# print(dataset)
x = dataset.iloc[:,1:10].values
x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = dataset.iloc[:,10].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=0)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,11,7,5,3,), random_state=1,max_iter=1000, learning_rate_init=0.01)
structure = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(structure)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Training set score: %f" % model.score(x_train, y_train))
print("Test set score: %f" % model.score(x_test, y_test))


