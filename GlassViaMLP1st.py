import numpy as np # linear algebra
import pandas as pd # data processing, reading CSV file
import theano
from keras.models import Sequential
import keras
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from theano import tensor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
seed = 7
# fix random seed for reproducibility
np.random.seed(seed)
# load pima indians dataset
dataset = pd.read_csv("Glass_dataset/glass.data")
# split the original dataset into input (X) and output (Y) variables
# pick the line data in the dataset
dataset = shuffle(dataset)
# print(dataset)
X = dataset.iloc[:,1:10].values
Y = dataset.iloc[:,10].values
Y = keras.utils.to_categorical(Y,num_classes=7)
print(Y)

# split into 50% for train and 50% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=seed)

print(X_train.shape)

# create model
model = Sequential()
# 1st approach
# add each layer to the model 77.99% accuracy
# model.add(Dense(32, input_dim=9, activation='relu')) # output matrix size (*, 32)
# model.add(Dense(24, activation='relu')) #32x24
# model.add(Dense(12, activation='relu')) #24x12
# model.add(Dense(6,  activation='relu')) #12x6
# model.add(Dense(1, activation='sigmoid')) #6x1

#2nd approach 81.12% accuracy
# model.add(Dense(64, input_dim=8, activation='sigmoid')) # output matrix size (*, 32)
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))

# 3rd approch
model.add(Dense(64, input_dim = 9, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=100)
scores = model.evaluate(X,Y)


# # Compile model
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#
# # Fit the model
# history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=400, batch_size=40)
#
# # evaluate the model
# scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()