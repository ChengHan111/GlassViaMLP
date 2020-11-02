import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

dataset = pd.read_csv('Glass_dataset/test/glass.csv')
dataset.info()
dataset = shuffle(dataset)
# print(dataset)
x = dataset.iloc[:, 1:10].values
# Doing the generalization when setting up the dataset
x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# Activation function:sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def sigmoid derivative for the backprop
def sigmoid_drivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Forward function
def forward(x, w1, w2, predict=False):
    a1 = np.matmul(x, w1)
    z1 = sigmoid(a1)

    #     create and add bias
    bias = np.ones((len(z1), 1))
    #  rows joining 4x5 join 4x1 == 4x6
    z1 = np.concatenate((bias, z1), axis=1)
    # print(z1.shape)
    a2 = np.matmul(z1, w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1, z1, a2, z2


# Backprop function
def backprop(a2, a1, z0, z1, z2, y):
    # Using SGD to do the derivative to MSE, making 1/2n*(z2 - y)**2 to 1/n*(z2 - y)
    # We look for the derivative equation of the Loss function
    delta2 = z2 - y
    # print(delta2)
    delta2 = delta2 * sigmoid_drivative(a2)
    Delta2 = np.matmul(z1.T, delta2)
    n = w2[1:, :].T
    m = delta2.dot(n)
    delta1 = (m) * sigmoid_drivative(a1)
    Delta1 = np.matmul(z0.T, delta1)
    return delta2, Delta1, Delta2


# init weights as random values
w1 = np.random.randn(9, 180)
# outputing 4x1, given scores
w2 = np.random.randn(181, 1)

# init learning rate
lr = 0.09

# init cost matrix for memo,since we want to plot the cost plot,
# we need to memorize all of the cost value we have
costs = []

# init epochs
epochs = 15000

m = len(x_train)

# Start training
for i in range(epochs):

    # Forward
    a1, z1, a2, z2 = forward(x_train, w1, w2)

    # Backprop
    delta2, Delta1, Delta2 = backprop(a2, a1, x_train, z1, z2, y_train)

    w1 -= lr * (1 / m) * Delta1
    w2 -= lr * (1 / m) * Delta2

    # Add costs to list for plotting
    c = np.mean(np.abs(delta2))
    # print(c)
    costs.append(c)

    if i % 1000 == 0:
        print(f"Iteration:{i}. Error: {c}")

# Training complete
print("Training Complete")

# Make predictions
z3 = forward(x_test, w1, w2, True)
print("Percentages: ")
print(z3)
print("Predictions: ")
print(np.round(z3))
z3 = np.round(z3)
# print(len(z3))
accuracy = 0
for i in range(len(z3)):
    if z3[i] == y_test[i]:
        accuracy += 1
print(accuracy / len(z3))

# Plot cost
plt.plot(costs)
plt.show()