import numpy as np
import matplotlib.pyplot as plt
import math
# follow the link -> https://cs231n.github.io/neural-networks-case-study/

# to fixed the random number
#np.random.seed(0)

#X = np.array([[1.0, 2.0, 3.0, 2.5],
#    [2.0,-1.2, 1.5, 2.1],
#    [0.5,-2.0, -1.5, 1.1],
#    ])
#

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, lr=0.01):
        self.weights = lr * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.output = np.exp(inputs - np.max(inputs))

number_of_neuron = 2

# generate 3 classification points in the image
X, y = create_data(100, 3)
data_sets, labels = create_data(100, 3)
print(data_sets.shape[1])

import matplotlib.pyplot as plt

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.scatter(data_sets[:, 0], data_sets[:, 1], c=labels, s=40, cmap="brg")
#plt.show()

# data_sets has (300, 2), 2 features for each data set
layer1 = Layer_Dense(data_sets.shape[1], number_of_neuron)

activation1 = Activation_ReLU()

layer1.forward(data_sets)
#print(layer1.output)
# make each value to positive e.g. > 0
activation1.forward(layer1.output)

#print(activation1.output)

layer_outputs = [
        [4.8, 1.21, 2.385],
        [8.9, -1.81, 0.2],
        [1.41, -1.051, -0.026]
        ]

# E = 2.71282182846
E = math.e 
exp_values = np.exp(layer_outputs - np.max(layer_outputs))
print(exp_values)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)


