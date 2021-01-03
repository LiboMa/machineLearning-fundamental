import numpy as np
import matplotlib.pyplot as plt
import math

# to fixed the random number
np.random.seed(0)

X = np.array([[1.0, 2.0, 3.0, 2.5],
    [2.0,-1.2, 1.5, 2.1],
    [0.5,-2.0, -1.5, 1.1],
    ])
#weights1 = [0.2, 0.8, -0.5, 1.0]
#weights2 = [-0.2, 0.8, -0.5, 1.0]
#weights3 = [0.2, -0.8, -0.27, 1.0]
weights = np.array([[0.2, 0.8, -0.5, 1.0],
        [-0.2, 0.8, -0.5, 1.0],
        [0.2, -0.8, -0.27, 1.0]])

weights2 = np.array([[0.2, 0.8, -0.5, 1.0],
        [-0.2, 0.8, -0.5, 1.0],
        [0.2, -0.8, -0.27, 1.0]])
#bias1 = 1.0
#bias2 = 2.0
#bias3 = 0.5
biases = np.array([1.0, 2.0, 0.5])
biases2 = np.array([-1.0, -2.0, 0.5])

# calculate of the first layer output = x*W.T+b
#output1 = inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2] * weights1[2] + inputs[3]*weights1[3] + bias1
#output2 = inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2] * weights2[2] + inputs[3]*weights2[3] + bias2
#output3 = inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2] * weights3[2] + inputs[3]*weights3[3] + bias3 #
#print("neuron 1 ->", output1)
#print("neuron 2 ->", output2)
#print("neuron 3 ->", output3)
#
#output=[output1, output2, output3]
#print("output ->", output)

# layer_outputs = []

#for neuron_weight, neuron_bias in zip(weights, biases):
#    neuron_output = 0.0
#    for n_input, weight in zip(inputs, neuron_weight):
#        neuron_output += n_input * weight
#    neuron_output = neuron_output + neuron_bias
#    layer_outputs.append(neuron_output)
#
#print(layer_outputs)
#print(inputs, weights, biases)

layer1_output = np.dot(X, weights.T) + biases
layer2_output = np.dot(X, weights2.T) + biases2

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, lr=0.01):
        self.weights = lr * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def activated(self, func='relu'):
        pass
    def activated_softMax():
        pass

number_of_neuron = 3
layer1 = Layer_Dense(4, number_of_neuron)
layer2 = Layer_Dense(number_of_neuron, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)
#print(layer1_output, "\n", layer2_output)


