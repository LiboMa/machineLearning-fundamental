import numpy as np
import matplotlib.pyplot as plt
import math

inputs = [1.0, 2.0]
#weights1 = [0.2, 0.8, -0.5, 1.0]
#weights2 = [-0.2, 0.8, -0.5, 1.0]
#weights3 = [0.2, -0.8, -0.27, 1.0]
weights = [[0.2, 0.8, -0.5, 1.0],
        [-0.2, 0.8, -0.4, 1.0],
        [0.2, -0.8, -0.27, 1.0]]
#bias1 = 1.0
#bias2 = 2.0
#bias3 = 0.5
biases = [1.0, 2.0, 0.5]

# calculate of the first layer output = x*W.T+b
#output1 = inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2] * weights1[2] + inputs[3]*weights1[3] + bias1
#output2 = inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2] * weights2[2] + inputs[3]*weights2[3] + bias2
#output3 = inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2] * weights3[2] + inputs[3]*weights3[3] + bias3
#
#print("neuron 1 ->", output1)
#print("neuron 2 ->", output2)
#print("neuron 3 ->", output3)
#
#output=[output1, output2, output3]
#print("output ->", output)

layer_outputs = []

for neuron_weight, neuron_bias in zip(weights, biases):
    print("weights", neuron_weight)
    print("bias", neuron_bias)
    neuron_output = 0.0
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += n_input * weight
    neuron_output = neuron_output + neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

