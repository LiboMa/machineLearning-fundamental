import numpy as np
import matplotlib.pyplot as plt
import math

inputs = [1.0, 2.0, 3.0, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [-0.2, 0.8, -0.5, 1.0]
bias1 = 1.0
bias2 = 2.0

# calculate of the first layer output = x*W.T+b
output1 = inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2] * weights1[2] + inputs[3]*weights1[3] + bias1
output2 = inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2] * weights2[2] + inputs[3]*weights2[3] + bias2

print("neuron 1 ->", output1)
print("neuron 2 ->", output2)

output=[output1, output2]
print("output ->", output)
