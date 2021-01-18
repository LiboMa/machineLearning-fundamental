#!/usr/bin/env python 

inputs = [1.0, 2.0, 3.0, 4.0]
weights = [0.1, 0.2, 0.3, 0.4]
bias = [1]

def dot_product(m1, m2):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i]*weights[i]
    return output


output = dot_product(inputs, weights) + bias[0]
print(output)
