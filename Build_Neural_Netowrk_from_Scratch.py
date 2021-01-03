# follow the link -> https://anderfernandez.com/en/blog/how-to-code-neural-network-from-scratch-in-python/#:~:text=Activation%20functoin%3A%20ReLu%20function,deep%20learning%20and%20neural%20network.

# in out case,  we will use neural network to solve a classification problem with two classes
# to do so,  we will create a small neural network with 4 layers , that will have the following:
#  1. A Input layer with two neurons, as we will  use two variables
#  2. 2 Hidden Layers with 4 and 8 neurons respectively
#  3. One output layer with just one neuron. (justify the reuslt)

import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# create neuron layers
from scipy import stats

'''
    1. A layer receives inputs. On the first layer, the inputs will be the data itself and that is why it is called the input layer. On the rest of the layers, the input will be the result of the previous layer.
    2. The input values of the neuron are weighted by a matrix, called W (weights). This matrix has as many rows as neurons in the previous and as many columns as neurons in this layer.
    3. To the result from the weighted sum, another parameter is added, named bias (b). In this case, every neuron has each one bias, so for a given layer, there will be as many biases as neurons are in that layer.
    4. Finally, we will apply a function to the previous, knows as the activation function. You might have realized that until now we have only make linear transformations of the data. The activation function will ensure the non-linearity of the model. If we wouldn’t apply an activation function, any neural network would indeed be a linear regression. The result after applying the activation function will be the result of the neuron.

Explanation: b, W, activation func.
    b -> bias parameters
    W -> weighted vector, the second parameter sets
    activation -> X

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
'''

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)

# Hidden Layer -> ReLu
def derivada_relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

relu = (
  lambda x: x * (x > 0),
  lambda x:derivada_relu(x)
  )

# Output Layer -> Sigmoid
sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )


# Number of neurons in each layer 
# The first value is equal to the number of predictors.
neuronas = [2,4,8,1] 

# Activation functions to use in each layer.
funciones_activacion = [relu, relu, sigmoid]

# Store all result in the red neorual
red_neuronal = []

# initialize all x result for each layers:w

for paso in range(len(neuronas)-1):
  print(neuronas[paso], neuronas[paso+1])
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  # activation functions, 0 - relu, 1 - relu, 2 - sigmoid
  red_neuronal.append(x)

#print(red_neuronal[0].b, red_neuronal[0].W)
# implment z = Wx + b
X =  np.round(np.random.randn(20,2),3) # Example of an input vector

#print(X, X.shape)
# first layer value of multiply the input values
z = X @ red_neuronal[0].W
#print(red_neuronal[0].W)
#print(red_neuronal[0].b)

#print(z[:10,:], X.shape, z.shape)
print("Sum of Vector W and x :\n", z[:5, :], X.shape, z.shape)
z = z + red_neuronal[0].b
print("Sum of Vector W and b :\n", z[:5,:])

# apply the activation func Rule for input layer

a = red_neuronal[0].funcion_act[0](z)
print("Apply activation function \n", a[:5,:])


## Apply for the all layers
output = [X]

for num_capa in range(len(red_neuronal)):
    z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
    a = red_neuronal[num_capa].funcion_act[0](z)
    print("=====>", output)
    output.append(a)

print("values of activation functions", output[-1], "len of Vector a", len(output[-1]), "\nlen of Matrix output",len(output))

# train the neural networks

# create Cost function  -> In order to train or improve our neural network we first need to know how much it has missed.

# Explanation Using MSE, The MSE is quite simple to calculate: 
# you subtract the real value from every prediction, square it, and calculate its square root

def mse(Ypredich, Yreal):
    # calcula mos el error
    x = (np.array(Ypredich) - np.array(Yreal)) **2
    x = np.mean(x)
    
    # calculate the deviation between x and y
    y = np.array(Ypredich) - np.array(Yreal)
    
    return(x, y)


from random import shuffle

Y = [0] * 10 + [1] * 10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)
 
print(Y)
predict = mse(output[-1], Y)

print("deviation of cost function", predict[0])


### Back-propagation: calculating the error in last layer

# check the values
red_neuronal[-1].b
red_neuronal[-1].W
##
a = output[-1]
## 
x = mse(a,Y)[1] * red_neuronal[-2].funcion_act[1](a)
#
#print(x)
#
## apply gradicent descent for last layer 
#
red_neuronal[-1].b = red_neuronal[-1].b - x.mean() * 0.01
red_neuronal[-1].W = red_neuronal[-1].W - (output[-1].T @ x) * 0.01
#
print("b of the last layer's result", red_neuronal[-1].b)
print("W of the last layer's result", red_neuronal[-1].W)
#
sys.exit(0)

## put all calculate together with for each layer

## learning rate
lr = 0.05

# We create the inverted index
back = list(range(len(output)-1))
back.reverse()
#print(back)

# create a vector where we will store the errors of each layer
delta = []

for capa in back:
    a = output[capa+1][1]
    if capa == back[0]:
        x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)
        print("-->", x)
        print(delta)
    else:
        print(W_temp)
        x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)
    # We store the values of W in order to use them on the next iteration
    W_temp = red_neuronal[capa].W.transpose()

    # Gradient Descent #

    # Adjust the values of two parameters 
    red_neuronal[capa].b = red_neuronal[capa].b - delta[-1].mean() * lr
    #print("--->", delta[-1])
    red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr


print('MSE: ' + str(mse(output[-1],Y)[0]) )
print('Estimation: ' + str(output[-1]) )


## trying out neural network 

import random

def circulo(num_datos = 100,R = 1, minimo = 0,maximo= 1):
  pi = math.pi
  r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size= num_datos)) * 10
  theta = stats.truncnorm.rvs(minimo, maximo, size= num_datos) * 2 * pi *10

  x = np.cos(theta) * r
  y = np.sin(theta) * r

  y = y.reshape((num_datos,1))
  x = x.reshape((num_datos,1))

  #We reduce the number of elements so that there is no overflow
  x = np.round(x,3)
  y = np.round(y,3)

  df = np.column_stack([x,y])
  return(df)

# create two ramdon data sets
datos_1 = circulo(num_datos = 150, R = 2)
datos_2 = circulo(num_datos = 150, R = 0.5)

X = np.concatenate([datos_1,datos_2])
# get 3 decimal from the value
X = np.round(X,3)
print(X)

Y = [0] * 150 + [1] * 150
print(Y)
#X = np.round(X,3)
Y = np.array(Y).reshape(len(Y),1)
print("reshape Y", Y)

#plt.cla()
#plt.scatter(X[0:150,0],X[0:150,1], c = "b")
#plt.scatter(X[150:300,0],X[150:300,1], c = "r")
#plt.show()

# train data
def entrenamiento(X,Y, red_neuronal, lr = 0.01):

  output = [X]

  for num_capa in range(len(red_neuronal)):
    z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b

    a = red_neuronal[num_capa].funcion_act[0](z)

    output.append(a)

  # Backpropagation

  back = list(range(len(output)-1))
  back.reverse()


  delta = []

  for capa in back:
    # Backprop #delta

    a = output[capa+1]

    if capa == back[0]:
      x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
      delta.append(x)

    else:
      x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
      delta.append(x)

    W_temp = red_neuronal[capa].W.transpose()

    # Gradient Descent #
    red_neuronal[capa].b = red_neuronal[capa].b - np.mean(delta[-1], axis = 0, keepdims = True) * lr
    red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr

  return output[-1]


# re-define the training class
class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)

neuronas = [2,4,8,1] 
funciones_activacion = [relu,relu, sigmoid]
red_neuronal = []

for paso in list(range(len(neuronas)-1)):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x) 

error = []
predicciones = []

for epoch in range(0,200):
  ronda = entrenamiento(X = X ,Y = Y ,red_neuronal = red_neuronal, lr = 0.001)
  predicciones.append(ronda)
  temp = mse(np.round(predicciones[-1]),Y)[0]
  error.append(temp)

epoch = list(range(0,200))
plt.plot(epoch, error)
plt.show()

