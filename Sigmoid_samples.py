import numpy as np
import math
import matplotlib.pyplot as plt


# define sigmoid 

## Usage -> simoid[0](x) func 1, sigmoid[1](x) func 2
sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )

#  create a random line space
rango1 = np.linspace(-10,10)
rango = np.linspace(-10,10).reshape([50,1])
print(rango, len(rango))
#print(rango1, len(rango1))

# input value to sigmoid func 1
datos_sigmoide = sigmoid[0](rango)
# input value to sigmoid func 2
datos_sigmoide_derivada = sigmoid[1](rango)

# create the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))

axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)

fig.tight_layout()
fig.show()
