import numpy as np
from network import Network
from layers.fullyConnectLayer import FullyConnectLayer
from layers.activationLayer import ActivationLayer
from activation import tanh, tanhPrime
from losses import loss, lossPrime

#xor problem
Xtrain = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
Ytrain = np.array([[[0]], [[1]], [[1]], [[0]]])
#                  1      1          1          0       0      0          0          0
xTry = np.array([[[0,1]], [[0,1]], [[1,0]], [[1,1]], [[0,0]], [[1,1]], [[0,0]], [1,1]])


net = Network()
net.add(FullyConnectLayer(2,3))
net.add(ActivationLayer(tanh, tanhPrime))
net.add(FullyConnectLayer(3,1))
net.add(ActivationLayer(tanh, tanhPrime))

net.setLoss(loss, lossPrime)
net.train(Xtrain, Ytrain, 1000, learningRate=0.2)

out = net.predict(Xtrain)
print(out)
out = net.predict(xTry)
print(out)