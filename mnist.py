import numpy as np
from network import Network
from layers.fullyConnectLayer import FullyConnectLayer
from layers.activationLayer import ActivationLayer
from activation import tanh, tanhPrime
from losses import loss, lossPrime
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
#from matrix to array 
xTrain = xTrain.reshape(xTrain.shape[0], 1, 28*28).astype('float32')
#from 0--255 to 0--1 for a better train
xTrain = xTrain /255

yTrain = np_utils.to_categorical(yTrain)
xTest = xTest.reshape(xTest.shape[0], 1, 28*28).astype('float32')/255
yTest = np_utils.to_categorical(yTest)

net = Network()
net.add(FullyConnectLayer(28*28,200))
net.add(ActivationLayer(tanh,tanhPrime))
net.add(FullyConnectLayer(200,100))
net.add(ActivationLayer(tanh,tanhPrime))
net.add(FullyConnectLayer(100,10))
net.add(ActivationLayer(tanh,tanhPrime))

net.setLoss(loss, lossPrime)
net.train(xTrain, yTrain, epochs=15, learningRate=0.1)

out =net.predict(xTest)
for i in range(len(out)):
    print(out[i])
    print(yTrain[i])
