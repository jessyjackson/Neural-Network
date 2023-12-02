from network import Network
from layers.fullyConnectLayer import FullyConnectLayer
from layers.activationLayer import ActivationLayer
from activation import tanh, tanhPrime
from losses import loss, lossPrime
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()

print('Train: X=%s, y=%s' % (xTrain.shape,yTrain.shape))


for i in range(1, 10):

    plt.subplot(3, 3, i)

    plt.imshow(xTrain[i], cmap=plt.get_cmap('gray'))
 
# Display the entire plot
plt.show()