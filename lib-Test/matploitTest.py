import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(xTrain[i],cmap=plt.get_cmap('gray'))



xTrain = xTrain.reshape(xTrain.shape[0], 1, 28*28).astype('float32')
plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(xTrain[i],cmap=plt.get_cmap('gray'))
xTrain = xTrain /255
plt.figure()

for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(xTrain[i],cmap=plt.get_cmap('gray'))

for i in range(1,10):
    print(yTrain[i])
yTrain = np_utils.to_categorical(yTrain)
for i in range(1,10):
    print(yTrain[i])
plt.show()