from layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forwardPropagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backwardPropagation(self, output_error, learning_rate):
        inputError = np.dot(output_error, self.weights.T)
        weightsError = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weightsError
        self.bias -= learning_rate * output_error
        return inputError

        