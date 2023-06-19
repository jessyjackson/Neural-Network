from layers.layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation, activationprime):
        self.activation = activation
        self.activationPrime = activationprime

    def forwardPropagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backwardPropagation(self, output_error, learning_rate):
        return self.activationPrime(self.input) * output_error