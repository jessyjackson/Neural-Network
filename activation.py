import numpy as np

class Activation:
    def activation(x):
        return np.tanh(x)
    
    def activationPrime(x):
        return 1 - np.tanh(x)**2
