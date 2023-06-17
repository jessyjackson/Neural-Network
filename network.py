
import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class Nnetwork:

    def __init__(self, input_size, hidden_size, output_size,learning_rate):
        self.no_of_in_nodes = input_size
        self.no_of_hidden_nodes = hidden_size
        self.no_of_out_nodes = output_size
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):

        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,self.no_of_in_nodes))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,self.no_of_hidden_nodes))

    def run(self, input_vector):
        # Turn the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        return output_vector_network


net = Nnetwork(input_size = 2, hidden_size = 4, output_size = 4,learning_rate = 3) 
print(net.run([3,2]))