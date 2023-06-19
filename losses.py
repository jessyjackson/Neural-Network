import numpy as np

def loss(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def lossPrime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size