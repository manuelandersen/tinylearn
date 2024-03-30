import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, 
                 lr: int = 0.01,
                 n_iters = 1000) -> None:
        
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weigths = np.random(num_features)
        self.biases = 0

    def predict(self, X):
        return np.dot(X, self.weights) + self.biases