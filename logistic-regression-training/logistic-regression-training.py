import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)

    m, n = X.shape
    # m --> Samples
    # n --> features

    w = np.zeros(n)
    b = 0

    

    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        
        weight_game = (X.T @ (p-y))/m
        bias_game = np.sum(p-y)/m

        w = w - lr * weight_game
        b = b - lr * bias_game

    return w, b