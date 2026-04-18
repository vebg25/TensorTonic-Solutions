import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    m, n = X.shape

    w = np.zeros(n)
    b = 0
    for i in range(steps):
        z = X@w + b
        p = _sigmoid(z)

        dw = ((p-y)@X)/m
        db = np.sum(p-y)/m

        w = w - lr * dw 
        b = b - lr * db

    return w, b