import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.array(A)
    m, n = A.shape
    if m==n and np.linalg.det(A)!=0:
        return np.linalg.inv(A)
    else:
        return None
