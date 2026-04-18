import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum = 0
    for i in range(len(y_true)):
        p = y_pred[i][y_true[i]]
        log_p =-np.log(p)
        sum+=log_p
        
    return sum/len(y_true)