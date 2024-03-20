import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)  # Convert to numpy array
    y_pred = np.array(y_pred)  # Convert to numpy array
    output = np.average((y_true - y_pred)**2, axis=0)
    return output
