import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    loss = np.maximum(0, margin - y_true * y_score)

    if reduction == "sum":
        return np.sum(loss)
    return np.mean(loss)