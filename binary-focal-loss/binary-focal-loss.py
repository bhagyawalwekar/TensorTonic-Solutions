import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    predictions = np.array(predictions)
    targets = np.array(targets)

    eps = 1e-9
    predictions = np.clip(predictions, eps, 1 - eps)

    # probability of the true class
    pt = np.where(targets == 1, predictions, 1 - predictions)

    # focal loss
    loss = -alpha * (1 - pt) ** gamma * np.log(pt)

    return np.mean(loss)