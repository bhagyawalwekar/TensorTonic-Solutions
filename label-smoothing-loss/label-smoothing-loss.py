import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    predictions = np.array(predictions)
    K = len(predictions)

    # create smoothed labels
    smoothed = np.full(K, epsilon / K)
    smoothed[target] += (1 - epsilon)

    # cross entropy loss
    loss = -np.sum(smoothed * np.log(predictions))

    return loss