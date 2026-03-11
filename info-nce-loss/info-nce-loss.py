import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)

    # similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # numerical stability trick
    S = S - np.max(S, axis=1, keepdims=True)

    exp_S = np.exp(S)

    pos = np.diag(exp_S)
    denom = np.sum(exp_S, axis=1)

    loss = -np.log(pos / denom)

    return np.mean(loss)