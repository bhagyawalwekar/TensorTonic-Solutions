import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p)
    q = np.array(q)

    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return np.sum(p * np.log(p / q))