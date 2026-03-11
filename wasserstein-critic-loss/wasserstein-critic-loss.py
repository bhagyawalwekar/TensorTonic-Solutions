import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores = np.array(real_scores)
    fake_scores = np.array(fake_scores)

    loss = np.mean(fake_scores) - np.mean(real_scores)

    return loss