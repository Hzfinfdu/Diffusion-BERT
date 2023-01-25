import torch
import utils
import numpy as np


def kl_divergence_with_probs(p=None, q=None, epsilon=1e-20):
    """Compute the KL between two categorical distributions from their probabilities.
    Args:
    p: [..., dim] array with probs for the first distribution.
    q: [..., dim] array with probs for the second distribution.
    epsilon: a small float to normalize probabilities with.
    Returns:
    an array of KL divergence terms taken over the last axis.
    """
    kl = (p * (torch.log(p + epsilon) - torch.log(q + epsilon))).sum(-1)
    ## KL divergence should be positive, this helps with numerical stability
    loss = torch.nn.functional.relu(kl)
    return loss


def cross_entropy_with_probs(probs, targets, epsilon=1e-20):
    """Compute cross entropy for a given distribution and targets.
    Cross entropy is taken over the last axis. Remaining axes are unchanged.
    Args:
    probs: [..., length, num_classes] float array.
    targets: categorical targets [..., length] int array.
    label_smoothing: label smoothing constant, used to determine the on and off
     values.
    epsilon: small noise to add to probs when converting to log space.
    Returns:
    Array with loss taken over the last axis.
    """
    assert probs.size()[:-1] == targets.size(), "Logits shape must agree with targets, except in the last dimension."

    # vocab_size = probs.size(-1)

    # soft_targets = torch.nn.functional.one_hot(targets, vocab_size)

    probs = torch.nn.functional.relu(probs)  # help with numerical stability
    loss = -(torch.log(probs + epsilon).gather(-1, targets.unsqueeze(-1))).squeeze(-1)
    return loss

