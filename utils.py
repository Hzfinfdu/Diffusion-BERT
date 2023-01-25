import torch
import numpy as np


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, torch.stack(ys) if ys[0] is not None else None


def word_frequency(path, basic_freq=.5):
    freq = torch.load(path)
    return basic_freq + (1 - basic_freq) * freq / freq.mean()

def min_max_norm(t, dim):
    return ((t - t.min(dim=dim, keepdims=True).values) / (t.max(dim=dim, keepdims=True).values - t.min(dim=dim, keepdims=True).values)) * 2 - 1
