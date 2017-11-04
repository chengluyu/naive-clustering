import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_ordinary(filename):
    """
    Load ordinary data set of points. It returns a NumPy 2d-array, containing
    all points.
    """
    fp = open(filename)
    xyks = [line.strip().split(',') for line in fp.readlines()]
    return np.array([[float(x), float(y)] for x, y, k in xyks])

__all__ = ['load_ordinary', 'load_mnist']
