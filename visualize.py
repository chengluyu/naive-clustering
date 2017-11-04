import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

def visualize(clusters, target=None):
    if target is None:
        _, target = plt.subplots()
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, 2 * len(clusters)))
    shuffle(colors)
    for points, color in zip(clusters, colors):
        xs, ys = zip(*points)
        target.scatter(xs, ys, color=color, marker='.')
    target.grid(True)
    plt.show()
