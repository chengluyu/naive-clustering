import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from dataset import load_ordinary
from spectrum import spectral_cluster

DATA_FILES = [
    ('flame.txt', 2),
    ('Aggregation.txt', 7),
    ('R15.txt', 15),
    ('mix.txt', 23)
]

def palette(n):
    cmap = plt.get_cmap('gist_rainbow')
    return np.array(cmap(np.linspace(0, 1, n)))

def visualize(ax, points, labels, cluster_count):
    colors = palette(cluster_count)
    ax.scatter(points[:, 0], points[:, 1], color=colors[labels], marker='.')

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2)
    for y, x in product(range(2), range(2)):
        filename, cluster_count = DATA_FILES[y * 2 + x]
        points = load_ordinary('data/' + filename)
        print('Cluster begin:', filename)
        labels = spectral_cluster(points, cluster_count)
        print('Cluster finished:', filename)
        visualize(axes[y, x], points, labels, cluster_count)
        axes[y, x].set_title(filename)
    plt.show()

