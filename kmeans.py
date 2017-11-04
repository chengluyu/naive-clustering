import numpy as np
from numpy.random import shuffle

def randomize_centroids(points, cluster_count):
    copy = points.copy()
    shuffle(copy)
    return copy[:cluster_count]

def closest_centroids(points, centroids):
    return np.argmin(np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2)), axis=0)

def move_centroids(points, closest, centroids):
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

def cluster(points, cluster_count):
    centroids = randomize_centroids(points, cluster_count)
    while True:
        closest = closest_centroids(points, centroids)
        centroids = move_centroids(points, closest, centroids)
        new_closest = closest_centroids(points, centroids)
        if (new_closest == closest).all():
            break
        closest = new_closest
    return [points[closest == k] for k in range(cluster_count)]

__all__ = ['cluster']
