import numpy as numpy
import numpy.linalg as lin
from itertools import product


def gaussian_kernel(u, v, sigma_i=0.8, sigma_j=1.0):
    return numpy.exp(- lin.norm(u - v) ** 2 / (2 * sigma_i * sigma_j))

def ordinary_affinity(points):
    return numpy.array([[gaussian_kernel(u, v) for u in points] for v in points])

def remove_deep_gap(dists):
    gaps = dists[1:] - dists[:-1]
    i = numpy.argmax(dists)
    return dists[:i + 1]

def local_scaling(points):
    n = points.shape[0]
    sigma = []
    for u in points:
        dists = [lin.norm(u - v) for v in points]
        dists.sort()
        sigma.append(numpy.mean(remove_deep_gap(numpy.array(dists[:7]))))
    ans = numpy.zeros((n, n))
    for i, j in product(range(n), range(n)):
        ans[i, j] = gaussian_kernel(points[i], points[j], sigma[i], sigma[j])
    return ans
