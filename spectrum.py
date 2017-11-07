import numpy as np
import numpy.linalg as linalg
import scipy.sparse.linalg
import affinity

from sklearn.cluster import KMeans

def kmeans(points, cluster_count):
    kmeans = KMeans(n_clusters=cluster_count)
    return kmeans.fit(points).labels_

def laplacian(A):
    D = np.zeros(A.shape)
    D.flat[::A.shape[0] + 1] = np.sum(A, axis=0) ** (-0.5)
    return D.dot(A).dot(D)

def spectral_cluster(points, cluster_count):
    L = laplacian(affinity.local_scaling(points))
    eigval, eigvec = scipy.sparse.linalg.eigs(L, cluster_count)
    X = eigvec.real
    rows_norm = linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    return kmeans(Y, cluster_count)
